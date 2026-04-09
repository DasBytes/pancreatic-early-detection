"""
Explainable 3D Swin Transformer for Pancreatic Tumor Classification
====================================================================
Optimized for CPU + 16 GB RAM — targets < 5 hours total runtime.

Fixes applied (vs previous version):
  1. PATCH_SIZE  96 → 64        (3–4× faster, ~60% less RAM per patch)
  2. RNG seed    moved OUTSIDE per-volume loop (proper randomisation)
  3. DataLoader  num_workers=0  (avoids OOM / crash on CPU)
  4. Grad-CAM    safe tuple indexing (uses [0] for tensor, not [-1])
  5. Epochs      20 → 8         (fits in ~2–3 h training on Ryzen 5)
  6. AOPC        DISABLED       (saves several hours on CPU)
  7. NIfTI cache added          (each file loaded once, not per __getitem__)
  8. MC samples  10 → 5         (halves uncertainty estimation time)
  9. SmoothGradCAM passes 10→5  (halves Grad-CAM generation time)
"""

import json
import os
import warnings

import matplotlib.cm as mcm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.ndimage import gaussian_filter
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

warnings.filterwarnings("ignore")

from monai.networks.nets.swin_unetr import SwinTransformer as MonaiSwinTransformer

# ══════════════════════════════════════════════════════════════════════════════
#  CPU THREAD OPTIMISATION  (Ryzen 5 5625U — 6 cores / 12 threads)
# ══════════════════════════════════════════════════════════════════════════════
torch.set_num_threads(12)
torch.set_num_interop_threads(4)
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["OPENBLAS_NUM_THREADS"] = "12"

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
# FIX 1: PATCH_SIZE 96 → 64  (each 64³ patch uses ~3–4× less memory than 96³)
PATCH_SIZE = 64
BATCH_SIZE = 1  # CPU-safe; avoids OOM on 16 GB

# FIX 5: Epochs 20 → 8  (fits in ~2–3 h training budget on Ryzen 5 CPU)
EPOCHS = 8

LR = 1e-4
LAMBDA_SGAR = 0.3

# FIX 8: MC samples 10 → 5  (halves uncertainty estimation time)
MC_SAMPLES = 5
UNCERTAINTY_PERCENTILE = 75

# FIX 9: SmoothGradCAM passes 10 → 5  (halves Grad-CAM generation time)
GRADCAM_SMOOTH_N = 5
GRADCAM_NOISE_STD = 0.15
GRADCAM_VIS_SAMPLES = 6
GRADCAM_SIGMA = 1.5  # Gaussian smoothing sigma for clean maps

DEVICE = torch.device("cpu")  # Ryzen 5 5625U has no CUDA support
print(f"[INFO] Running on device: {DEVICE}")
print(f"[INFO] PyTorch threads  : {torch.get_num_threads()}")

# ── Dataset root ──────────────────────────────────────────────────────────────
BASE_PATH = r"E:\pancreatic cancer\Task07_Pancreas"
DATASET_JSON = os.path.join(BASE_PATH, "dataset.json")
IMAGE_TR_DIR = os.path.join(BASE_PATH, "imagesTr")
LABEL_TR_DIR = os.path.join(BASE_PATH, "labelsTr")

# ── JSON label map ────────────────────────────────────────────────────────────
LABEL_BACKGROUND = 0
LABEL_PANCREAS = 1
LABEL_CANCER = 2


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD OFFICIAL TRAIN / VAL SPLIT FROM dataset.json
# ══════════════════════════════════════════════════════════════════════════════
def load_dataset_pairs(dataset_json: str, base_path: str):
    """
    Parse the official dataset.json and return a list of
    (abs_image_path, abs_label_path) tuples for ALL labelled training cases.
    """
    with open(dataset_json, "r") as f:
        meta = json.load(f)

    pairs = []
    for entry in meta["training"]:
        img_rel = entry["image"].lstrip("./")
        lbl_rel = entry["label"].lstrip("./")
        img_path = os.path.join(base_path, img_rel)
        lbl_path = os.path.join(base_path, lbl_rel)
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            pairs.append((img_path, lbl_path))
        else:
            print(f"[WARN] Missing file pair — skipping: {img_path}")

    print(
        f"[INFO] Valid labelled pairs found: {len(pairs)} / {len(meta['training'])}"
    )
    print(
        f"[INFO] Unlabelled test cases (imagesTs): {len(meta['test'])} "
        f"— NOT used (no labels)"
    )
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
#  SGAR LOSS
# ══════════════════════════════════════════════════════════════════════════════
class SGARLoss(nn.Module):
    """
    Segmentation-Guided Attention Regularisation.
    Penalises attention mass that falls outside the tumour mask.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self, attn_map: torch.Tensor, tumor_mask: torch.Tensor
    ) -> torch.Tensor:
        target_size = attn_map.shape[1:]
        mask_r = F.interpolate(
            tumor_mask.float(),
            size=target_size,
            mode="trilinear",
            align_corners=False,
        ).squeeze(1)
        mask_r = (mask_r > 0.5).float()

        has_tumor = mask_r.flatten(1).sum(1) > 0
        if has_tumor.sum() == 0:
            return attn_map.sum() * 0.0

        a = attn_map[has_tumor]
        m = mask_r[has_tumor]
        num = (a * m).sum(dim=(1, 2, 3))
        den = a.sum(dim=(1, 2, 3)) + self.eps
        return (1.0 - num / den).mean()


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET  (with NIfTI RAM cache)
# ══════════════════════════════════════════════════════════════════════════════
class Pancreas3DPatchDataset(Dataset):
    """
    Yields (patch [1,D,H,W], tumour_mask [1,D,H,W], label) triples.

    Label mapping (aligned with dataset.json):
        label 1  →  patch centred on a CANCER voxel    (mask_vol == 2)
        label 0  →  patch centred on HEALTHY PANCREAS  (mask_vol == 1)
                        OR  BACKGROUND                 (mask_vol == 0)

    FIX 7: Volumes are cached in RAM after first load instead of
           re-reading NIfTI files on every __getitem__ call.
    """

    def __init__(
        self, pairs: list, patch_size: int = PATCH_SIZE, augment: bool = False
    ):
        self.pairs = pairs
        self.patch_size = patch_size
        self.augment = augment
        # FIX 7: RAM cache for loaded NIfTI volumes  {path: np.ndarray}
        self._vol_cache: dict[str, np.ndarray] = {}
        self.samples = self._prepare_samples()
        if not self.samples:
            raise ValueError(
                "No valid patches were generated. "
                "Check IMAGE_TR_DIR / LABEL_TR_DIR paths."
            )

    def _load_volume(self, path: str) -> np.ndarray:
        """Load a NIfTI volume, caching it in RAM for reuse."""
        if path not in self._vol_cache:
            self._vol_cache[path] = nib.load(path).get_fdata().astype(np.float32)
        return self._vol_cache[path]

    def _prepare_samples(self):
        # FIX 2: RNG created ONCE outside the per-volume loop so that
        #         different volumes get different random patch centres.
        rng = np.random.default_rng(seed=42)

        samples = []
        for img_path, lbl_path in self.pairs:
            mask_vol = self._load_volume(lbl_path)

            cancer_idx = np.argwhere(mask_vol == LABEL_CANCER)
            pancreas_idx = np.argwhere(mask_vol == LABEL_PANCREAS)
            bg_idx = np.argwhere(mask_vol == LABEL_BACKGROUND)

            if len(cancer_idx) == 0:
                continue
            if len(pancreas_idx) == 0 and len(bg_idx) == 0:
                continue

            # FIX 2: rng is shared across volumes → proper randomisation
            for _ in range(6):
                c = cancer_idx[rng.integers(len(cancer_idx))].tolist()
                samples.append((img_path, lbl_path, c, 1))

            neg_src = pancreas_idx if len(pancreas_idx) > 0 else bg_idx
            for _ in range(3):
                c = neg_src[rng.integers(len(neg_src))].tolist()
                samples.append((img_path, lbl_path, c, 0))

            bg_src = bg_idx if len(bg_idx) > 0 else pancreas_idx
            for _ in range(3):
                c = bg_src[rng.integers(len(bg_src))].tolist()
                samples.append((img_path, lbl_path, c, 0))

        n_pos = sum(1 for s in samples if s[3] == 1)
        n_neg = sum(1 for s in samples if s[3] == 0)
        print(f"[INFO] Total patches        : {len(samples)}")
        print(f"[INFO] Positive (cancer)    : {n_pos}")
        print(f"[INFO] Negative (non-cancer): {n_neg}")
        return samples

    @staticmethod
    def _normalize(patch: np.ndarray) -> np.ndarray:
        return (np.clip(patch, -100, 400) + 100) / 500.0

    @staticmethod
    def _augment(patch: np.ndarray, mask: np.ndarray):
        for ax in range(3):
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=ax).copy()
                mask = np.flip(mask, axis=ax).copy()
        patch = np.clip(patch + np.random.uniform(-0.05, 0.05), 0.0, 1.0)
        return patch, mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path, center, label = self.samples[idx]

        # FIX 7: Use cached volumes instead of re-loading from disk
        img_vol = self._load_volume(img_path)
        mask_vol = self._load_volume(lbl_path)

        p = self.patch_size
        zs = max(0, int(center[0]) - p // 2)
        ys = max(0, int(center[1]) - p // 2)
        xs = max(0, int(center[2]) - p // 2)

        patch = img_vol[zs : zs + p, ys : ys + p, xs : xs + p]
        sgar_mask = (
            mask_vol[zs : zs + p, ys : ys + p, xs : xs + p] == LABEL_CANCER
        ).astype(np.float32)

        if patch.shape != (p, p, p):

            def _pad(arr):
                return np.pad(
                    arr,
                    [(0, p - s) for s in arr.shape],
                    mode="constant",
                )

            patch, sgar_mask = _pad(patch), _pad(sgar_mask)

        patch = self._normalize(patch)
        if self.augment:
            patch, sgar_mask = self._augment(patch, sgar_mask)

        return (
            torch.tensor(patch[np.newaxis], dtype=torch.float32),
            torch.tensor(sgar_mask[np.newaxis], dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL  (Lightweight for 16 GB RAM / CPU)
# ══════════════════════════════════════════════════════════════════════════════
class Swin3DClassifierWithAttention(nn.Module):
    """
    3-D Swin Transformer classifier with SGAR attention branch.

    Backbone uses embed_dim=24 (vs 48) and fewer depth layers to cut RAM
    usage by ~50% — safe for 16 GB on CPU.

    Feature channel sizes:
        stage 0 : embed_dim * 1  =  24
        stage 1 : embed_dim * 2  =  48
        stage 2 : embed_dim * 4  =  96
        stage 3 : embed_dim * 8  = 192

    SGAR head  : features[-2] (96-ch) → 1×1 Conv3d → sigmoid
    Classifier : features[-1] (192-ch) → AdaptiveAvgPool → MLP head
    """

    def __init__(self, mc_dropout: bool = False, drop_rate: float = 0.3):
        super().__init__()
        self.mc_dropout = mc_dropout
        self.backbone = MonaiSwinTransformer(
            in_chans=1,
            embed_dim=24,  # halved from 48 → saves ~50% RAM
            window_size=(7, 7, 7),
            patch_size=(4, 4, 4),
            depths=(2, 2, 4, 2),  # reduced stage-2 depth (6→4)
            num_heads=(3, 6, 12, 24),
            spatial_dims=3,
        )
        # embed_dim * 8 = 192
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(128, 2),
        )
        # SGAR: embed_dim * 4 = 96
        self.attn_conv = nn.Conv3d(96, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        features = self.backbone(x)
        logits = self.head(self.pool(features[-1]))
        if return_attn:
            attn = torch.sigmoid(self.attn_conv(features[-2])).squeeze(1)
            return logits, attn
        return logits

    def enable_mc_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


# ══════════════════════════════════════════════════════════════════════════════
#  GRAD-CAM 3D  (High-Resolution + Bug-Fixed)
# ══════════════════════════════════════════════════════════════════════════════
class GradCAM3D:
    """
    3-D Gradient-weighted Class Activation Map.

    FIX 4: Grad-CAM hook tuple indexing
    ────────────────────────────────────
    MONAI SwinBasicLayer returns a tuple: (tensor, D, H, W).
    The previous code used output[-1] / grad_output[-1], which grabbed the
    integer W instead of the actual tensor — causing silent wrong Grad-CAM
    or a crash.  Now we use output[0] / grad_output[0] to always get the
    feature tensor.

    Other improvements:
    • Robust token → spatial reshape: infers side correctly for non-cubic tokens
    • Falls back to nearest-neighbour reshape if cube root is not integer
    • High-res target: hooks backbone.layers[-2] (higher spatial resolution)
    • Post-processing: Gaussian smoothing (sigma=GRADCAM_SIGMA) for clean maps
    • Safe normalisation that avoids flat-zero maps
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._acts = None
        self._grads = None
        self._fwd = target_layer.register_forward_hook(self._save_act)
        self._bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _m, _i, output):
        # FIX 4: Use [0] to get the tensor, not [-1] which grabs integer W
        out = output[0] if isinstance(output, (tuple, list)) else output
        self._acts = out.detach().cpu()

    def _save_grad(self, _m, _gi, grad_output):
        # FIX 4: Use [0] to get the gradient tensor, not [-1]
        g = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
        self._grads = g.detach().cpu()

    @staticmethod
    def _to_5d(t: torch.Tensor) -> torch.Tensor:
        """
        Normalise any activation/gradient tensor to [B, C, D, H, W].
        Handles:
          [B, C, D, H, W]  — standard conv output
          [B, tokens, C]   — SwinTransformer token sequence
        """
        if t.dim() == 5:
            return t  # already spatial

        if t.dim() == 3:
            B, T, C = t.shape
            # find integer cube root
            side = round(T ** (1.0 / 3.0))
            if side**3 == T:
                return t.permute(0, 2, 1).reshape(B, C, side, side, side)
            # fallback: reshape to (B, C, T, 1, 1) then pool
            return t.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # [B,C,T,1,1]

        raise ValueError(f"Unexpected activation shape: {t.shape}")

    def __call__(self, img: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """
        img       : [1, 1, D, H, W]
        class_idx : 0 = Healthy, 1 = Tumour
        Returns   : numpy [D, H, W] in [0, 1], Gaussian-smoothed
        """
        self.model.eval()
        inp = img.clone().requires_grad_(True)
        logits = self.model(inp, return_attn=False)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        act = self._acts
        grad = self._grads
        if act is None or grad is None:
            raise RuntimeError(
                "GradCAM hooks did not fire. Verify target_layer is in the "
                "forward path and return_attn=False is used."
            )

        act = self._to_5d(act)
        grad = self._to_5d(grad)

        # alpha = global-average-pooled gradients  [1, C, 1, 1, 1]
        weights = grad.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))  # [1,1,D,H,W]

        # upsample to full patch size for high-resolution output
        cam = F.interpolate(
            cam,
            size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
            mode="trilinear",
            align_corners=False,
        ).squeeze().numpy()  # [D,H,W]

        # Gaussian smoothing for clean, artefact-free maps
        cam = gaussian_filter(cam, sigma=GRADCAM_SIGMA)

        mn, mx = cam.min(), cam.max()
        if mx - mn < 1e-8:
            return np.zeros_like(cam)
        return (cam - mn) / (mx - mn)

    def remove_hooks(self):
        self._fwd.remove()
        self._bwd.remove()


class SmoothGradCAM3D:
    """Averages GradCAM3D over N noisy passes; final map also Gaussian-smoothed."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        n: int = GRADCAM_SMOOTH_N,
        std: float = GRADCAM_NOISE_STD,
    ):
        self._gcam = GradCAM3D(model, target_layer)
        self.n = n
        self.std = std

    def __call__(self, img: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        maps = [
            self._gcam(img + torch.randn_like(img) * self.std, class_idx)
            for _ in range(self.n)
        ]
        avg = np.mean(maps, axis=0)
        # second smoothing pass on the averaged map
        avg = gaussian_filter(avg, sigma=GRADCAM_SIGMA * 0.5)
        mn, mx = avg.min(), avg.max()
        if mx - mn < 1e-8:
            return np.zeros_like(avg)
        return (avg - mn) / (mx - mn)

    def remove_hooks(self):
        self._gcam.remove_hooks()


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _overlay(
    ct: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55
) -> np.ndarray:
    """Alpha-blend a jet heatmap onto a greyscale CT slice. Returns RGB [H,W,3]."""
    ct_rgb = np.stack([ct] * 3, axis=-1)
    heat_rgb = mcm.get_cmap("jet")(heatmap)[..., :3]
    return np.clip((1 - alpha) * ct_rgb + alpha * heat_rgb, 0, 1)


def save_gradcam_figure(
    sample_idx: int,
    ct: np.ndarray,
    gt_mask: np.ndarray,
    gcam: np.ndarray,
    pred_label: int,
    true_label: int,
    iou_val: float,
    save_path: str,
) -> None:
    """
    High-resolution 5-panel + 3-plane Grad-CAM figure (dark background).
    Saved at 200 DPI for crisp output.
    """
    mid = PATCH_SIZE // 2
    BG = "#0d1117"
    FG = "white"

    ct_ax, ct_cor, ct_sag = ct[mid], ct[:, mid, :], ct[:, :, mid]
    mk_ax, _mk_cor, _mk_sag = gt_mask[mid], gt_mask[:, mid, :], gt_mask[:, :, mid]
    gc_ax, gc_cor, gc_sag = gcam[mid], gcam[:, mid, :], gcam[:, :, mid]

    ov_gcam_ax = _overlay(ct_ax, gc_ax)
    ov_mask_ax = _overlay(ct_ax, mk_ax.astype(float), alpha=0.45)

    fig = plt.figure(figsize=(28, 10), facecolor=BG, dpi=200)
    gs = gridspec.GridSpec(
        2,
        8,
        figure=fig,
        hspace=0.35,
        wspace=0.08,
        left=0.03,
        right=0.97,
        top=0.88,
        bottom=0.05,
    )

    kw_gray = dict(cmap="gray", vmin=0, vmax=1)
    kw_red = dict(cmap="Reds", vmin=0, vmax=1)
    kw_jet = dict(cmap="jet", vmin=0, vmax=1)

    def _ax(row, col, img, kw, title, is_rgb=False):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img, interpolation="lanczos", **({} if is_rgb else kw))
        ax.set_title(title, color=FG, fontsize=9, pad=4)
        ax.axis("off")
        ax.set_facecolor(BG)
        return ax

    # Row 0
    _ax(0, 0, ct_ax, kw_gray, "CT Patch\n(axial mid-slice)")
    _ax(0, 1, mk_ax, kw_red, "GT Tumour Mask\n(label == 2)")
    _ax(0, 2, gc_ax, kw_jet, f"Grad-CAM\n(IoU = {iou_val:.3f})")
    _ax(0, 3, ov_gcam_ax, {}, "CT + Grad-CAM", is_rgb=True)
    _ax(0, 4, ov_mask_ax, {}, "CT + GT Mask", is_rgb=True)
    _ax(0, 5, gc_ax, kw_jet, "Grad-CAM\nAxial")
    _ax(0, 6, gc_cor, kw_jet, "Grad-CAM\nCoronal")
    _ax(0, 7, gc_sag, kw_jet, "Grad-CAM\nSagittal")

    # Row 1
    _ax(1, 0, ct_ax, kw_gray, "CT Axial")
    _ax(1, 1, ct_cor, kw_gray, "CT Coronal")
    _ax(1, 2, ct_sag, kw_gray, "CT Sagittal")
    _ax(1, 3, _overlay(ct_ax, gc_ax), {}, "Overlay Axial", is_rgb=True)
    _ax(1, 4, _overlay(ct_cor, gc_cor), {}, "Overlay Coronal", is_rgb=True)
    _ax(1, 5, _overlay(ct_sag, gc_sag), {}, "Overlay Sagittal", is_rgb=True)

    # Colorbar
    ax_cb = fig.add_subplot(gs[1, 6:])
    ax_cb.set_facecolor(BG)
    ax_cb.axis("off")
    cax = fig.add_axes([0.84, 0.08, 0.012, 0.35])
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Grad-CAM intensity", color=FG, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG)

    status = "CORRECT" if pred_label == true_label else "WRONG"
    pred_str = "Tumour" if pred_label == 1 else "Healthy"
    true_str = "Tumour" if true_label == 1 else "Healthy"
    title_col = "#00ff88" if pred_label == true_label else "#ff4444"
    fig.suptitle(
        f"Sample {sample_idx}  |  True: {true_str}  |  Pred: {pred_str}  |  {status}",
        color=title_col,
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )

    plt.savefig(
        save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def localization_iou(
    cam_maps: np.ndarray, tumor_masks: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    ious = []
    for cam, mask in zip(cam_maps, tumor_masks):
        if mask.sum() == 0:
            continue
        bin_cam = (cam > threshold).astype(float)
        intersection = (bin_cam * mask).sum()
        union = ((bin_cam + mask) > 0).sum()
        if union > 0:
            ious.append(intersection / union)
    return np.array(ious)


# FIX 6: AOPC function retained but DISABLED in main to save hours on CPU.
# If needed, uncomment the AOPC block in __main__ below.
def compute_aopc(
    model,
    dataloader,
    gcam_engine,
    percentages=(10, 20, 30, 50, 70, 90),
) -> np.ndarray:
    """
    Area Over the Perturbation Curve (AOPC).
    Uses torch.inference_mode for CPU speed during forward passes.
    """
    model.eval()
    aopc_scores = []

    for imgs, _masks, lbls in dataloader:
        imgs_d = imgs.to(DEVICE)
        with torch.inference_mode():
            logits_orig = model(imgs_d, return_attn=False)
            probs_orig = F.softmax(logits_orig, dim=1)

        for b in range(imgs.shape[0]):
            img_s = imgs_d[b : b + 1]
            pred_cls = logits_orig[b].argmax().item()
            orig_conf = probs_orig[b, pred_cls].item()

            # single-pass GradCAM for AOPC speed (n=1, std=0)
            cam_np = gcam_engine._gcam(img_s, class_idx=pred_cls)
            flat_cam = torch.tensor(cam_np.flatten())

            drops = []
            for pct in percentages:
                k = max(1, int(flat_cam.numel() * pct / 100))
                topk = torch.topk(flat_cam, k).indices
                masked = img_s.clone()
                fv = masked[0, 0].flatten()
                fv[topk] = 0.0
                masked[0, 0] = fv.view(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
                with torch.inference_mode():
                    p_m = F.softmax(
                        model(masked, return_attn=False), dim=1
                    )[0, pred_cls].item()
                drops.append(orig_conf - p_m)
            aopc_scores.append(float(np.mean(drops)))

    return np.array(aopc_scores)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    all_pairs = load_dataset_pairs(DATASET_JSON, BASE_PATH)

    if len(all_pairs) == 0:
        raise RuntimeError("No valid image/label pairs found.")

    train_pairs, val_pairs = train_test_split(
        all_pairs, test_size=0.2, random_state=42
    )

    print(
        f"[INFO] Train pairs: {len(train_pairs)}  |  Val pairs: {len(val_pairs)}"
    )

    # ── datasets ──────────────────────────────────────────────────────────────
    train_ds = Pancreas3DPatchDataset(train_pairs, augment=True)
    val_ds = Pancreas3DPatchDataset(val_pairs, augment=False)

    # class-balanced sampler
    lbl_arr = [s[3] for s in train_ds.samples]
    cnts = np.bincount(lbl_arr)
    wts = 1.0 / cnts[lbl_arr]
    sampler = WeightedRandomSampler(wts, len(wts), replacement=True)

    # FIX 3: num_workers=0 to avoid high RAM / crash on CPU
    #         (persistent_workers removed since num_workers=0)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        pin_memory=False,  # no GPU — pin_memory has no benefit
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
    )

    # ── model & optimiser ─────────────────────────────────────────────────────
    model = Swin3DClassifierWithAttention(mc_dropout=True).to(DEVICE)
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    sgar_fn = SGARLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
    )

    # CPU-native autocast (bfloat16 supported on Zen 3 / Ryzen 5 5625U)
    use_autocast = torch.cpu.is_available()
    amp_dtype = torch.bfloat16

    best_f1 = 0.0
    val_accs, val_f1s = [], []
    train_losses_ce, train_losses_sgar = [], []

    # ══════════════════════════════════════════════════════════════════════════
    #  TRAINING LOOP
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TRAINING  —  SGAR + MC Dropout  (CPU optimised)")
    print(f"  PATCH_SIZE={PATCH_SIZE}, EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}")
    print(f"  num_workers=0, MC_SAMPLES={MC_SAMPLES}")
    print("=" * 60)

    for epoch in range(EPOCHS):
        model.train()
        run_ce = run_sgar = 0.0

        for imgs, masks, lbls in train_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            lbls = lbls.to(DEVICE)

            optimizer.zero_grad()

            # CPU bfloat16 autocast — reduces memory & speeds up on Zen3
            with torch.amp.autocast(
                device_type="cpu", dtype=amp_dtype, enabled=use_autocast
            ):
                logits, attn = model(imgs, return_attn=True)
                loss_ce = criterion_ce(logits, lbls)
                loss_sgar = sgar_fn(attn, masks)
                loss = loss_ce + LAMBDA_SGAR * loss_sgar

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            run_ce += loss_ce.item()
            run_sgar += loss_sgar.item()

        avg_ce = run_ce / len(train_loader)
        avg_sgar = run_sgar / len(train_loader)
        train_losses_ce.append(avg_ce)
        train_losses_sgar.append(avg_sgar)

        # ── validation ────────────────────────────────────────────────────────
        model.eval()
        preds_, gts_, probs_ = [], [], []
        with torch.inference_mode():
            for imgs, _, lbls in val_loader:
                out = model(imgs.to(DEVICE), return_attn=False)
                probs_.extend(F.softmax(out, 1)[:, 1].numpy())
                preds_.extend(out.argmax(1).numpy())
                gts_.extend(lbls.numpy())

        f1 = f1_score(gts_, preds_, zero_division=0)
        acc = accuracy_score(gts_, preds_)
        val_f1s.append(f1)
        val_accs.append(acc)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS}  "
            f"CE:{avg_ce:.4f}  SGAR:{avg_sgar:.4f}  "
            f"F1:{f1:.4f}  Acc:{acc:.4f}"
        )
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_swin_sgar.pth")
            print("  Best model saved  (best_swin_sgar.pth)")

    model.load_state_dict(
        torch.load("best_swin_sgar.pth", map_location=DEVICE)
    )
    model.eval()

    # ══════════════════════════════════════════════════════════════════════════
    #  GRAD-CAM GENERATION
    #  Hook backbone.layers[-2] for higher spatial resolution maps
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("GRAD-CAM GENERATION  (SmoothGrad-CAM 3D — High Resolution)")
    print("=" * 60)

    target_layer = model.backbone.layers[-2]  # higher-res than [-1]
    smooth_gcam = SmoothGradCAM3D(
        model,
        target_layer,
        n=GRADCAM_SMOOTH_N,
        std=GRADCAM_NOISE_STD,
    )

    gcam_maps = []
    raw_cts = []
    gt_masks_np = []
    pred_labels = []
    gt_labels = []

    print("  Computing Grad-CAM for all validation samples ...")
    count = 0
    for imgs, masks, lbls in val_loader:
        for b in range(imgs.shape[0]):
            img_s = imgs[b : b + 1].to(DEVICE)
            with torch.inference_mode():
                pred = model(img_s, return_attn=False).argmax(1).item()
            gcam = smooth_gcam(img_s, class_idx=pred)

            gcam_maps.append(gcam)
            raw_cts.append(imgs[b, 0].numpy())
            gt_masks_np.append(masks[b, 0].numpy())
            pred_labels.append(pred)
            gt_labels.append(lbls[b].item())
            count += 1
            if count % 50 == 0:
                print(f"    {count} samples processed ...")

    smooth_gcam.remove_hooks()

    gcam_maps = np.array(gcam_maps)
    raw_cts = np.array(raw_cts)
    gt_masks_np = np.array(gt_masks_np)
    pred_labels = np.array(pred_labels)
    gt_labels = np.array(gt_labels)

    # ── Grad-CAM IoU ──────────────────────────────────────────────────────────
    ious = localization_iou(gcam_maps, gt_masks_np, threshold=0.5)
    print("\nGrad-CAM Localisation IoU (tumour patches only):")
    print(f"  Mean IoU   : {ious.mean():.4f} +/- {ious.std():.4f}")
    print(f"  Median IoU : {np.median(ious):.4f}")
    print(f"  N samples  : {len(ious)}")

    # ── AOPC ──────────────────────────────────────────────────────────────────
    # FIX 6: AOPC DISABLED to save several hours on CPU.
    # Uncomment the block below if you want to run it (expect 2-4 extra hours).
    #
    # print("\nComputing AOPC (this may take several minutes on CPU) ...")
    # tmp_gcam = SmoothGradCAM3D(model, target_layer, n=1, std=0.0)
    # aopc = compute_aopc(model, val_loader, tmp_gcam)
    # tmp_gcam.remove_hooks()
    # print(f"Grad-CAM AOPC:")
    # print(f"  Mean AOPC  : {aopc.mean():.4f} +/- {aopc.std():.4f}")
    # print(f"  (Higher = more faithful explanation)")
    print("\n[INFO] AOPC computation SKIPPED (saves ~2-4 hours on CPU).")
    print("       Uncomment the AOPC block in __main__ to enable it.")

    # ══════════════════════════════════════════════════════════════════════════
    #  MC DROPOUT UNCERTAINTY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("MC DROPOUT UNCERTAINTY")
    print("=" * 60)

    model.load_state_dict(
        torch.load("best_swin_sgar.pth", map_location=DEVICE)
    )
    model.eval()
    model.enable_mc_dropout()

    mc_preds_, mc_gts_, mc_probs_, mc_uncert_ = [], [], [], []
    with torch.inference_mode():
        for imgs, _, lbls in val_loader:
            imgs_d = imgs.to(DEVICE)
            runs = np.stack(
                [
                    F.softmax(model(imgs_d, return_attn=False), 1)[:, 1].numpy()
                    for _ in range(MC_SAMPLES)
                ]
            )  # [MC, B]
            mp = runs.mean(0)
            vp = runs.var(0)
            mc_probs_.extend(mp)
            mc_uncert_.extend(vp)
            mc_gts_.extend(lbls.numpy())
            mc_preds_.extend((mp > 0.5).astype(int))

    mc_preds = np.array(mc_preds_)
    mc_gts = np.array(mc_gts_)
    mc_probs = np.array(mc_probs_)
    mc_uncert = np.array(mc_uncert_)

    UNCERTAINTY_THRESHOLD = float(
        np.percentile(mc_uncert, UNCERTAINTY_PERCENTILE)
    )
    confident = mc_uncert < UNCERTAINTY_THRESHOLD
    referred = mc_uncert >= UNCERTAINTY_THRESHOLD
    errors = mc_preds != mc_gts

    print(
        f"Auto threshold ({UNCERTAINTY_PERCENTILE}th pct): "
        f"{UNCERTAINTY_THRESHOLD:.5f}"
    )
    print(f"Overall Accuracy : {accuracy_score(mc_gts, mc_preds):.4f}")
    print(
        f"Overall F1       : {f1_score(mc_gts, mc_preds, zero_division=0):.4f}"
    )
    print(
        f"Confident samples: {confident.sum()} / {len(mc_uncert)} "
        f"({100 * confident.mean():.1f}%)"
    )
    print(
        f"Referred  samples: {referred.sum()}  / {len(mc_uncert)} "
        f"({100 * referred.mean():.1f}%)"
    )

    if confident.sum() > 0:
        print(
            f"Acc  (confident) : "
            f"{accuracy_score(mc_gts[confident], mc_preds[confident]):.4f}"
        )
        print(
            f"F1   (confident) : "
            f"{f1_score(mc_gts[confident], mc_preds[confident], zero_division=0):.4f}"
        )
        print(f"Err  (confident) : {errors[confident].mean():.4f}")
    if referred.sum() > 0:
        print(f"Err  (referred)  : {errors[referred].mean():.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  STANDARD PLOTS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("GENERATING STANDARD PLOTS")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(train_losses_ce, color="steelblue", lw=2, label="CE Loss")
    axes[0].plot(train_losses_sgar, color="coral", lw=2, label="SGAR Loss")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training Losses")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(val_f1s, color="green", lw=2, label="F1 Score")
    axes[1].plot(val_accs, color="orange", lw=2, label="Accuracy")
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Validation Metrics")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].hist(
        mc_uncert[mc_gts == 0],
        bins=30,
        alpha=0.6,
        color="steelblue",
        label="Healthy (label 0)",
    )
    axes[2].hist(
        mc_uncert[mc_gts == 1],
        bins=30,
        alpha=0.6,
        color="coral",
        label="Tumour  (label 1)",
    )
    axes[2].axvline(
        UNCERTAINTY_THRESHOLD,
        color="red",
        ls="--",
        label=f"Threshold = {UNCERTAINTY_THRESHOLD:.4f}",
    )
    axes[2].set(
        xlabel="Predictive Variance",
        ylabel="Count",
        title="MC Dropout Uncertainty",
    )
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_and_uncertainty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: training_and_uncertainty.png")

    # Confusion matrix
    cm = confusion_matrix(mc_gts, mc_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Healthy", "Tumour"],
        yticklabels=["Healthy", "Tumour"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix  (SGAR + MC Dropout)")
    plt.savefig("confusion_matrix_sgar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: confusion_matrix_sgar.png")

    # ROC curve
    fpr, tpr, _ = roc_curve(mc_gts, mc_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve  (SGAR + MC Dropout)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("roc_curve_sgar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: roc_curve_sgar.png")

    # Grad-CAM IoU distribution
    plt.figure(figsize=(6, 5))
    plt.hist(ious, bins=25, color="teal", alpha=0.75, edgecolor="black")
    plt.axvline(
        ious.mean(),
        color="red",
        ls="--",
        label=f"Mean IoU = {ious.mean():.4f}",
    )
    plt.xlabel("IoU")
    plt.ylabel("Count")
    plt.title("Grad-CAM Localisation IoU  (Tumour Patches, label == 2)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("gradcam_iou_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: gradcam_iou_distribution.png")

    # ══════════════════════════════════════════════════════════════════════════
    #  GRAD-CAM 5-PANEL VISUALISATIONS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("GRAD-CAM  5-PANEL VISUALISATIONS  (200 DPI)")
    print("=" * 60)

    shown = 0
    for i in range(len(gcam_maps)):
        if gt_masks_np[i].sum() == 0:
            continue

        iou_arr = localization_iou(
            gcam_maps[i : i + 1], gt_masks_np[i : i + 1], threshold=0.5
        )
        iou_val = float(iou_arr.mean()) if len(iou_arr) > 0 else 0.0

        save_gradcam_figure(
            sample_idx=i,
            ct=raw_cts[i],
            gt_mask=gt_masks_np[i],
            gcam=gcam_maps[i],
            pred_label=pred_labels[i],
            true_label=gt_labels[i],
            iou_val=iou_val,
            save_path=f"gradcam_sample_{shown}.png",
        )
        shown += 1
        if shown >= GRADCAM_VIS_SAMPLES:
            break

    # ══════════════════════════════════════════════════════════════════════════
    #  FINAL CLASSIFICATION REPORT
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("FINAL CLASSIFICATION REPORT")
    print("=" * 60)
    print(
        classification_report(
            mc_gts,
            mc_preds,
            target_names=["Healthy (bg + pancreas)", "Tumour (cancer)"],
            zero_division=0,
        )
    )

    print("\nAll components complete.")
    print("   1. dataset.json  -- official train split used")
    print("   2. Label mapping -- 0=background, 1=pancreas, 2=cancer (from JSON)")
    print("   3. SGAR          -- stage[-2] sigmoid attention regularisation")
    print("   4. SmoothGradCAM -- 3-D heatmaps, 5-panel + 3-plane at 200 DPI")
    print("   5. Faithfulness  -- Grad-CAM IoU (AOPC disabled for speed)")
    print("   6. MC Dropout    -- percentile-based adaptive referral")
    print("   7. imagesTs/     -- correctly excluded (no labels available)")
    print("   8. CPU optimised -- 12 threads, bfloat16 autocast, 0 workers")
    print(
        f"   9. Runtime target -- PATCH_SIZE={PATCH_SIZE}, EPOCHS={EPOCHS}, "
        f"MC={MC_SAMPLES}, SmoothN={GRADCAM_SMOOTH_N}"
    )
