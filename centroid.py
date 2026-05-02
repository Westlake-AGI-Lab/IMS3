"""
Sample new images from a pre-trained DiT, then select distilled subgroups:
- For each class, generate G subgroups (each with K images).
- Extract features, compute each subgroup centroid.
- Selection objective (maximize):
    L = alpha * sum_i [ -log( d(c_{i,g_i}, r_i) + eps ) ]   # close to real
        + beta  * sum_{i<j} [ log( d(c_{i,g_i}, c_{j,g_j}) + eps ) ]   # inter-class separation
  where d(u,v) = 1 - cos(u,v), eps>0 for stability.

Author: you+chatgpt
"""

import os
import io
import math
import random
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision import transforms, models
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models

# ----------------------------
# Utils: paths & io
# ----------------------------

def read_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return [x.strip() for x in f.readlines() if x.strip()]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _normalize_exts(exts: List[str]) -> Tuple[str, ...]:
    # 支持逗号分隔或空格分隔，自动扩展大小写
    tokens: List[str] = []
    for e in exts:
        e = e.strip()
        if not e:
            continue
        if "," in e:
            tokens.extend([x.strip() for x in e.split(",") if x.strip()])
        else:
            tokens.append(e)
    uniq = set()
    for t in tokens:
        if not t.startswith("."):
            t = "." + t
        uniq.add(t)
        uniq.add(t.upper())
    return tuple(sorted(uniq))

def list_images(root: Path, exts=(".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff")) -> List[Path]:
    imgs = []
    for e in exts:
        imgs += list(root.rglob(f"*{e}"))
    return imgs

# ----------------------------
# Feature extractor backbones
# ----------------------------
from torchvision import transforms, models

class _BaseFeatNet(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

    @torch.no_grad()
    def encode(self, img_tensor_bchw: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _tfm_imagenet(img_size=224):
        return transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225]),
        ])

class ResNet50Feats(_BaseFeatNet):
    def __init__(self, device="cuda", img_size=224):
        super().__init__(device)
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.backbone = models.resnet50(weights=weights)
        except Exception:
            self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = torch.nn.Identity()
        self.backbone = self.backbone.to(device).eval()
        self.tfm = self._tfm_imagenet(img_size)

    @torch.no_grad()
    def encode(self, img_tensor_bchw: torch.Tensor) -> torch.Tensor:
        x = (img_tensor_bchw + 1.0) * 0.5
        x = self.tfm(x)
        feats = self.backbone(x)          # (B, 2048)
        return F.normalize(feats, dim=1)

class ResNet18Feats(_BaseFeatNet):
    def __init__(self, device="cuda", img_size=224):
        super().__init__(device)
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
        except Exception:
            self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = torch.nn.Identity()
        self.backbone = self.backbone.to(device).eval()
        self.tfm = self._tfm_imagenet(img_size)

    @torch.no_grad()
    def encode(self, img_tensor_bchw: torch.Tensor) -> torch.Tensor:
        x = (img_tensor_bchw + 1.0) * 0.5
        x = self.tfm(x)
        feats = self.backbone(x)          # (B, 512)
        return F.normalize(feats, dim=1)

class ResNet101Feats(_BaseFeatNet):
    def __init__(self, device="cuda", img_size=224):
        super().__init__(device)
        try:
            weights = models.ResNet101_Weights.IMAGENET1K_V2
            self.backbone = models.resnet101(weights=weights)
        except Exception:
            self.backbone = models.resnet101(pretrained=True)
        self.backbone.fc = torch.nn.Identity()
        self.backbone = self.backbone.to(device).eval()
        self.tfm = self._tfm_imagenet(img_size)

    @torch.no_grad()
    def encode(self, img_tensor_bchw):
        x = (img_tensor_bchw + 1.0) * 0.5
        x = self.tfm(x)
        feats = self.backbone(x)  # (B, 2048)
        return F.normalize(feats, dim=1)

class EfficientNetFeats(_BaseFeatNet):
    def __init__(self, device="cuda", img_size=224):
        super().__init__(device)
        try:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
        except Exception:
            self.backbone = models.efficientnet_b0(pretrained=True)
        # EfficientNet 用 classifier 替代 fc
        self.backbone.classifier = torch.nn.Identity()
        self.backbone = self.backbone.to(device).eval()
        self.tfm = self._tfm_imagenet(img_size)

    @torch.no_grad()
    def encode(self, img_tensor_bchw):
        x = (img_tensor_bchw + 1.0) * 0.5
        x = self.tfm(x)
        feats = self.backbone(x)  # (B, 1280)
        return F.normalize(feats, dim=1)

class CLIPFeats(_BaseFeatNet):
    """
    优先使用 open_clip；不可用则回退到 openai/clip。
    默认使用 ViT-B/32 (openai 权重)。你也可以通过 CLI 配置。
    """
    def __init__(self, device="cuda", arch="ViT-B-32", pretrained="openai", img_size=224):
        super().__init__(device)
        self.img_size = img_size
        self._impl = None    # "open_clip" | "openai_clip"

        # 先尝试 open_clip
        try:
            import open_clip
            self._impl = "open_clip"
            self.model, _, _preprocess = open_clip.create_model_and_transforms(
                arch, pretrained=pretrained, device=device
            )
            self.model.eval()
        except Exception:
            # 回退到 openai/clip
            import clip
            self._impl = "openai_clip"
            self.model, _preprocess = clip.load(arch.replace("-", "/"), device=device, jit=False)
            self.model.eval()

        # 统一用 CLIP 的标准归一化参数
        self.tfm = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    @torch.no_grad()
    def encode(self, img_tensor_bchw: torch.Tensor) -> torch.Tensor:
        x = (img_tensor_bchw + 1.0) * 0.5
        x = self.tfm(x)
        if self._impl == "open_clip":
            feats = self.model.encode_image(x)   # (B, D)
        else:
            feats = self.model.encode_image(x)
        feats = feats.float()
        return F.normalize(feats, dim=1)

def build_feature_extractor(name: str, device="cuda",
                            img_size=224,
                            clip_arch="ViT-B-32",
                            clip_pretrained="openai"):
    """
    name: 'resnet50' | 'resnet18' | 'clip'
    """
    n = (name or "resnet50").lower()
    if n in ["resnet50", "rn50"]:
        return ResNet50Feats(device=device, img_size=img_size)
    if n in ["resnet18", "rn18"]:
        return ResNet18Feats(device=device, img_size=img_size)
    if n in ["resnet101", "rn101"]:
        return ResNet101Feats(device=device, img_size=img_size)
    if n in ["Efficient"]:
        return EfficientNetFeats(device=device, img_size=img_size)
    if n in ["clip", "clip_vitb32", "clip-vitb32", "clip-vitl14", "clip_vitl14"]:
        return CLIPFeats(device=device, arch=clip_arch, pretrained=clip_pretrained, img_size=img_size)
    # fallback
    return ResNet50Feats(device=device, img_size=img_size)


# ----------------------------
# Centroid & distances
# ----------------------------

def compute_centroid(feats: torch.Tensor) -> torch.Tensor:
    # feats: (N, D), assumed L2-normalized rows
    c = feats.mean(dim=0)
    c = F.normalize(c, dim=0)
    return c  # (D,)

def cosine_distance(u: torch.Tensor, v: torch.Tensor) -> float:
    # u, v: (D,)
    return float(1.0 - torch.dot(u, v).clamp(-1, 1))

# ---- extra distance utils for selection objective ----

def cosine_dist(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # u: (..., D), v: (..., D) —— L2 normalized
    return 1.0 - torch.sum(u * v, dim=-1).clamp(-1, 1)

def combined_objective(
    selected_idx: List[int],
    cand_centroids: List[torch.Tensor],  # len=C, each (G,D)
    real_centroids: List[torch.Tensor],  # len=C, each (D,)
    alpha: float,
    beta: float,
    eps: float,
) -> float:
    """
    计算 L（越大越好）。
    """
    C = len(cand_centroids)
    # 贴近真实项
    close_terms = []
    picked = []
    for i in range(C):
        c_i = cand_centroids[i][selected_idx[i]]  # (D,)
        r_i = real_centroids[i]                   # (D,)
        d_real = cosine_dist(c_i, r_i)           # scalar
        close_terms.append(-torch.log(d_real + eps))
        picked.append(c_i.unsqueeze(0))
    picked = torch.cat(picked, dim=0)  # (C,D)

    # 类间分离项：两两 log 距离
    sims = (picked @ picked.t()).clamp(-1, 1)     # (C,C)
    dmat = 1.0 - sims                              # (C,C)
    i_idx, j_idx = torch.triu_indices(C, C, offset=1)
    sep_terms = torch.log(dmat[i_idx, j_idx] + eps)

    L = alpha * torch.stack(close_terms).sum() + beta * sep_terms.sum()
    return float(L.item())

def optimize_selection(
    cand_centroids: List[torch.Tensor],  # len=C, each (G,D)
    real_centroids: List[torch.Tensor],  # len=C, each (D,)
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-6,
    max_iters: int = 5,
) -> List[int]:
    """
    坐标上升选择：返回每类选中的 subgroup 索引。
    """
    C = len(cand_centroids)
    G = cand_centroids[0].size(0)

    # Warm-start init: pick the candidate subgroup whose centroid is closest to
    # the real centroid for each class. The paper (Algorithm 2) describes a
    # random init; we use closest-to-real as a deterministic warm start to
    # speed up coordinate ascent without changing the final objective.
    selected = []
    for i in range(C):
        r_i = real_centroids[i].unsqueeze(0)         # (1,D)
        d_all = cosine_dist(cand_centroids[i], r_i)  # (G,1)->(G,)
        gi0 = int(torch.argmin(d_all).item())
        selected.append(gi0)

    best_val = combined_objective(selected, cand_centroids, real_centroids, alpha, beta, eps)

    # 坐标上升
    for _ in range(max_iters):
        improved = False
        for i in range(C):
            cur_gi = selected[i]
            best_gi_i = cur_gi
            best_val_i = best_val
            for gi in range(G):
                if gi == cur_gi:
                    continue
                trial = list(selected)
                trial[i] = gi
                val = combined_objective(trial, cand_centroids, real_centroids, alpha, beta, eps)
                if val > best_val_i:
                    best_val_i = val
                    best_gi_i = gi
            if best_gi_i != cur_gi:
                selected[i] = best_gi_i
                best_val = best_val_i
                improved = True
        if not improved:
            break

    return selected

# ----------------------------
# Real-train set centroid per class
# ----------------------------

@torch.no_grad()
def compute_real_class_centroid(
    feat_net: torch.nn.Module,
    class_dir: Path,
    device: str = "cuda",
    max_imgs: int = 200,
    exts: Tuple[str,...] = (".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"),
    seed: int = 0,
    unify_hw: Tuple[int,int] = (256,256),   # 方案A：统一尺寸，避免 stack 报错
) -> torch.Tensor:
    """
    Load up to max_imgs images from class_dir, compute normalized features, return centroid (D,)
    - 方案A：先把每张图 resize 成统一的 (H,W)，从而可以安全 stack。
    """
    rng = random.Random(seed)
    paths = list_images(class_dir, exts)
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found in {class_dir}")
    if len(paths) > max_imgs:
        rng.shuffle(paths)
        paths = paths[:max_imgs]

    imgs = []
    to_tensor = transforms.ToTensor()  # [0,1]
    Ht, Wt = unify_hw
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        t = to_tensor(img)             # (3,H,W) in [0,1]
        t = t.unsqueeze(0)             # (1,3,H,W)
        # 统一尺寸（双线性），再映射到 [-1,1]
        t = F.interpolate(t, size=(Ht, Wt), mode="bilinear", align_corners=False).squeeze(0)
        t = t * 2.0 - 1.0              # [-1,1]
        imgs.append(t)
    if len(imgs) == 0:
        raise RuntimeError(f"All images failed to load for {class_dir}")

    # Batch encode
    B = 64
    feats_list = []
    for s in range(0, len(imgs), B):
        batch = torch.stack(imgs[s:s+B], dim=0).to(device)  # (B,3,H,W) 同尺寸
        feats = feat_net.encode(batch)  # (B,D), L2-normalized
        feats_list.append(feats)
    feats_all = torch.cat(feats_list, dim=0)  # (N,D)
    c = compute_centroid(feats_all)  # (D,)
    return c

# ----------------------------
# DiT sampling (kept from your original, wrapped)
# ----------------------------

@torch.no_grad()
def sample_one_group_for_one_class(
    model, diffusion, vae, device,
    class_label: int,
    num_steps: int,
    cfg_scale: float,
    latent_size: int,
    k_per_group: int,
    sample_batch: int = 0,
) -> torch.Tensor:
    """
    Generate K images for a single class. Sampling is batched in chunks of
    `sample_batch` (defaults to k_per_group, i.e. one batch per group).
    Returns (K, 3, H, W) in [-1, 1].
    """
    if sample_batch <= 0:
        sample_batch = k_per_group
    images = []
    remaining = k_per_group
    while remaining > 0:
        bs = min(sample_batch, remaining)
        z = torch.randn(bs, 4, latent_size, latent_size, device=device)
        y = torch.tensor([class_label] * bs, device=device)
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * bs, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=cfg_scale)
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z,
            clip_denoised=False, model_kwargs=model_kwargs,
            progress=False, device=device
        )
        samples, _ = samples.chunk(2, dim=0)
        imgs = vae.decode(samples / 0.18215).sample  # (bs,3,H,W) in [-1,1]
        images.append(imgs.detach().cpu())
        remaining -= bs
    return torch.cat(images, dim=0)  # (K,3,H,W)

# ----------------------------
# Main
# ----------------------------

def main(args):
    # Setup
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- prepare class list -----
    all_classes = read_lines("./misc/class_indices.txt")
    if args.spec == 'woof':
        file_list = './misc/class_woof.txt'
    elif args.spec == 'nette':
        file_list = './misc/class_nette.txt'
    else:
        file_list = './misc/class100.txt'
    sel_classes = read_lines(file_list)

    phase = max(0, args.phase)
    cls_from = args.nclass * phase
    cls_to = args.nclass * (phase + 1)
    sel_classes = sel_classes[cls_from:cls_to]
    class_labels = [all_classes.index(x) for x in sel_classes]

    # ----- load DiT & VAE -----
    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device).eval()

    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device).eval()

    # ----- dirs -----
    save_root = ensure_dir(Path(args.save_dir))
    tmp_root  = ensure_dir(save_root / "tmp_candidates")   # 临时候选
    final_root = ensure_dir(save_root / "final_distilled") # 最终蒸馏（train/class/...）

    # ----- feature extractor -----
    feat_net = build_feature_extractor(
        args.feature_backbone,
        device=device,
        img_size=args.feat_img_size,
        clip_arch=args.clip_arch,
        clip_pretrained=args.clip_pretrained,
    )

    # ----- compute REAL centroids per class -----
    assert args.real_train_dir is not None, \
        "Please set --real-train-dir to your original training set root."
    real_root = Path(args.real_train_dir)

    # 兼容逗号/空格混合传参，且自动大小写
    real_exts = _normalize_exts(args.real_exts)

    real_centroids: Dict[int, torch.Tensor] = {}
    print("[Real] Computing real centroids per class ...")
    for sel_class, class_id in tqdm(list(zip(sel_classes, class_labels)), desc="RealCentroids"):
        class_dir = real_root / sel_class
        c = compute_real_class_centroid(
            feat_net, class_dir, device=device,
            max_imgs=args.real_max_per_class,
            exts=real_exts,
            seed=args.seed,
            unify_hw=(args.real_unify_h, args.real_unify_w),
        )
        real_centroids[class_id] = c  # (D,)

    # Store per-class subgroups: images, feats, centroids
    per_class_groups_imgs: Dict[int, List[torch.Tensor]] = {}
    per_class_groups_centroids: Dict[int, torch.Tensor] = {}  # class_id -> (G,D)
    per_class_group_savepaths: Dict[int, List[List[Path]]] = {}

    print(f"[Generation] Each class: G={args.groups} subgroups, K={args.ipc} images per subgroup.")
    for class_id, sel_class in tqdm(list(zip(class_labels, sel_classes)), desc="Classes"):
        per_class_groups_imgs[class_id] = []
        per_class_group_savepaths[class_id] = []
        # generate G subgroups
        for g_idx in range(args.groups):
            imgs = sample_one_group_for_one_class(
                model, diffusion, vae, device,
                class_label=class_id,
                num_steps=args.num_sampling_steps,
                cfg_scale=args.cfg_scale,
                latent_size=latent_size,
                k_per_group=args.ipc,
                sample_batch=args.sample_batch,
            )  # (K,3,H,W) in [-1,1]
            per_class_groups_imgs[class_id].append(imgs)

            # save tmp
            out_dir = ensure_dir(tmp_root / f"{sel_class}" / f"group_{g_idx:02d}")
            paths = []
            for k in range(imgs.size(0)):
                out_path = out_dir / f"{k + args.total_shift:04d}.png"
                save_image(imgs[k], out_path, normalize=True, value_range=(-1, 1))
                paths.append(out_path)
            per_class_group_savepaths[class_id].append(paths)

        # compute subgroup centroids for this class
        with torch.no_grad():
            centroids = []
            for g_idx in range(args.groups):
                imgs = per_class_groups_imgs[class_id][g_idx]  # (K,3,H,W)
                # batch features
                chunk = 64
                f_list = []
                for s in range(0, imgs.size(0), chunk):
                    batch = imgs[s:s+chunk].to(device)
                    f_list.append(feat_net.encode(batch))  # (B,D)
                feats = torch.cat(f_list, dim=0)  # (K,D)
                c = compute_centroid(feats)       # (D,)
                centroids.append(c.unsqueeze(0))
            per_class_groups_centroids[class_id] = torch.cat(centroids, dim=0)  # (G,D)

    # ----- per-class selection: maximize combined objective -----
    print("[Select] Optimizing subgroup selection with combined objective (close-to-real + inter-class separation) ...")

    # 组装为列表（与 sel_classes 顺序对齐）
    cand_centroids_list: List[torch.Tensor] = []
    real_centroids_list: List[torch.Tensor] = []
    for sel_class, class_id in zip(sel_classes, class_labels):
        cand_centroids_list.append(per_class_groups_centroids[class_id])  # (G,D)
        real_centroids_list.append(real_centroids[class_id])              # (D,)

    selected_list = optimize_selection(
        cand_centroids=cand_centroids_list,
        real_centroids=real_centroids_list,
        alpha=args.w_real,
        beta=args.w_sep,
        eps=args.sel_eps,
        max_iters=args.sel_max_iters,
    )

    # 回填为 dict（class_id -> gi）
    selected_group_idx: Dict[int, int] = {}
    for (sel_class, class_id), gi in zip(zip(sel_classes, class_labels), selected_list):
        selected_group_idx[class_id] = int(gi)

    # ----- materialize final distilled dataset -----
    print("[Finalize] Copying selected subgroups to final distilled directory...")
    for class_id, sel_class in zip(class_labels, sel_classes):
        g_idx = selected_group_idx[class_id]
        src_paths = per_class_group_savepaths[class_id][g_idx]  # list of Paths
        dst_dir = ensure_dir(final_root / "train" / sel_class)
        if args.clean_final and dst_dir.exists():
            for f in dst_dir.glob("*"):
                if f.is_file():
                    f.unlink()
        for p in src_paths:
            shutil.copy2(p, dst_dir / p.name)

    # ----- summary -----
    # 打印每类距离信息
    print("\n[Summary per class]")
    for class_id, sel_class in zip(class_labels, sel_classes):
        gi = selected_group_idx[class_id]
        real_c = real_centroids[class_id]
        cand_c = per_class_groups_centroids[class_id]  # (G,D)
        sims = torch.mv(cand_c, real_c)
        dists = (1.0 - sims.clamp(-1, 1)).tolist()
        print(f" - Class {sel_class}: picked group #{gi} | dists_to_real={['%.4f'%d for d in dists]}")

    # 评估最终选择的总体目标值与两项分量，便于调参
    def _eval_terms(selected_idx):
        C = len(cand_centroids_list)
        close_terms = []
        picked = []
        for i in range(C):
            c_i = cand_centroids_list[i][selected_idx[i]]
            r_i = real_centroids_list[i]
            d_real = cosine_dist(c_i, r_i)
            close_terms.append(float((-torch.log(d_real + args.sel_eps)).item()))
            picked.append(c_i.unsqueeze(0))
        picked = torch.cat(picked, dim=0)
        sims = (picked @ picked.t()).clamp(-1, 1)
        dmat = 1.0 - sims
        i_idx, j_idx = torch.triu_indices(C, C, offset=1)
        sep = torch.log(dmat[i_idx, j_idx] + args.sel_eps).sum()
        return sum(close_terms), float(sep.item())

    close_val, sep_val = _eval_terms(selected_list)
    total_val = args.w_real * close_val + args.w_sep * sep_val
    print(f"\n[Objective] total={total_val:.4f} | alpha*close={args.w_real*close_val:.4f} (close={close_val:.4f}) | beta*sep={args.w_sep*sep_val:.4f} (sep={sep_val:.4f})")

    print(f"\nDone. Final distilled dataset at: {final_root / 'train'}")
    print(f"(Temporary candidates kept at: {tmp_root})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ---- original args ----
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to a DiT checkpoint. For the full ImS3 method, this should be the IM-fine-tuned checkpoint produced by train_dit.py. If None, falls back to auto-downloading the pre-trained DiT-XL/2 (i.e. the +S3-only ablation).")
    parser.add_argument("--spec", type=str, default='woof', help='specific subset for generation')
    parser.add_argument("--save-dir", type=str, default='../logs/test', help='the root directory for outputs')
    parser.add_argument("--total-shift", type=int, default=0, help='index offset for the file name')
    parser.add_argument("--nclass", type=int, default=10, help='the class number for generation')
    parser.add_argument("--phase", type=int, default=0, help='the phase number for generating large datasets')

    # ---- subgroup selection (new) ----
    parser.add_argument("--groups", type=int, default=5, help="number of subgroups per class (G)")
    parser.add_argument("--ipc", type=int, default=50, help="images per subgroup (K)")

    # ---- feature extractor config ----
    parser.add_argument("--feature-backbone", type=str, default="resnet18",
                    choices=["resnet50","resnet18","clip","resnet101","Efficient"],
                    help="feature extractor backbone")
    parser.add_argument("--feat-img-size", type=int, default=224,
                    help="resize/center-crop size for feature extractor")
    parser.add_argument("--clip-arch", type=str, default="ViT-B-32",
                    help="CLIP arch, e.g., ViT-B-32 / ViT-L-14")
    parser.add_argument("--clip-pretrained", type=str, default="openai",
                    help="CLIP pretrained tag, e.g., openai / laion2b_s34b_b79k")

    parser.add_argument("--clean-final", action="store_true", help="clean existing files in final class dir before copy")

    # ---- real-train centroid config ----
    parser.add_argument("--real-train-dir", type=str, required=True,
                        help="root of the ORIGINAL training set; must contain subfolders named by class ids (e.g., n02086240).")
    parser.add_argument("--real-max-per-class", type=int, default=200,
                        help="max #real images per class used to estimate real centroid (trade speed/accuracy).")
    parser.add_argument("--real-exts", nargs="+", default=[".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff",".JPEG",".JPG",".PNG"],
                        help="allowed image extensions (支持逗号/空格混合输入，自动大小写)。例：.JPEG,.JPG,.PNG .jpeg .jpg .png")
    parser.add_argument("--real-unify-h", type=int, default=256, help="方案A：统一 resize 高度")
    parser.add_argument("--real-unify-w", type=int, default=256, help="方案A：统一 resize 宽度")

    # ---- objective weights ----
    parser.add_argument("--w-real", type=float, default=1.0,
                        help="权重 α：贴近真实项（-log(d_real+eps)）系数")
    parser.add_argument("--w-sep", type=float, default=1.0,
                        help="权重 β：类间分离项（log(d_between+eps)）系数")
    parser.add_argument("--sel-eps", type=float, default=1e-6,
                        help="log 的数值稳定项 eps（论文 Eq.(5)/(8) 中的 stability epsilon）")
    parser.add_argument("--sel-max-iters", type=int, default=5,
                        help="坐标上升最大轮数（每轮遍历所有类别，若无改进则提前停止）")
    parser.add_argument("--sample-batch", type=int, default=0,
                        help="diffusion sampling batch size per call (<=0 means use --ipc, i.e. one batch per subgroup)")

    args = parser.parse_args()
    main(args)
