# [CVPR2026] IMS3: Breaking Distributional Aggregation in Diffusion-Based Dataset Distillation

Reference implementation of **IMS3** (CVPR 2026)

[📄 Paper (arXiv)](https://arxiv.org/abs/2603.13960) &nbsp;

> Diffusion-based dataset distillation tends to over-concentrate synthetic samples
> in high-density regions of the data manifold, leaving boundary regions — which
> are crucial for classification — underrepresented. IMS3 addresses this with two
> complementary strategies:
>
> - **Inversion-Matching (IM)**: a fine-tuning loss that aligns each training noise
>   latent `z_t` with its DDIM-inverted counterpart `z_t^inv`, exploiting the
>   inherent instability of inversion to push the generator toward low-density
>   regions and broaden distributional coverage.
> - **Selective Subgroup Sampling (S³)**: a training-free sampler that draws G
>   candidate subgroups per class, computes feature centroids, and selects the
>   tuple (g₁, …, g_C) that is simultaneously close to the per-class real
>   centroids and far from other-class centroids.

## Setup

```bash
conda create -n ims3 python=3.10 -y
conda activate ims3
pip install -r requirements.txt
```

Tested on a single NVIDIA H200 / A100 (40GB+) with PyTorch 2.4.1 + CUDA 12.1.

## Data

IMS3 fine-tunes on a class-folder ImageNet subset and validates on the
corresponding real validation set. Layout:

```
<data_root>/
  imagewoof2/
    train/
      n02086240/  *.JPEG ...
      n02087394/  ...
      ...
    val/
      n02086240/  ...
      ...
```

Edit the paths in `run.sh` (`REAL_TRAIN_DIR`, `REAL_ROOT`) before running.
Alternative subsets are selected via `--spec {woof,nette,100}` and resolved
against the class lists in `misc/`.

## Pretrained DiT

The IM stage initialises from the public DiT-XL/2 checkpoint:

```bash
mkdir -p pretrained_models
curl -L -o pretrained_models/DiT-XL-2-256x256.pt \
    https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt
```

`download.py` does the same on demand.

## Pipeline

```bash
bash run.sh                       # full Imagewoof / IPC=10 pipeline
```

`run.sh` chains three stages:

| Stage | Script        | Paper ref                      | Output                                                   |
|-------|---------------|--------------------------------|----------------------------------------------------------|
| 1     | `train_dit.py`| Algorithm 1, Eq. (5)–(7)       | IM-fine-tuned DiT checkpoints under `../logs/...`        |
| 2     | `centroid.py` | Algorithm 2, Eq. (8)–(11)      | Distilled images under `<DISTILL_DIR>/final_distilled/train/<class>/` |
| 3     | `train.py`    | Sec. 5 evaluation protocol     | Top-1 accuracy over a real validation set                |

To reproduce a single stage, look at the corresponding command block in
`run.sh` — every flag is named there.

### Plain (no-S³) sampling

For the DiT and "+IM" ablation rows, replace Stage 2 with:

```bash
python sample.py --model DiT-XL/2 --image-size 256 \
    --ckpt <path/to/finetuned.pt> --save-dir <out> --spec woof
```

## Hyperparameters

The defaults baked into `run.sh` are the ones we used to obtain the reproduced
number above. The most influential knobs:

| Flag             | Default | Note                                              |
|------------------|--------:|---------------------------------------------------|
| `--lambda_match` | 0.002   | IM loss weight λ_IM (paper Eq. 7)                 |
| `--w-real` (α)   | 0.4     | proximity-to-real weight (paper Eq. 10)           |
| `--w-sep`  (β)   | 0.9     | inter-class separation weight (paper Eq. 10)      |
| `--sel-eps`      | 0       | log-stability ε (paper notation)                  |
| `--groups` (G)   | 5       | candidate subgroups per class                     |
| `--ipc` (K)      | 10      | images per subgroup                               |
| `--lr`           | 0.1     | downstream classifier learning rate (Stage 3)     |

## Repository layout

```
argument.py            # parser used by train.py
centroid.py            # Stage 2 — S³ sampler
data.py                # ImageFolder for IM fine-tuning + classifier
diffusion/             # OpenAI guided-diffusion / IDDPM port
download.py            # auto-downloads DiT-XL/2
misc/                  # class-index files + shared utils
models.py              # DiT model + DiT_models registry
run.sh                 # full pipeline
sample.py              # plain class-conditional sampling (no S³)
train.py               # Stage 3 — classifier training
train_dit.py           # Stage 1 — IM fine-tuning
train_models/          # ResNet / ResNetAP / ConvNet / DenseNet zoo
```

## Citation

```bibtex
@article{wang2026ims3,
  title   = {IMS3: Breaking Distributional Aggregation in Diffusion-Based Dataset Distillation},
  author  = {Wang, Chenru and Chen, Yunyi and Yang, Zijun and Zhou, Joey Tianyi and Zhang, Chi},
  journal = {arXiv preprint arXiv:2603.13960},
  year    = {2026}
}
```

## Acknowledgements

The DiT backbone and diffusion utilities are adapted from
[DiT](https://github.com/facebookresearch/DiT) and OpenAI's
[guided-diffusion](https://github.com/openai/guided-diffusion).
The Stage-3 evaluator follows the protocol of
[Minimax-Diffusion](https://github.com/vimar-gu/MinimaxDiffusion).
We would also like to thank the amazing work of [RDED](https://github.com/LINs-lab/RDED/tree/main), [CaO2](https://github.com/hatchetProject/CaO2), and other related works for their inspiring and impactful contributions to this line of research.
