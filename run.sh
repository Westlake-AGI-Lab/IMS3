#!/bin/bash
# Full ImS3 pipeline on Imagewoof:
#   Stage 1: IM fine-tuning (train_dit.py, Algorithm 1, lambda_match = lambda_IM)
#   Stage 2: S3 sampling     (centroid.py,  Algorithm 2)
#   Stage 3: Classifier eval (train.py)
#
# Usage: bash run.sh
#
# Edit the paths below for your environment.

set -e
export CUDA_VISIBLE_DEVICES=0

# ---- paths ----
REAL_TRAIN_DIR="/home/share/dataset/Imagewoof/imagewoof2/train"
REAL_ROOT="/home/share/dataset/Imagewoof/imagewoof2"
PRETRAINED_CKPT="pretrained_models/DiT-XL-2-256x256.pt"
FT_RESULTS_DIR="../logs/ims3-imagewoof"
DISTILL_DIR="../results/ims3-imagewoof"

# ---- hyper-parameters ----
SPEC="woof"
NCLASS=10
IPC=10
LAMBDA_IM=0.002      # paper Eq.(7)
W_REAL=0.4           # alpha in paper Eq.(10)
W_SEP=0.9            # beta  in paper Eq.(10)
SEL_EPS=0            # log stability term
GROUPS=5             # G in paper Algorithm 2
SAMPLE_BATCH=10      # diffusion sampling batch size per call
LR=0.1               # downstream classifier learning rate

# =====================================================================
# Stage 1 — IM fine-tuning
# =====================================================================
torchrun --nnode=1 --master_port=25678 train_dit.py \
    --model DiT-XL/2 \
    --data_path ${REAL_TRAIN_DIR} \
    --ckpt ${PRETRAINED_CKPT} \
    --global_batch_size 8 \
    --epochs 8 \
    --tag im \
    --ckpt_every 1000 \
    --log_every 200 \
    --condense \
    --finetune_ipc -1 \
    --results_dir ${FT_RESULTS_DIR} \
    --spec ${SPEC} \
    --lambda_match ${LAMBDA_IM}

# Use the step-8000 checkpoint (paper-validated as the best stopping point on
# Imagewoof). Falls back to the latest checkpoint if 0008000.pt is absent.
FT_CKPT=$(ls ${FT_RESULTS_DIR}/*-DiT-XL-2-im/checkpoints/0008000.pt 2>/dev/null | head -n 1)
[ -z "${FT_CKPT}" ] && FT_CKPT=$(ls -t ${FT_RESULTS_DIR}/*-DiT-XL-2-im/checkpoints/*.pt | head -n 1)
echo "[run.sh] using fine-tuned checkpoint: ${FT_CKPT}"

# =====================================================================
# Stage 2 — S3 sampling (centroid-based subgroup selection)
# =====================================================================
python centroid.py \
    --model DiT-XL/2 \
    --image-size 256 \
    --ckpt ${FT_CKPT} \
    --save-dir ${DISTILL_DIR} \
    --spec ${SPEC} \
    --nclass ${NCLASS} \
    --ipc ${IPC} \
    --groups ${GROUPS} \
    --real-train-dir ${REAL_TRAIN_DIR} \
    --w-real ${W_REAL} \
    --w-sep ${W_SEP} \
    --sel-eps ${SEL_EPS} \
    --sample-batch ${SAMPLE_BATCH} \
    --feature-backbone resnet18 \
    --seed 0

# =====================================================================
# Stage 3 — Train downstream classifier on the distilled set
# =====================================================================
python train.py \
    -d imagenet \
    --imagenet_dir ${DISTILL_DIR}/final_distilled/train ${REAL_ROOT} \
    -n resnet_ap \
    --nclass ${NCLASS} \
    --norm_type instance \
    --ipc ${IPC} \
    --tag ims3 \
    --slct_type random \
    --spec ${SPEC} \
    --lr ${LR} \
    --randaug true --randaug_n 1 --randaug_m 6
