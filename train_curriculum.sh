#!/bin/bash
# =============================================================================
# InkTrace Curriculum Training Script
# From Stage 3 checkpoint → Stage 8 (multi-path)
# 
# Features:
#   - Progressive curriculum learning
#   - Mixed-stage training to prevent overfitting
#   - Checkpoint management for resume
#   - Differential learning rates (encoder vs decoder)
#   - Support for interruption and resume
#
# Usage:
#   ./train_curriculum.sh [start_phase] [gpu_id]
#   ./train_curriculum.sh 1 0       # Start from phase 1 on GPU 0
#   ./train_curriculum.sh 3 0       # Resume from phase 3
# =============================================================================

set -e  # Exit on error

# ======================== Configuration ========================
BASE_CKPT="checkpoints_dense/stage3_base.pth"  # Your stage3 checkpoint
SAVE_DIR="checkpoints_dense"
LOG_DIR="logs"
GPU_ID=${2:-0}
START_PHASE=${1:-1}

# Training hyperparameters (based on your 5090 48GB config)
BATCH_SIZE=128         # Your tested config
EPOCH_LENGTH=50000     # Samples per epoch
NUM_WORKERS=8
RUST_THREADS=8

# Create directories
mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

# ======================== Helper Functions ========================
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] $1" | tee -a "$LOG_DIR/training.log"
}

check_checkpoint() {
    local ckpt=$1
    if [ -f "$ckpt" ]; then
        echo "$ckpt"
    else
        echo ""
    fi
}

# Get the latest checkpoint for a phase
get_phase_ckpt() {
    local phase=$1
    local ckpt="$SAVE_DIR/phase${phase}_final.pth"
    check_checkpoint "$ckpt"
}

# ======================== Training Phases ========================
# Phase 1: Transition to Continuous (Frozen Encoder)
# Stage 4 (continuous 2-3 segments), frozen encoder to adapt decoder
train_phase1() {
    log "===== Phase 1: Continuous Introduction (Frozen Encoder) ====="
    
    local init_ckpt=$(get_phase_ckpt 0)
    [ -z "$init_ckpt" ] && init_ckpt="$BASE_CKPT"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python train_dense.py \
        --stage 4 \
        --init_from "$init_ckpt" \
        --freeze_encoder \
        --lr 5e-4 \
        --batch_size $BATCH_SIZE \
        --epochs 25 \
        --epoch_length $EPOCH_LENGTH \
        --num_workers $NUM_WORKERS \
        --rust_threads $RUST_THREADS \
        --save_dir "$SAVE_DIR" \
        --vis_interval 3
    
    # Save phase checkpoint
    cp "$SAVE_DIR/best_dense_model.pth" "$SAVE_DIR/phase1_final.pth"
    log "Phase 1 completed. Checkpoint: $SAVE_DIR/phase1_final.pth"
}

# Phase 2: Continuous Training (End-to-End, Stage 4-5 mixed)
train_phase2() {
    log "===== Phase 2: Continuous Deep Training (E2E) ====="
    
    local init_ckpt=$(get_phase_ckpt 1)
    [ -z "$init_ckpt" ] && { log "ERROR: Phase 1 checkpoint not found"; exit 1; }
    
    # Stage 5 with lower LR for fine-tuning
    CUDA_VISIBLE_DEVICES=$GPU_ID python train_dense.py \
        --stage 5 \
        --init_from "$init_ckpt" \
        --lr 3e-4 \
        --batch_size $BATCH_SIZE \
        --epochs 35 \
        --epoch_length $EPOCH_LENGTH \
        --num_workers $NUM_WORKERS \
        --rust_threads $RUST_THREADS \
        --save_dir "$SAVE_DIR" \
        --vis_interval 3
    
    cp "$SAVE_DIR/best_dense_model.pth" "$SAVE_DIR/phase2_final.pth"
    log "Phase 2 completed. Checkpoint: $SAVE_DIR/phase2_final.pth"
}

# Phase 3: Continuous Mastery (Stage 6, longer sequences)
train_phase3() {
    log "===== Phase 3: Continuous Mastery (Stage 6) ====="
    
    local init_ckpt=$(get_phase_ckpt 2)
    [ -z "$init_ckpt" ] && { log "ERROR: Phase 2 checkpoint not found"; exit 1; }
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python train_dense.py \
        --stage 6 \
        --init_from "$init_ckpt" \
        --lr 2e-4 \
        --batch_size $BATCH_SIZE \
        --epochs 45 \
        --epoch_length $EPOCH_LENGTH \
        --num_workers $NUM_WORKERS \
        --rust_threads $RUST_THREADS \
        --save_dir "$SAVE_DIR" \
        --vis_interval 3
    
    cp "$SAVE_DIR/best_dense_model.pth" "$SAVE_DIR/phase3_final.pth"
    log "Phase 3 completed. Checkpoint: $SAVE_DIR/phase3_final.pth"
}

# Phase 4: Multi-Path Introduction (Stage 7)
train_phase4() {
    log "===== Phase 4: Multi-Path Introduction (Stage 7) ====="
    
    local init_ckpt=$(get_phase_ckpt 3)
    [ -z "$init_ckpt" ] && { log "ERROR: Phase 3 checkpoint not found"; exit 1; }
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python train_dense.py \
        --stage 7 \
        --init_from "$init_ckpt" \
        --lr 2e-4 \
        --batch_size $BATCH_SIZE \
        --epochs 35 \
        --epoch_length $EPOCH_LENGTH \
        --num_workers $NUM_WORKERS \
        --rust_threads $RUST_THREADS \
        --save_dir "$SAVE_DIR" \
        --vis_interval 3
    
    cp "$SAVE_DIR/best_dense_model.pth" "$SAVE_DIR/phase4_final.pth"
    log "Phase 4 completed. Checkpoint: $SAVE_DIR/phase4_final.pth"
}

# Phase 5: Multi-Path Mastery (Stage 8)
train_phase5() {
    log "===== Phase 5: Multi-Path Mastery (Stage 8) ====="
    
    local init_ckpt=$(get_phase_ckpt 4)
    [ -z "$init_ckpt" ] && { log "ERROR: Phase 4 checkpoint not found"; exit 1; }
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python train_dense.py \
        --stage 8 \
        --init_from "$init_ckpt" \
        --lr 1e-4 \
        --batch_size $BATCH_SIZE \
        --epochs 50 \
        --epoch_length $EPOCH_LENGTH \
        --num_workers $NUM_WORKERS \
        --rust_threads $RUST_THREADS \
        --save_dir "$SAVE_DIR" \
        --vis_interval 3
    
    cp "$SAVE_DIR/best_dense_model.pth" "$SAVE_DIR/phase5_final.pth"
    log "Phase 5 completed. Checkpoint: $SAVE_DIR/phase5_final.pth"
}

# Phase 6: Mixed Consolidation (防止遗忘)
# Randomly sample from all stages to consolidate learning
train_phase6() {
    log "===== Phase 6: Mixed Consolidation ====="
    
    local init_ckpt=$(get_phase_ckpt 5)
    [ -z "$init_ckpt" ] && { log "ERROR: Phase 5 checkpoint not found"; exit 1; }
    
    # Train on each stage briefly to prevent catastrophic forgetting
    for stage in 2 4 6 8; do
        log "Consolidation: Stage $stage"
        CUDA_VISIBLE_DEVICES=$GPU_ID python train_dense.py \
            --stage $stage \
            --init_from "$init_ckpt" \
            --lr 5e-5 \
            --batch_size $BATCH_SIZE \
            --epochs 15 \
            --epoch_length 30000 \
            --num_workers $NUM_WORKERS \
            --rust_threads $RUST_THREADS \
            --save_dir "$SAVE_DIR" \
            --vis_interval 5
        
        init_ckpt="$SAVE_DIR/best_dense_model.pth"
    done
    
    cp "$SAVE_DIR/best_dense_model.pth" "$SAVE_DIR/phase6_final.pth"
    cp "$SAVE_DIR/best_dense_model.pth" "$SAVE_DIR/final_model.pth"
    log "Phase 6 (Consolidation) completed. Final model: $SAVE_DIR/final_model.pth"
}

# ======================== Main Execution ========================
log "=========================================="
log "InkTrace Curriculum Training Started"
log "Start Phase: $START_PHASE, GPU: $GPU_ID"
log "Base Checkpoint: $BASE_CKPT"
log "=========================================="

# Check base checkpoint exists
if [ ! -f "$BASE_CKPT" ]; then
    log "ERROR: Base checkpoint not found: $BASE_CKPT"
    log "Please copy your stage3 checkpoint to this location."
    exit 1
fi

# Execute phases based on start point
case $START_PHASE in
    1) train_phase1; train_phase2; train_phase3; train_phase4; train_phase5; train_phase6 ;;
    2) train_phase2; train_phase3; train_phase4; train_phase5; train_phase6 ;;
    3) train_phase3; train_phase4; train_phase5; train_phase6 ;;
    4) train_phase4; train_phase5; train_phase6 ;;
    5) train_phase5; train_phase6 ;;
    6) train_phase6 ;;
    *)
        log "Invalid phase: $START_PHASE (valid: 1-6)"
        exit 1
        ;;
esac

log "=========================================="
log "All training completed!"
log "Final model: $SAVE_DIR/final_model.pth"
log "=========================================="
