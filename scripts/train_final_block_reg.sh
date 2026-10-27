#!/bin/bash
# ==============================================================================
# Script Pelatihan Akhir Model BiLSTM ECG-to-BP dengan Regularisasi & Block-Split
# ==============================================================================
# Konfigurasi:
#   - Split Mode: Block (Generalisasi Lintas-Rekaman)
#   - Optimizer: AdamW (lr=3e-4, weight_decay=1e-2)
#   - Regularisasi: LSTM Dropout 0.3, FC Dropout 0.3
#   - Arsitektur: BiLSTM Patch-25 + 25-Fitur Proyeksi (64) + Adaptive Gating
#   - Loss: Smooth L1 + Konsistensi Fisiologis (lambda=0.10)
#   - Checkpoint: checkpoints/run_adamw_block_reg
# ==============================================================================

MODE=${1:-"tmux"}
EPOCHS=${2:-"30"}
BATCH_SIZE=${3:-"128"}

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

# Deteksi python environment
if [ -d ".venv" ]; then
    PYTHON_EXEC="$PROJECT_DIR/.venv/bin/python"
elif [ -d "venv" ]; then
    PYTHON_EXEC="$PROJECT_DIR/venv/bin/python"
elif [ -d "/mnt/nas-hpg9/djakyharis/ecg-bp-clean/.venv" ]; then
    PYTHON_EXEC="/mnt/nas-hpg9/djakyharis/ecg-bp-clean/.venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

CHECKPOINT_DIR="$PROJECT_DIR/checkpoints/run_adamw_block_reg"
LOG_FILE="$CHECKPOINT_DIR/train.log"
mkdir -p "$CHECKPOINT_DIR"

CMD="PYTHONPATH=. $PYTHON_EXEC src/train.py \
    --parts all \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --split-mode block \
    --optimizer adamw \
    --lr 3e-4 \
    --weight-decay 1e-2 \
    --scheduler-type cosine \
    --min-lr 1e-5 \
    --loss-fn smooth_l1 \
    --lambda-phys 0.10 \
    --lstm-dropout 0.3 \
    --fc-dropout 0.3 \
    --patch-size 25 \
    --use-projection \
    --proj-dim 64 \
    --use-gating \
    --feature-set all25 \
    --device auto \
    --amp \
    --checkpoint-dir $CHECKPOINT_DIR \
    --log-file $LOG_FILE"

echo "=============================================================================="
echo "  PELATIHAN AKHIR MODEL ECG-BP (REGULARISASI + BLOCK SPLIT)"
echo "=============================================================================="
echo "  Direktori Proyek: $PROJECT_DIR"
echo "  Python Exec     : $PYTHON_EXEC"
echo "  Mode Eksekusi   : $MODE"
echo "  Target Epoch    : $EPOCHS"
echo "  Batch Size      : $BATCH_SIZE"
echo "  Checkpoint Dir  : $CHECKPOINT_DIR"
echo "  File Log        : $LOG_FILE"
echo "=============================================================================="

if command -v nvidia-smi &> /dev/null; then
    echo "🎮 Status GPU Terdeteksi:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo "------------------------------------------------------------------------------"
fi

if [ "$MODE" == "tmux" ]; then
    SESSION_NAME="train_ecg_final"
    echo "🚀 Memulai sesi tmux: $SESSION_NAME"
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "⚠️ Sesi tmux '$SESSION_NAME' sudah berjalan. Lampirkan dengan: tmux attach -t $SESSION_NAME"
    else
        tmux new-session -d -s "$SESSION_NAME" "cd '$PROJECT_DIR' && $CMD; echo 'Pelatihan selesai. Tekan Enter untuk keluar.'; read"
        echo "✅ Sesi tmux berhasil diluncurkan!"
        echo "📌 Untuk memantau training: tmux attach -t $SESSION_NAME"
        echo "📌 Untuk melihat log live: tail -f $LOG_FILE"
    fi
elif [ "$MODE" == "nohup" ]; then
    echo "🚀 Menjalankan pelatihan di background via nohup..."
    nohup bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "✅ Proses berjalan di PID: $PID"
    echo "📌 Untuk melihat log: tail -f $LOG_FILE"
else
    echo "🚀 Menjalankan di foreground..."
    eval "$CMD"
fi
