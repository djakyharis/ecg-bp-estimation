#!/bin/bash
# ==============================================================================
# Script Otomasi Pelatihan Model BiLSTM ECG-to-BP di Server GPU Remote via SSH
# ==============================================================================
# Penggunaan:
#   bash scripts/train_remote_ssh.sh [tmux | nohup | foreground] [all | 1 | 1,2] [epochs]
#
# Contoh:
#   bash scripts/train_remote_ssh.sh tmux all 50
#   bash scripts/train_remote_ssh.sh nohup all 100
#   bash scripts/train_remote_ssh.sh foreground 1 15
# ==============================================================================

MODE=${1:-"tmux"}
PARTS=${2:-"all"}
EPOCHS=${3:-"500"}
BATCH_SIZE=${4:-"128"}

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

# Deteksi virtualenv jika ada
if [ -d ".venv" ]; then
    PYTHON_EXEC="$PROJECT_DIR/.venv/bin/python"
elif [ -d "venv" ]; then
    PYTHON_EXEC="$PROJECT_DIR/venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"
LOG_FILE="$CHECKPOINT_DIR/train.log"
mkdir -p "$CHECKPOINT_DIR"

CMD="PYTHONPATH=. $PYTHON_EXEC src/train.py \
    --parts $PARTS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --optimizer sgd \
    --lr 8e-4 \
    --momentum 0.9 \
    --weight-decay 0.1 \
    --loss-fn rmse \
    --loss-min-weight 0.15 \
    --loss-momentum 0.70 \
    --device auto \
    --amp \
    --auto-resume \
    --checkpoint-dir $CHECKPOINT_DIR \
    --log-file $LOG_FILE"

echo "=============================================================================="
echo "  SISTEM ESTIMASI TEKANAN DARAH (ECG-TO-BP) — GPU SSH LAUNCHER"
echo "=============================================================================="
echo "  Mode Eksekusi   : $MODE"
echo "  Dataset Parts   : $PARTS"
echo "  Target Epoch    : $EPOCHS"
echo "  Batch Size      : $BATCH_SIZE"
echo "  Checkpoint Dir  : $CHECKPOINT_DIR"
echo "  File Log        : $LOG_FILE"
echo "=============================================================================="

# Cek GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 Status GPU Terdeteksi:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo "------------------------------------------------------------------------------"
fi

if [ "$MODE" == "tmux" ]; then
    SESSION_NAME="ecg_bp_train"
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "⚠️ Sesi tmux '$SESSION_NAME' sudah ada!"
        echo "Gunakan 'tmux attach -t $SESSION_NAME' untuk melihatnya."
        exit 1
    fi
    echo "🚀 Memulai training di dalam sesi tmux: $SESSION_NAME"
    tmux new-session -d -s "$SESSION_NAME" "$CMD"
    echo "✅ Sesi tmux berhasil dibuat dan berjalan di background server."
    echo ""
    echo "📌 Panduan SSH Disconnect:"
    echo "   - Anda sekarang aman untuk menutup sesi SSH atau laptop Anda."
    echo "   - Untuk masuk ke sesi monitoring: tmux attach -t $SESSION_NAME"
    echo "   - Untuk keluar tanpa mematikan training: tekan Ctrl+B, lalu tekan D"
    echo "   - Untuk melihat log secara langsung: tail -f $LOG_FILE"

elif [ "$MODE" == "nohup" ]; then
    echo "🚀 Menjalankan via nohup di background..."
    nohup $CMD > /dev/null 2>&1 &
    PID=$!
    echo "✅ Berhasil dijalankan dengan PID: $PID"
    echo "Pantau training dengan: tail -f $LOG_FILE"

elif [ "$MODE" == "foreground" ]; then
    echo "🚀 Menjalankan langsung di foreground terminal..."
    eval $CMD

else
    echo "❌ Mode tidak dikenali: $MODE. Pilihan: tmux, nohup, foreground"
    exit 1
fi
