#!/bin/bash
# ML4VRP 2026 — VM Setup Script
# For Ubuntu with NVIDIA Quadro RTX 4000 (dual GPU)
# Usage: bash scripts/setup_vm.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJECT_DIR/data"

echo "=== ML4VRP VM Setup ==="
echo "Project: $PROJECT_DIR"

# 1. Check GPU
echo ""
echo "--- GPU Check ---"
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 2. Create virtual environment
echo "--- Python Environment ---"
if [ ! -d "$PROJECT_DIR/venv" ]; then
    python3 -m venv "$PROJECT_DIR/venv"
    echo "Created venv"
else
    echo "venv already exists"
fi
source "$PROJECT_DIR/venv/bin/activate"

# 3. Install PyTorch (CUDA 12.x)
echo ""
echo "--- Installing PyTorch ---"
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install PyG dependencies
echo ""
echo "--- Installing PyTorch Geometric ---"
TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))")
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html"
pip install torch-geometric

# 5. Install remaining dependencies
echo ""
echo "--- Installing Project Dependencies ---"
pip install pyvrp gymnasium numpy matplotlib

# 6. Download X-dataset
echo ""
echo "--- Downloading X-dataset ---"
mkdir -p "$DATA_DIR"
EXISTING=$(find "$DATA_DIR" -name "X-n*.vrp" 2>/dev/null | wc -l)
if [ "$EXISTING" -ge 50 ]; then
    echo "$EXISTING instances already in $DATA_DIR"
else
    TEMP_DIR=$(mktemp -d)
    git clone --depth 1 https://github.com/ML4VRP/ML4VRP2026.git "$TEMP_DIR/ml4vrp_repo"
    cp "$TEMP_DIR/ml4vrp_repo/Instances/cvrp/vrp/X-n"*.vrp "$DATA_DIR/" 2>/dev/null || true
    cd "$PROJECT_DIR"
    rm -rf "$TEMP_DIR"
    echo "Downloaded $(find "$DATA_DIR" -name "X-n*.vrp" | wc -l) instances"
fi

# 7. Run smoke tests
echo ""
echo "--- Smoke Tests ---"
cd "$PROJECT_DIR"
python -m src.main

# 8. Print launch command
echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start training, run:"
echo ""
echo "  source venv/bin/activate"
echo "  python -m src.main train \\"
echo "      --instance_path data \\"
echo "      --epochs 200 \\"
echo "      --batch_size 16 \\"
echo "      --manager_lr 1e-4 \\"
echo "      --episodes_per_epoch 8 \\"
echo "      --fp16 \\"
echo "      --ent_coeff 0.05 \\"
echo "      --save_interval 10 \\"
echo "      --curriculum_epochs 20"
echo ""
echo "To run in background (persists after SSH disconnect):"
echo ""
echo "  nohup python -m src.main train \\"
echo "      --instance_path data \\"
echo "      --epochs 200 \\"
echo "      --batch_size 16 \\"
echo "      --manager_lr 1e-4 \\"
echo "      --episodes_per_epoch 8 \\"
echo "      --fp16 \\"
echo "      --ent_coeff 0.05 \\"
echo "      --save_interval 10 \\"
echo "      --curriculum_epochs 20 \\"
echo "      > training.log 2>&1 &"
echo ""
echo "  tail -f training.log   # watch progress"
