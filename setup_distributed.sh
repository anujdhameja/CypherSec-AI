#!/bin/bash
# Setup script for distributed training across multiple machines

echo "=== Distributed Training Setup ==="

# Check if running on Windows (adapt for your environment)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "Windows environment detected"
    PYTHON_CMD="python"
else
    echo "Unix environment detected"
    PYTHON_CMD="python3"
fi

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --master-node     Run as master node (rank 0)"
    echo "  --worker-node     Run as worker node"
    echo "  --rank N          Set rank for this node (default: 0)"
    echo "  --world-size N    Total number of nodes (default: 2)"
    echo "  --master-addr IP  Master node IP address (default: localhost)"
    echo "  --master-port P   Master node port (default: 12355)"
    echo "  --memory-only     Run memory-efficient single-machine training"
    echo "  --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Master node (Machine 1):"
    echo "  $0 --master-node --world-size 3 --master-addr 192.168.1.100"
    echo ""
    echo "  # Worker node (Machine 2):"
    echo "  $0 --worker-node --rank 1 --world-size 3 --master-addr 192.168.1.100"
    echo ""
    echo "  # Worker node (Machine 3):"
    echo "  $0 --worker-node --rank 2 --world-size 3 --master-addr 192.168.1.100"
    echo ""
    echo "  # Memory-efficient single machine:"
    echo "  $0 --memory-only"
}

# Default values
RANK=0
WORLD_SIZE=2
MASTER_ADDR="localhost"
MASTER_PORT="12355"
MODE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --master-node)
            MODE="master"
            RANK=0
            shift
            ;;
        --worker-node)
            MODE="worker"
            shift
            ;;
        --rank)
            RANK="$2"
            shift 2
            ;;
        --world-size)
            WORLD_SIZE="$2"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --memory-only)
            MODE="memory"
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ -z "$MODE" ]]; then
    echo "Error: Must specify --master-node, --worker-node, or --memory-only"
    usage
    exit 1
fi

# Check Python and dependencies
echo "Checking Python environment..."
$PYTHON_CMD -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
    echo "Error: PyTorch not found. Please install PyTorch first."
    exit 1
}

$PYTHON_CMD -c "import torch_geometric; print(f'PyTorch Geometric version: {torch_geometric.__version__}')" || {
    echo "Error: PyTorch Geometric not found. Please install it first."
    exit 1
}

# Check if data exists
if [[ ! -d "data/input" ]]; then
    echo "Error: Input data directory not found. Please run data preprocessing first:"
    echo "  python main.py -c -e"
    exit 1
fi

# Check available memory
echo "Checking system resources..."
if command -v free &> /dev/null; then
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
    echo "Available RAM: ${TOTAL_RAM}GB"
    if [[ $TOTAL_RAM -lt 8 ]]; then
        echo "Warning: Less than 8GB RAM available. Consider using --memory-only mode."
    fi
fi

# Check GPU availability
if $PYTHON_CMD -c "import torch; print('CUDA available:', torch.cuda.is_available())" | grep -q "True"; then
    GPU_COUNT=$($PYTHON_CMD -c "import torch; print(torch.cuda.device_count())")
    echo "GPUs available: $GPU_COUNT"
else
    echo "No GPUs available, using CPU training"
fi

# Run appropriate training mode
case $MODE in
    "master")
        echo "Starting as MASTER node (rank $RANK) with $WORLD_SIZE total nodes"
        echo "Master address: $MASTER_ADDR:$MASTER_PORT"
        echo "Waiting for worker nodes to connect..."
        
        $PYTHON_CMD distributed_training.py \
            --rank $RANK \
            --world-size $WORLD_SIZE \
            --master-addr $MASTER_ADDR \
            --master-port $MASTER_PORT
        ;;
        
    "worker")
        echo "Starting as WORKER node (rank $RANK) connecting to master at $MASTER_ADDR:$MASTER_PORT"
        
        $PYTHON_CMD distributed_training.py \
            --rank $RANK \
            --world-size $WORLD_SIZE \
            --master-addr $MASTER_ADDR \
            --master-port $MASTER_PORT
        ;;
        
    "memory")
        echo "Starting memory-efficient single-machine training..."
        $PYTHON_CMD memory_efficient_training.py
        ;;
esac

echo "Training completed!"