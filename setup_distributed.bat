@echo off
REM Setup script for distributed training on Windows

echo === Distributed Training Setup ===

REM Default values
set RANK=0
set WORLD_SIZE=2
set MASTER_ADDR=localhost
set MASTER_PORT=12355
set MODE=

REM Parse command line arguments
:parse_args
if "%1"=="--master-node" (
    set MODE=master
    set RANK=0
    shift
    goto parse_args
)
if "%1"=="--worker-node" (
    set MODE=worker
    shift
    goto parse_args
)
if "%1"=="--rank" (
    set RANK=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--world-size" (
    set WORLD_SIZE=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--master-addr" (
    set MASTER_ADDR=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--master-port" (
    set MASTER_PORT=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--memory-only" (
    set MODE=memory
    shift
    goto parse_args
)
if "%1"=="--help" (
    goto show_usage
)
if "%1"=="" goto check_mode
shift
goto parse_args

:show_usage
echo Usage: %0 [OPTIONS]
echo Options:
echo   --master-node     Run as master node (rank 0)
echo   --worker-node     Run as worker node
echo   --rank N          Set rank for this node (default: 0)
echo   --world-size N    Total number of nodes (default: 2)
echo   --master-addr IP  Master node IP address (default: localhost)
echo   --master-port P   Master node port (default: 12355)
echo   --memory-only     Run memory-efficient single-machine training
echo   --help           Show this help message
echo.
echo Examples:
echo   # Master node (Machine 1):
echo   %0 --master-node --world-size 3 --master-addr 192.168.1.100
echo.
echo   # Worker node (Machine 2):
echo   %0 --worker-node --rank 1 --world-size 3 --master-addr 192.168.1.100
echo.
echo   # Memory-efficient single machine:
echo   %0 --memory-only
goto end

:check_mode
if "%MODE%"=="" (
    echo Error: Must specify --master-node, --worker-node, or --memory-only
    goto show_usage
)

REM Check Python and dependencies
echo Checking Python environment...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" >nul 2>&1
if errorlevel 1 (
    echo Error: PyTorch not found. Please install PyTorch first.
    goto end
)

python -c "import torch_geometric; print(f'PyTorch Geometric version: {torch_geometric.__version__}')" >nul 2>&1
if errorlevel 1 (
    echo Error: PyTorch Geometric not found. Please install it first.
    goto end
)

REM Check if data exists
if not exist "data\input" (
    echo Error: Input data directory not found. Please run data preprocessing first:
    echo   python main.py -c -e
    goto end
)

REM Check GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())" | findstr "True" >nul
if errorlevel 1 (
    echo No GPUs available, using CPU training
) else (
    for /f %%i in ('python -c "import torch; print(torch.cuda.device_count())"') do set GPU_COUNT=%%i
    echo GPUs available: %GPU_COUNT%
)

REM Run appropriate training mode
if "%MODE%"=="master" (
    echo Starting as MASTER node (rank %RANK%) with %WORLD_SIZE% total nodes
    echo Master address: %MASTER_ADDR%:%MASTER_PORT%
    echo Waiting for worker nodes to connect...
    
    python distributed_training.py --rank %RANK% --world-size %WORLD_SIZE% --master-addr %MASTER_ADDR% --master-port %MASTER_PORT%
) else if "%MODE%"=="worker" (
    echo Starting as WORKER node (rank %RANK%) connecting to master at %MASTER_ADDR%:%MASTER_PORT%
    
    python distributed_training.py --rank %RANK% --world-size %WORLD_SIZE% --master-addr %MASTER_ADDR% --master-port %MASTER_PORT%
) else if "%MODE%"=="memory" (
    echo Starting memory-efficient single-machine training...
    python memory_efficient_training.py
)

echo Training completed!

:end