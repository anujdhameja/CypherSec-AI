#!/bin/bash

# Joern Multi-Language Verification System - Environment Setup Script
# This script automates the installation and configuration process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
JOERN_VERSION="latest"
PYTHON_MIN_VERSION="3.8"
JAVA_MIN_VERSION="8"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

version_compare() {
    # Compare version strings (returns 0 if $1 >= $2)
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

check_python() {
    log_info "Checking Python installation..."
    
    if check_command python3; then
        PYTHON_CMD="python3"
    elif check_command python; then
        PYTHON_CMD="python"
    else
        log_error "Python not found. Please install Python $PYTHON_MIN_VERSION or higher."
        return 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    
    if version_compare "$PYTHON_VERSION" "$PYTHON_MIN_VERSION"; then
        log_success "Python $PYTHON_VERSION found"
        return 0
    else
        log_error "Python $PYTHON_VERSION is too old. Minimum required: $PYTHON_MIN_VERSION"
        return 1
    fi
}

check_java() {
    log_info "Checking Java installation..."
    
    if check_command java; then
        JAVA_VERSION=$(java -version 2>&1 | head -n1 | cut -d'"' -f2 | cut -d'.' -f1-2)
        
        # Handle Java version format changes (e.g., "11.0.1" vs "1.8.0")
        if [[ $JAVA_VERSION == 1.* ]]; then
            JAVA_MAJOR=$(echo "$JAVA_VERSION" | cut -d'.' -f2)
        else
            JAVA_MAJOR=$(echo "$JAVA_VERSION" | cut -d'.' -f1)
        fi
        
        if [ "$JAVA_MAJOR" -ge "$JAVA_MIN_VERSION" ]; then
            log_success "Java $JAVA_VERSION found"
            return 0
        else
            log_error "Java $JAVA_VERSION is too old. Minimum required: Java $JAVA_MIN_VERSION"
            return 1
        fi
    else
        log_error "Java not found. Please install Java $JAVA_MIN_VERSION or higher."
        return 1
    fi
}

install_python_dependencies() {
    log_info "Installing Python dependencies..."
    
    cd "$PROJECT_DIR"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        $PYTHON_CMD -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        log_success "Python dependencies installed"
    else
        log_warning "requirements.txt not found, installing basic dependencies"
        pip install pathlib typing dataclasses
    fi
}

download_joern() {
    log_info "Downloading Joern CLI..."
    
    cd "$PROJECT_DIR"
    
    # Check if Joern already exists
    if [ -d "joern-cli" ]; then
        log_warning "Joern CLI directory already exists"
        read -p "Do you want to re-download Joern? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping Joern download"
            return 0
        fi
        rm -rf joern-cli
    fi
    
    # Download Joern
    if check_command wget; then
        wget -q --show-progress https://github.com/joernio/joern/releases/latest/download/joern-cli.zip
    elif check_command curl; then
        curl -L -o joern-cli.zip https://github.com/joernio/joern/releases/latest/download/joern-cli.zip
    else
        log_error "Neither wget nor curl found. Please install one of them."
        return 1
    fi
    
    # Extract Joern
    if check_command unzip; then
        unzip -q joern-cli.zip
        rm joern-cli.zip
    else
        log_error "unzip not found. Please install unzip."
        return 1
    fi
    
    # Set permissions
    chmod +x joern-cli/*.bat
    chmod +x joern-cli/*.sh 2>/dev/null || true  # Ignore errors for .sh files that might not exist
    
    log_success "Joern CLI downloaded and extracted"
}

create_configuration() {
    log_info "Creating configuration file..."
    
    cd "$PROJECT_DIR"
    
    if [ -f "joern_verification_config.json" ]; then
        log_warning "Configuration file already exists"
        read -p "Do you want to overwrite it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Keeping existing configuration"
            return 0
        fi
    fi
    
    # Copy basic configuration template
    if [ -f "examples/configurations/basic_config.json" ]; then
        cp examples/configurations/basic_config.json joern_verification_config.json
        log_success "Configuration file created from template"
    else
        # Create minimal configuration
        cat > joern_verification_config.json << 'EOF'
{
  "system": {
    "joern_path": "joern-cli",
    "output_base_dir": "verification_output",
    "temp_dir": "temp_test_files",
    "max_concurrent_tests": 1,
    "cleanup_temp_files": true,
    "verbose_logging": false
  },
  "languages": {
    "python": {
      "name": "Python",
      "file_extension": ".py",
      "tool_name": "pysrc2cpg.bat",
      "command_template": "{tool_path} {memory_allocation} {input_file} --output {output_dir}",
      "memory_allocation": "-J-Xmx4g",
      "timeout_seconds": 300,
      "alternative_tools": ["tree-sitter", "ast", "libcst"]
    },
    "java": {
      "name": "Java",
      "file_extension": ".java",
      "tool_name": "javasrc2cpg.bat",
      "command_template": "{tool_path} {memory_allocation} {input_file} --output {output_dir}",
      "memory_allocation": "-J-Xmx4g",
      "timeout_seconds": 300,
      "alternative_tools": ["eclipse-jdt", "spoon", "javaparser"]
    }
  }
}
EOF
        log_success "Minimal configuration file created"
    fi
}

validate_installation() {
    log_info "Validating installation..."
    
    cd "$PROJECT_DIR"
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Run validation
    if $PYTHON_CMD -m joern_verification.main --validate; then
        log_success "Installation validation passed"
        return 0
    else
        log_error "Installation validation failed"
        return 1
    fi
}

run_quick_test() {
    log_info "Running quick verification test..."
    
    cd "$PROJECT_DIR"
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Run quick test
    if $PYTHON_CMD -m joern_verification.main --languages python --timeout 60 --summary-only; then
        log_success "Quick test passed"
        return 0
    else
        log_warning "Quick test failed, but installation may still be functional"
        return 1
    fi
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -q, --quick         Skip interactive prompts"
    echo "  -t, --test          Run quick test after installation"
    echo "  --skip-joern        Skip Joern download"
    echo "  --skip-python       Skip Python dependency installation"
    echo "  --skip-config       Skip configuration file creation"
    echo ""
    echo "Examples:"
    echo "  $0                  Interactive installation"
    echo "  $0 --quick --test   Quick installation with test"
    echo "  $0 --skip-joern     Install without downloading Joern"
}

main() {
    local QUICK_MODE=false
    local RUN_TEST=false
    local SKIP_JOERN=false
    local SKIP_PYTHON=false
    local SKIP_CONFIG=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_usage
                exit 0
                ;;
            -q|--quick)
                QUICK_MODE=true
                shift
                ;;
            -t|--test)
                RUN_TEST=true
                shift
                ;;
            --skip-joern)
                SKIP_JOERN=true
                shift
                ;;
            --skip-python)
                SKIP_PYTHON=true
                shift
                ;;
            --skip-config)
                SKIP_CONFIG=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    log_info "Starting Joern Multi-Language Verification System setup..."
    
    # Check prerequisites
    if ! check_python; then
        exit 1
    fi
    
    if ! check_java; then
        exit 1
    fi
    
    # Install Python dependencies
    if [ "$SKIP_PYTHON" = false ]; then
        if ! install_python_dependencies; then
            log_error "Failed to install Python dependencies"
            exit 1
        fi
    fi
    
    # Download Joern
    if [ "$SKIP_JOERN" = false ]; then
        if ! download_joern; then
            log_error "Failed to download Joern"
            exit 1
        fi
    fi
    
    # Create configuration
    if [ "$SKIP_CONFIG" = false ]; then
        create_configuration
    fi
    
    # Validate installation
    if ! validate_installation; then
        log_error "Installation validation failed"
        exit 1
    fi
    
    # Run quick test if requested
    if [ "$RUN_TEST" = true ]; then
        run_quick_test
    fi
    
    log_success "Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Run verification: python -m joern_verification.main"
    echo "3. Check examples: ls examples/"
    echo "4. Read documentation: cat README.md"
}

# Run main function
main "$@"