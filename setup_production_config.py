#!/usr/bin/env python3
"""
Setup production configuration for large multi-language datasets
"""

import json
import os

def setup_production_config(dataset_name=None):
    """Setup production configuration for multi-language processing"""
    
    print("🔧 SETTING UP PRODUCTION CONFIGURATION")
    print("=" * 60)
    
    # Load current config
    with open('configs.json', 'r') as f:
        config = json.load(f)
    
    # Production settings for large datasets
    production_settings = {
        "create": {
            "filter_column_value": {
                "project": "FFmpeg"  # Standard project filter
            },
            "slice_size": 100,  # Optimal batch size for multi-language
            "joern_cli_dir": "joern/joern-cli/"
        },
        "paths": {
            "cpg": "data/cpg/",      # Main CPG directory for create task
            "joern": "data/joern/",   # Temporary Joern files
            "raw": "data/raw/",       # Raw datasets
            "input": "data/input/",   # Final input tensors
            "model": "models/",       # Trained models
            "tokens": "data/tokens/", # Tokenized data
            "w2v": "data/w2v/"       # Word2Vec models
        },
        "embed": {
            "nodes_dim": 205,        # Node dimension for embeddings
            "word2vec_args": {
                "vector_size": 100,   # Word2Vec vector size
                "alpha": 0.01,        # Learning rate
                "window": 5,          # Context window
                "min_count": 3,       # Minimum word frequency
                "sample": 1e-05,      # Subsampling threshold
                "workers": 4,         # Parallel workers
                "sg": 1,              # Skip-gram model
                "hs": 0,              # No hierarchical softmax
                "negative": 5         # Negative sampling
            },
            "edge_type": "Ast"       # AST edge type
        }
    }
    
    # Update config with production settings
    config.update(production_settings)
    
    # Set dataset name if provided
    if dataset_name:
        config["files"]["raw"] = dataset_name
        print(f"✅ Dataset set to: {dataset_name}")
    
    # Save updated config
    with open('configs.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Production configuration updated")
    
    # Verify directories exist
    print("\n🔍 Verifying directory structure...")
    for path_name, path_value in config["paths"].items():
        if not os.path.exists(path_value):
            os.makedirs(path_value, exist_ok=True)
            print(f"✅ Created directory: {path_value}")
        else:
            print(f"✅ Directory exists: {path_value}")
    
    # Show current configuration
    print(f"\n📋 Current Configuration:")
    print(f"   Dataset: {config['files']['raw']}")
    print(f"   Slice size: {config['create']['slice_size']}")
    print(f"   Project filter: {config['create']['filter_column_value']['project']}")
    print(f"   CPG path: {config['paths']['cpg']}")
    
    return config

def switch_dataset(dataset_name):
    """Quick function to switch dataset"""
    with open('configs.json', 'r') as f:
        config = json.load(f)
    
    config["files"]["raw"] = dataset_name
    
    with open('configs.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Switched to dataset: {dataset_name}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        setup_production_config(dataset_name)
    else:
        setup_production_config()