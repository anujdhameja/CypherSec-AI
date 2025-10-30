#!/usr/bin/env python3
"""
Add vulnerability labels to the multi-language dataset for testing purposes
"""

import json
import random

def add_vulnerability_labels():
    """Add vulnerability labels to the multi-language dataset"""
    
    # Load the dataset
    with open('data/raw/dataset_tester_multilang.json', 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} samples")
    
    # Add labels based on simple heuristics
    for item in dataset:
        func_code = item['func'].lower()
        
        # Simple vulnerability detection heuristics
        vulnerable = False
        
        # Check for common vulnerability patterns
        vuln_patterns = [
            'strcpy', 'strcat', 'sprintf', 'gets', 'scanf',  # C/C++ buffer overflows
            'system(', 'exec(', 'eval(',  # Command injection
            'malloc', 'free',  # Memory management
            'sql', 'query',  # SQL injection hints
            'buffer', 'overflow'  # Direct mentions
        ]
        
        for pattern in vuln_patterns:
            if pattern in func_code:
                vulnerable = True
                break
        
        # Add some randomness to make it more realistic (30% chance of flipping)
        if random.random() < 0.3:
            vulnerable = not vulnerable
        
        # Add the target label
        item['target'] = 1 if vulnerable else 0
    
    # Calculate distribution
    vuln_count = sum(1 for item in dataset if item['target'] == 1)
    safe_count = len(dataset) - vuln_count
    
    print(f"Added labels:")
    print(f"  Vulnerable: {vuln_count} ({vuln_count/len(dataset):.1%})")
    print(f"  Safe: {safe_count} ({safe_count/len(dataset):.1%})")
    
    # Save the updated dataset
    output_path = 'data/raw/dataset_tester_multilang_labeled.json'
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Saved labeled dataset to: {output_path}")
    
    # Show some examples
    print("\nSample labeled entries:")
    for i, item in enumerate(dataset[:5]):
        label = "VULNERABLE" if item['target'] == 1 else "SAFE"
        func_preview = item['func'][:50].replace('\n', ' ')
        print(f"  {i+1}. [{label}] {item['language']}: {func_preview}...")

if __name__ == "__main__":
    random.seed(42)  # For reproducible results
    add_vulnerability_labels()