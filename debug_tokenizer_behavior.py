#!/usr/bin/env python3
"""
Debug tokenizer behavior to understand why some nodes still fail
"""

import sys
sys.path.append('src')
from src.utils.functions.parse import tokenizer

def debug_tokenizer():
    print("=" * 60)
    print("DEBUGGING TOKENIZER BEHAVIOR")
    print("=" * 60)
    
    # Test different inputs
    test_cases = [
        # Individual words (what nodes contain)
        'void',
        'buffer',
        'strcpy',
        'buffer_overflow_test',
        
        # Full code context (what training data contains)
        'void buffer_overflow_test(char *input) { char buffer[8]; strcpy(buffer, input); }',
        'char buffer[8];',
        'strcpy(buffer, input);',
        'void buffer_overflow_test',
    ]
    
    print("\n🔍 Testing tokenizer on different inputs:")
    
    for test_input in test_cases:
        try:
            tokens = tokenizer(test_input)
            print(f"\nInput: '{test_input}'")
            print(f"Tokens: {tokens}")
            
            # Check if any tokens are generic (FUN1, VAR1, etc.)
            generic_tokens = [t for t in tokens if t.startswith(('FUN', 'VAR'))]
            if generic_tokens:
                print(f"  ✓ Contains generic tokens: {generic_tokens}")
            else:
                print(f"  ❌ No generic tokens found")
                
        except Exception as e:
            print(f"Error tokenizing '{test_input}': {e}")
    
    print(f"\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("The tokenizer only converts to FUN1/VAR1 when it sees:")
    print("1. Function calls in context: func_name() → FUN1")
    print("2. Variable declarations in context: int var_name → VAR1")
    print("")
    print("But individual words like 'buffer' or 'strcpy' stay as-is!")
    print("This explains why they're not found in the Word2Vec vocabulary.")

if __name__ == "__main__":
    debug_tokenizer()