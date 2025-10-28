#!/usr/bin/env python3
"""
Fix for the embedding tokenization mismatch
Shows the current broken code and the fixed version
"""

print("=" * 80)
print("EMBEDDING TOKENIZATION FIX")
print("=" * 80)

print("\n🚨 CURRENT BROKEN CODE (src/prepare/embeddings.py, lines ~378-382):")
print("""
# BROKEN: Uses simple regex instead of proper tokenization
try:
    tokens = self._tok_re.findall(code_text)  # ❌ WRONG!
except TypeError as e:
    print(f"⚠️ Tokenization error for node {nid}: {e}")
    tokens = []

# This finds raw tokens like ['buffer', 'overflow', 'test']
# But Word2Vec vocabulary has ['VAR1', 'FUN1', 'void', 'char', ...]
# So 'buffer' is NOT FOUND → zero embedding
""")

print("\n✅ FIXED CODE:")
print("""
# FIXED: Use the same tokenizer as training phase
from src.utils.functions.parse import tokenizer

try:
    # Apply the SAME tokenization used during training
    # This converts 'buffer_overflow_test' → 'FUN1'
    # and 'buffer' → 'VAR1', etc.
    tokens = tokenizer(code_text)  # ✅ CORRECT!
except Exception as e:
    print(f"⚠️ Tokenization error for node {nid}: {e}")
    tokens = []

# Now tokens are ['FUN1', 'VAR1', 'void', 'char', ...]
# These ARE FOUND in Word2Vec vocabulary → valid embeddings
""")

print("\n🔧 IMPLEMENTATION:")

# Show the exact fix to apply
fix_code = '''
# In src/prepare/embeddings.py, at the top, add this import:
from src.utils.functions.parse import tokenizer

# In the NodesEmbedding class, replace lines ~378-382:
# OLD (BROKEN):
            # Tokenize safely
            try:
                tokens = self._tok_re.findall(code_text)
            except TypeError as e:
                print(f"⚠️ Tokenization error for node {nid}: {e}")
                print(f"   code_text type: {type(code_text)}, value: {repr(code_text)[:100]}")
                tokens = []

# NEW (FIXED):
            # Tokenize using the SAME function as training phase
            try:
                tokens = tokenizer(code_text)  # This applies FUN1/VAR1 conversion
            except Exception as e:
                print(f"⚠️ Tokenization error for node {nid}: {e}")
                print(f"   code_text type: {type(code_text)}, value: {repr(code_text)[:100]}")
                tokens = []
'''

print(fix_code)

print("\n📊 EXPECTED RESULTS AFTER FIX:")
print("Before fix:")
print("  - Node text: 'buffer_overflow_test'")
print("  - Regex tokens: ['buffer', 'overflow', 'test']")
print("  - Word2Vec lookup: NOT FOUND → zero embedding")
print("")
print("After fix:")
print("  - Node text: 'buffer_overflow_test'")
print("  - Tokenizer output: ['FUN1']")
print("  - Word2Vec lookup: FOUND → valid embedding")
print("")
print("This should fix the 40% zero features issue!")

print("\n💡 WHY THIS WORKS:")
print("1. Training phase: tokenizer('buffer_overflow_test') → ['FUN1']")
print("2. Word2Vec learns: 'FUN1' → [0.123, -0.456, ...]")
print("3. Embedding phase: tokenizer('buffer_overflow_test') → ['FUN1']")
print("4. Lookup: 'FUN1' in vocabulary → FOUND → valid embedding")
print("")
print("The key is using the SAME tokenizer in both phases!")