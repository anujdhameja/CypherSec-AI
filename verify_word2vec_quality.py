"""
Word2Vec Quality Verification Script
Checks if Word2Vec embeddings are properly trained and meaningful
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec
from collections import Counter
import src.utils.functions.parse as parse

print("="*80)
print("WORD2VEC QUALITY VERIFICATION")
print("="*80)

# 1. Check if Word2Vec model exists and loads
print("\n1. CHECKING WORD2VEC MODEL...")
w2v_path = 'data/w2v/w2v.model'

if not os.path.exists(w2v_path):
    print(f"❌ Word2Vec model not found at: {w2v_path}")
    exit(1)

try:
    w2v_model = Word2Vec.load(w2v_path)
    print(f"✓ Word2Vec model loaded successfully")
    print(f"  Vocabulary size: {len(w2v_model.wv)}")
    print(f"  Vector size: {w2v_model.wv.vector_size}")
    print(f"  Training epochs: {w2v_model.epochs}")
    print(f"  Window size: {w2v_model.window}")
    print(f"  Min count: {w2v_model.min_count}")
except Exception as e:
    print(f"❌ Error loading Word2Vec model: {e}")
    exit(1)

# 2. Check vocabulary quality
print(f"\n2. ANALYZING VOCABULARY QUALITY...")

vocab = list(w2v_model.wv.key_to_index.keys())
vocab_sample = vocab[:20]
print(f"Sample vocabulary (first 20): {vocab_sample}")

# Check for code-specific tokens
code_tokens = ['if', 'else', 'for', 'while', 'int', 'char', 'void', 'return', 'NULL', '=', '+', '-', '*', '/', '(', ')', '{', '}']
found_code_tokens = [token for token in code_tokens if token in w2v_model.wv]
print(f"Code tokens found: {len(found_code_tokens)}/{len(code_tokens)}")
print(f"Found: {found_code_tokens}")

# Check for variable/function placeholders
var_tokens = [token for token in vocab if token.startswith('VAR')]
fun_tokens = [token for token in vocab if token.startswith('FUN')]
print(f"Variable tokens (VAR*): {len(var_tokens)}")
print(f"Function tokens (FUN*): {len(fun_tokens)}")

if len(var_tokens) == 0 and len(fun_tokens) == 0:
    print("⚠️ WARNING: No VAR/FUN tokens found - tokenization might not be working")

# 3. Check training data quality
print(f"\n3. CHECKING TRAINING DATA...")

# Look for token files
tokens_path = Path('data/tokens')
if tokens_path.exists():
    token_files = list(tokens_path.glob('*.pkl'))
    print(f"Token files found: {len(token_files)}")
    
    if token_files:
        # Load first token file to check quality
        sample_file = token_files[0]
        try:
            df = pd.read_pickle(sample_file)
            print(f"Sample token file: {sample_file.name}")
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            
            if 'tokens' in df.columns:
                sample_tokens = df['tokens'].iloc[0] if len(df) > 0 else []
                print(f"  Sample tokens: {sample_tokens[:10]}...")
                
                # Check token quality
                all_tokens = []
                for tokens in df['tokens']:
                    all_tokens.extend(tokens)
                
                token_counts = Counter(all_tokens)
                print(f"  Total tokens: {len(all_tokens)}")
                print(f"  Unique tokens: {len(token_counts)}")
                print(f"  Most common: {token_counts.most_common(10)}")
                
        except Exception as e:
            print(f"❌ Error loading token file: {e}")
else:
    print("❌ No token files found - Word2Vec training data missing")

# 4. Test embedding quality
print(f"\n4. TESTING EMBEDDING QUALITY...")

# Check if embeddings are meaningful (not random)
test_words = ['if', 'else', 'for', 'while', 'VAR1', 'VAR2', 'FUN1']
available_words = [w for w in test_words if w in w2v_model.wv]

if len(available_words) >= 2:
    print(f"Testing with words: {available_words[:5]}")
    
    # Check vector properties
    vectors = [w2v_model.wv[word] for word in available_words[:5]]
    vectors_array = np.array(vectors)
    
    print(f"Vector statistics:")
    print(f"  Mean: {vectors_array.mean():.6f}")
    print(f"  Std: {vectors_array.std():.6f}")
    print(f"  Range: [{vectors_array.min():.6f}, {vectors_array.max():.6f}]")
    
    # Check if vectors are too similar (indicating poor training)
    if len(available_words) >= 2:
        word1, word2 = available_words[0], available_words[1]
        try:
            similarity = w2v_model.wv.similarity(word1, word2)
            print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
            
            if similarity > 0.9:
                print("⚠️ WARNING: Very high similarity - vectors might be too similar")
            elif similarity < -0.9:
                print("⚠️ WARNING: Very low similarity - vectors might be random")
            else:
                print("✓ Reasonable similarity")
        except Exception as e:
            print(f"❌ Error computing similarity: {e}")
    
    # Test most similar words
    try:
        if 'if' in w2v_model.wv:
            similar = w2v_model.wv.most_similar('if', topn=5)
            print(f"Words most similar to 'if': {similar}")
        elif available_words:
            word = available_words[0]
            similar = w2v_model.wv.most_similar(word, topn=5)
            print(f"Words most similar to '{word}': {similar}")
    except Exception as e:
        print(f"❌ Error finding similar words: {e}")

else:
    print("❌ No test words available in vocabulary")

# 5. Test tokenization pipeline
print(f"\n5. TESTING TOKENIZATION PIPELINE...")

test_code = """
int main() {
    int x = 10;
    if (x > 5) {
        printf("Hello");
        return 0;
    }
    return 1;
}
"""

try:
    tokens = parse.tokenizer(test_code)
    print(f"Test code tokenization:")
    print(f"  Input: {repr(test_code[:50])}...")
    print(f"  Tokens: {tokens[:15]}...")
    print(f"  Token count: {len(tokens)}")
    
    # Check how many tokens are in vocabulary
    tokens_in_vocab = [t for t in tokens if t in w2v_model.wv]
    print(f"  Tokens in W2V vocab: {len(tokens_in_vocab)}/{len(tokens)} ({len(tokens_in_vocab)/len(tokens)*100:.1f}%)")
    
    if len(tokens_in_vocab) < len(tokens) * 0.5:
        print("⚠️ WARNING: Less than 50% of tokens are in vocabulary")
    else:
        print("✓ Good vocabulary coverage")
        
except Exception as e:
    print(f"❌ Error in tokenization: {e}")

# 6. Overall assessment
print(f"\n" + "="*80)
print("OVERALL ASSESSMENT")
print("="*80)

issues = []
recommendations = []

# Check vocabulary size
if len(w2v_model.wv) < 500:
    issues.append("Very small vocabulary (< 500 words)")
    recommendations.append("Increase training data or reduce min_count")
elif len(w2v_model.wv) < 1000:
    issues.append("Small vocabulary (< 1000 words)")
    recommendations.append("Consider more training data")

# Check vector quality
if len(available_words) >= 2:
    vectors = [w2v_model.wv[word] for word in available_words[:5]]
    vectors_array = np.array(vectors)
    
    if vectors_array.std() < 0.1:
        issues.append("Very low vector variance - embeddings might be poorly trained")
        recommendations.append("Retrain Word2Vec with more epochs or different parameters")
    
    if abs(vectors_array.mean()) > 0.5:
        issues.append("High vector mean - embeddings might be biased")
        recommendations.append("Check training data quality")

# Check code token coverage
if len(found_code_tokens) < len(code_tokens) * 0.5:
    issues.append("Poor coverage of common code tokens")
    recommendations.append("Check if tokenization is working correctly")

if not issues:
    print("✅ WORD2VEC QUALITY LOOKS GOOD!")
    print("   Embeddings should provide meaningful features for training")
else:
    print("❌ WORD2VEC QUALITY ISSUES FOUND:")
    for issue in issues:
        print(f"   - {issue}")
    
    print(f"\n🔧 RECOMMENDATIONS:")
    for rec in recommendations:
        print(f"   - {rec}")
    
    print(f"\n💡 TO FIX:")
    print(f"   1. Run: python retrain_word2vec.py")
    print(f"   2. Or check the embed_task() in main.py")
    print(f"   3. Ensure you have enough training data in data/tokens/")

print(f"\n🎯 NEXT STEPS:")
if not issues:
    print("   Word2Vec is good - the feature quality issue is likely elsewhere")
    print("   Check how embeddings are applied to graph nodes")
else:
    print("   Fix Word2Vec training first, then retest model performance")
    print("   Expected improvement: 55% → 70%+ accuracy")