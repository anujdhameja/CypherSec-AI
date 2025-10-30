#!/usr/bin/env python3
"""
Command-line interface for explainable vulnerability prediction

This script provides explainable predictions using the attention-enhanced model
that shows which nodes (code lines) the model pays attention to.

Usage:
    python predict_with_explanation.py --demo
    python predict_with_explanation.py --test
    python predict_with_explanation.py --real-data
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append('src')

from src.inference.explainable_predictor import ExplainablePredictor
from torch_geometric.data import Data
import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Explainable vulnerability prediction with attention mechanism')
    
    parser.add_argument('--model', '-m', 
                       default='models/final_model_with_attention.pth',
                       help='Path to attention-enhanced model file')
    
    parser.add_argument('--w2v', '-w',
                       default='data/w2v/w2v.model', 
                       help='Path to Word2Vec model')
    
    parser.add_argument('--demo', '-d',
                       action='store_true',
                       help='Run demo with synthetic vulnerable/safe code examples')
    
    parser.add_argument('--test', '-t',
                       action='store_true', 
                       help='Test with real data from input folder')
    
    parser.add_argument('--real-data', '-r',
                       action='store_true',
                       help='Analyze real vulnerability data with explanations')
    
    parser.add_argument('--top-k', '-k',
                       type=int, default=10,
                       help='Number of top attention nodes to show (default: 10)')
    
    parser.add_argument('--save-viz', '-s',
                       help='Save attention visualization to file')
    
    args = parser.parse_args()
    
    print("="*90)
    print("🔍 EXPLAINABLE VULNERABILITY PREDICTION CLI")
    print("="*90)
    
    try:
        # Initialize explainable predictor
        predictor = ExplainablePredictor(
            model_path=args.model,
            w2v_path=args.w2v
        )
        
        if args.demo:
            print("\n🎮 Running explainable demo...")
            run_explainable_demo(predictor, args.top_k, args.save_viz)
            
        elif args.test:
            print("\n🧪 Running test with real data...")
            run_real_data_test(predictor, args.top_k, args.save_viz)
            
        elif args.real_data:
            print("\n📊 Analyzing real vulnerability data...")
            run_real_data_analysis(predictor, args.top_k, args.save_viz)
            
        else:
            print("\n💡 Usage examples:")
            print("  Demo mode:      python predict_with_explanation.py --demo")
            print("  Test mode:      python predict_with_explanation.py --test")
            print("  Real data:      python predict_with_explanation.py --real-data")
            print("  Custom top-k:   python predict_with_explanation.py --demo --top-k 15")
            print("  Save viz:       python predict_with_explanation.py --demo --save-viz attention.txt")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_explainable_demo(predictor, top_k, save_viz):
    """Run demo with explainable predictions"""
    
    print("Creating explainable vulnerability examples...")
    
    # Example 1: Buffer overflow vulnerability
    print("\n" + "="*70)
    print("EXAMPLE 1: BUFFER OVERFLOW VULNERABILITY")
    print("="*70)
    
    vuln_graph, vuln_labels = create_buffer_overflow_example()
    
    print("📄 Vulnerable code:")
    for i, line in enumerate(vuln_labels):
        print(f"   {i:2d}: {line}")
    
    result1 = predictor.predict_with_explanation(
        vuln_graph, 
        node_labels=vuln_labels, 
        top_k=top_k
    )
    
    predictor.print_detailed_explanation(result1, "Buffer Overflow Example")
    predictor.visualize_attention(result1, save_path=f"{save_viz}_vuln.txt" if save_viz else None)
    
    # Example 2: Safe code with proper bounds checking
    print("\n" + "="*70)
    print("EXAMPLE 2: SAFE CODE WITH BOUNDS CHECKING")
    print("="*70)
    
    safe_graph, safe_labels = create_safe_code_example()
    
    print("📄 Safe code:")
    for i, line in enumerate(safe_labels):
        print(f"   {i:2d}: {line}")
    
    result2 = predictor.predict_with_explanation(
        safe_graph, 
        node_labels=safe_labels, 
        top_k=top_k
    )
    
    predictor.print_detailed_explanation(result2, "Safe Code Example")
    predictor.visualize_attention(result2, save_path=f"{save_viz}_safe.txt" if save_viz else None)
    
    # Comparison
    print("\n" + "="*70)
    print("📊 EXPLAINABLE COMPARISON")
    print("="*70)
    
    compare_explanations(result1, result2, vuln_labels, safe_labels)


def run_real_data_test(predictor, top_k, save_viz):
    """Test with real data from input folder"""
    
    input_path = Path("data/input")
    
    if not input_path.exists():
        print(f"❌ Input folder not found: {input_path}")
        return
    
    pkl_files = list(input_path.glob("*.pkl"))
    
    if not pkl_files:
        print(f"❌ No .pkl files found in {input_path}")
        return
    
    print(f"📁 Found {len(pkl_files)} input files")
    
    # Load first file
    test_file = pkl_files[0]
    print(f"📄 Loading test data from: {test_file.name}")
    
    try:
        df = pd.read_pickle(test_file)
        print(f"✅ Loaded {len(df)} samples")
        
        # Test on first few samples with explanations
        test_samples = min(3, len(df))
        print(f"🧪 Testing on first {test_samples} samples with explanations...")
        
        for i in range(test_samples):
            sample = df.iloc[i]
            graph_data = sample['input']
            true_label = sample['target']
            
            print(f"\n" + "="*60)
            print(f"SAMPLE {i+1}: {'VULNERABLE' if true_label else 'SAFE'} (Ground Truth)")
            print("="*60)
            
            # Create node labels (simplified)
            num_nodes = graph_data.x.shape[0]
            node_labels = [f"Node_{j}" for j in range(num_nodes)]
            
            result = predictor.predict_with_explanation(
                graph_data, 
                node_labels=node_labels, 
                top_k=min(top_k, num_nodes)
            )
            
            # Add ground truth info
            result['true_label'] = int(true_label)
            result['sample_id'] = i
            
            predictor.print_detailed_explanation(result, f"Real Sample {i+1}")
            
            # Show correctness
            correct = result['is_vulnerable'] == true_label
            print(f"\n🎯 PREDICTION ACCURACY: {'✅ CORRECT' if correct else '❌ INCORRECT'}")
            
            if save_viz:
                predictor.visualize_attention(result, save_path=f"{save_viz}_sample_{i+1}.txt")
        
    except Exception as e:
        print(f"❌ Error testing with real data: {e}")


def run_real_data_analysis(predictor, top_k, save_viz):
    """Comprehensive analysis of real vulnerability data"""
    
    input_path = Path("data/input")
    
    if not input_path.exists():
        print(f"❌ Input folder not found: {input_path}")
        return
    
    pkl_files = list(input_path.glob("*.pkl"))
    
    if not pkl_files:
        print(f"❌ No .pkl files found in {input_path}")
        return
    
    print(f"📁 Analyzing {len(pkl_files)} data files...")
    
    all_results = []
    
    for file_idx, pkl_file in enumerate(pkl_files[:2]):  # Limit to first 2 files for demo
        print(f"\n📄 Processing: {pkl_file.name}")
        
        try:
            df = pd.read_pickle(pkl_file)
            print(f"   Loaded {len(df)} samples")
            
            # Analyze a subset
            sample_size = min(10, len(df))
            samples = df.sample(n=sample_size, random_state=42)
            
            for i, (idx, sample) in enumerate(samples.iterrows()):
                graph_data = sample['input']
                true_label = sample['target']
                
                # Create basic node labels
                num_nodes = graph_data.x.shape[0]
                node_labels = [f"Node_{j}" for j in range(num_nodes)]
                
                result = predictor.predict_with_explanation(
                    graph_data, 
                    node_labels=node_labels, 
                    top_k=min(5, num_nodes)  # Fewer for batch processing
                )
                
                result['true_label'] = int(true_label)
                result['file_idx'] = file_idx
                result['sample_idx'] = i
                result['original_idx'] = idx
                
                all_results.append(result)
                
                # Show progress
                if (i + 1) % 5 == 0:
                    print(f"   Processed {i + 1}/{sample_size} samples")
        
        except Exception as e:
            print(f"❌ Error processing {pkl_file.name}: {e}")
    
    # Analyze results
    if all_results:
        analyze_batch_results(all_results, save_viz)


def analyze_batch_results(results, save_viz):
    """Analyze batch results and provide insights"""
    
    print(f"\n" + "="*70)
    print("📊 BATCH ANALYSIS RESULTS")
    print("="*70)
    
    total_samples = len(results)
    correct_predictions = sum(1 for r in results if r['is_vulnerable'] == r['true_label'])
    accuracy = correct_predictions / total_samples
    
    # Basic statistics
    print(f"\n📈 Overall Statistics:")
    print(f"   Total samples analyzed: {total_samples}")
    print(f"   Correct predictions: {correct_predictions}")
    print(f"   Accuracy: {accuracy:.1%}")
    
    # Separate by true labels
    vulnerable_results = [r for r in results if r['true_label'] == 1]
    safe_results = [r for r in results if r['true_label'] == 0]
    
    print(f"\n🎯 By Ground Truth:")
    print(f"   Vulnerable samples: {len(vulnerable_results)}")
    print(f"   Safe samples: {len(safe_results)}")
    
    if vulnerable_results:
        vuln_correct = sum(1 for r in vulnerable_results if r['is_vulnerable'] == 1)
        vuln_recall = vuln_correct / len(vulnerable_results)
        print(f"   Vulnerable recall: {vuln_recall:.1%}")
        
        # Attention analysis for vulnerable samples
        vuln_max_attentions = [r['attention_stats']['max'] for r in vulnerable_results]
        vuln_high_attention_counts = [r['attention_stats']['num_high_attention'] for r in vulnerable_results]
        
        print(f"   Vulnerable attention stats:")
        print(f"     Avg max attention: {np.mean(vuln_max_attentions):.3f}")
        print(f"     Avg high attention nodes: {np.mean(vuln_high_attention_counts):.1f}")
    
    if safe_results:
        safe_correct = sum(1 for r in safe_results if r['is_vulnerable'] == 0)
        safe_precision = safe_correct / len(safe_results)
        print(f"   Safe precision: {safe_precision:.1%}")
        
        # Attention analysis for safe samples
        safe_max_attentions = [r['attention_stats']['max'] for r in safe_results]
        safe_high_attention_counts = [r['attention_stats']['num_high_attention'] for r in safe_results]
        
        print(f"   Safe attention stats:")
        print(f"     Avg max attention: {np.mean(safe_max_attentions):.3f}")
        print(f"     Avg high attention nodes: {np.mean(safe_high_attention_counts):.1f}")
    
    # Risk level distribution
    risk_levels = [r['risk_level'] for r in results]
    risk_counts = {level: risk_levels.count(level) for level in set(risk_levels)}
    
    print(f"\n🎯 Risk Level Distribution:")
    for level, count in risk_counts.items():
        print(f"   {level}: {count} ({count/total_samples:.1%})")
    
    # Show some interesting cases
    print(f"\n🔍 Interesting Cases:")
    
    # High confidence correct predictions
    high_conf_correct = [r for r in results if r['confidence'] > 0.8 and r['is_vulnerable'] == r['true_label']]
    if high_conf_correct:
        print(f"   High confidence correct: {len(high_conf_correct)} samples")
    
    # High attention vulnerable cases
    high_attention_vuln = [r for r in vulnerable_results if r['attention_stats']['max'] > 0.8]
    if high_attention_vuln:
        print(f"   High attention vulnerable: {len(high_attention_vuln)} samples")
        
        # Show one example
        example = high_attention_vuln[0]
        print(f"     Example: Max attention {example['attention_stats']['max']:.3f}, "
              f"Risk: {example['risk_level']}")
    
    # Save summary if requested
    if save_viz:
        summary_path = f"{save_viz}_batch_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Batch Analysis Summary\n")
            f.write("="*50 + "\n")
            f.write(f"Total samples: {total_samples}\n")
            f.write(f"Accuracy: {accuracy:.1%}\n")
            f.write(f"Risk levels: {risk_counts}\n")
        print(f"💾 Summary saved to: {summary_path}")


def create_buffer_overflow_example():
    """Create a realistic buffer overflow vulnerability example"""
    
    node_labels = [
        "void vulnerable_function(char* input) {",    # 0
        "    char buffer[64];",                       # 1 - Small buffer
        "    char temp[32];",                         # 2 - Another buffer
        "    if (input != NULL) {",                   # 3 - Basic check
        "        strcpy(buffer, input);",             # 4 - VULNERABLE: No bounds check
        "        strcat(buffer, \"_processed\");",    # 5 - VULNERABLE: Potential overflow
        "        sprintf(temp, \"%s\", buffer);",     # 6 - VULNERABLE: sprintf without bounds
        "        printf(temp);",                      # 7 - VULNERABLE: Format string
        "        return;",                            # 8
        "    }",                                      # 9
        "    printf(\"Input was null\");",            # 10
        "}",                                          # 11
    ]
    
    num_nodes = len(node_labels)
    
    # Create node features with emphasis on vulnerable patterns
    x = torch.randn(num_nodes, 100) * 0.7
    
    # Emphasize vulnerable lines (4, 5, 6, 7)
    for vuln_node in [4, 5, 6, 7]:
        x[vuln_node] += torch.randn(100) * 0.6
    
    # Create edges representing control and data flow
    edges = []
    
    # Sequential flow
    for i in range(num_nodes - 1):
        edges.extend([[i, i+1], [i+1, i]])
    
    # Data dependencies
    edges.extend([
        [1, 4], [4, 1],    # buffer used in strcpy
        [1, 5], [5, 1],    # buffer used in strcat
        [4, 5], [5, 4],    # strcpy result used in strcat
        [1, 6], [6, 1],    # buffer used in sprintf
        [2, 6], [6, 2],    # temp used in sprintf
        [5, 6], [6, 5],    # strcat result used in sprintf
        [6, 7], [7, 6],    # sprintf result used in printf
        [2, 7], [7, 2],    # temp used in printf
    ])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index), node_labels


def create_safe_code_example():
    """Create a safe code example with proper bounds checking"""
    
    node_labels = [
        "void safe_function(char* input) {",          # 0
        "    char buffer[64];",                       # 1
        "    char temp[32];",                         # 2
        "    if (input != NULL) {",                   # 3
        "        size_t len = strlen(input);",        # 4 - SAFE: Check length
        "        if (len < 50) {",                    # 5 - SAFE: Bounds check
        "            strncpy(buffer, input, 50);",    # 6 - SAFE: strncpy with limit
        "            buffer[50] = '\\0';",            # 7 - SAFE: Null termination
        "            strncat(buffer, \"_ok\", 13);",  # 8 - SAFE: strncat with limit
        "            snprintf(temp, 31, \"%s\", buffer);", # 9 - SAFE: snprintf with limit
        "            printf(\"%s\\n\", temp);",       # 10 - SAFE: Format string
        "        } else {",                           # 11
        "            printf(\"Input too long\\n\");", # 12
        "        }",                                  # 13
        "    }",                                      # 14
        "}",                                          # 15
    ]
    
    num_nodes = len(node_labels)
    
    # Create more uniform node features (safer pattern)
    x = torch.randn(num_nodes, 100) * 0.5
    
    # Create edges
    edges = []
    
    # Sequential flow
    for i in range(num_nodes - 1):
        edges.extend([[i, i+1], [i+1, i]])
    
    # Data dependencies (safer patterns)
    edges.extend([
        [4, 5], [5, 4],    # length check used in condition
        [1, 6], [6, 1],    # buffer used in strncpy
        [1, 8], [8, 1],    # buffer used in strncat
        [6, 8], [8, 6],    # strncpy result used in strncat
        [1, 9], [9, 1],    # buffer used in snprintf
        [2, 9], [9, 2],    # temp used in snprintf
        [8, 9], [9, 8],    # strncat result used in snprintf
        [9, 10], [10, 9],  # snprintf result used in printf
        [2, 10], [10, 2],  # temp used in printf
    ])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index), node_labels


def compare_explanations(vuln_result, safe_result, vuln_labels, safe_labels):
    """Compare explanations between vulnerable and safe code"""
    
    print(f"🔍 Attention Comparison:")
    print(f"   Vulnerable code max attention: {vuln_result['attention_stats']['max']:.3f}")
    print(f"   Safe code max attention: {safe_result['attention_stats']['max']:.3f}")
    
    print(f"\n🎯 Most Suspicious Lines:")
    print(f"   Vulnerable: {vuln_result['top_suspicious_nodes'][0][2]} "
          f"(attention: {vuln_result['top_suspicious_nodes'][0][1]:.3f})")
    print(f"   Safe: {safe_result['top_suspicious_nodes'][0][2]} "
          f"(attention: {safe_result['top_suspicious_nodes'][0][1]:.3f})")
    
    print(f"\n📊 Risk Assessment:")
    print(f"   Vulnerable code: {vuln_result['risk_level']}")
    print(f"   Safe code: {safe_result['risk_level']}")
    
    print(f"\n💡 Key Insights:")
    if vuln_result['is_vulnerable'] and not safe_result['is_vulnerable']:
        print(f"   ✅ Model correctly distinguishes vulnerable from safe code")
    
    if vuln_result['attention_stats']['max'] > safe_result['attention_stats']['max']:
        print(f"   ✅ Vulnerable code shows higher attention focus")
    
    if vuln_result['attention_stats']['num_high_attention'] > safe_result['attention_stats']['num_high_attention']:
        print(f"   ✅ Vulnerable code has more high-attention nodes")


if __name__ == "__main__":
    main()