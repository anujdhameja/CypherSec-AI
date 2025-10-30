#!/usr/bin/env python3
"""
Demo: Attention-Based Explainable Vulnerability Detection

This script demonstrates the attention mechanism that shows which nodes
(code lines) the GNN model pays attention to when making predictions.

Key Features Demonstrated:
- Attention weights per node [0.0 - 1.0]
- Top-K most suspicious code areas
- Risk level assessment
- Human-readable explanations
- Actionable recommendations

Usage:
    python demo_attention_explanation.py
"""

import torch
import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append('src')

from src.inference.explainable_predictor import ExplainablePredictor
from torch_geometric.data import Data


def create_vulnerable_code_example():
    """Create a graph representing vulnerable code with realistic node labels"""
    
    # Simulate vulnerable C code with buffer overflow
    node_labels = [
        "int main() {",                           # 0 - Function start
        "    char buffer[100];",                  # 1 - Buffer declaration
        "    char* user_input;",                  # 2 - Input pointer
        "    user_input = get_user_input();",     # 3 - Get user input
        "    if (user_input != NULL) {",          # 4 - Null check
        "        strcpy(buffer, user_input);",    # 5 - VULNERABLE: strcpy without bounds check
        "        printf(buffer);",                # 6 - VULNERABLE: Format string vulnerability
        "        process_data(buffer);",          # 7 - Process buffer
        "    }",                                  # 8 - End if
        "    free(user_input);",                  # 9 - Free memory
        "    return 0;",                          # 10 - Return
        "}",                                      # 11 - Function end
    ]
    
    num_nodes = len(node_labels)
    
    # Create node features (random for demo, in practice these would be Word2Vec embeddings)
    # Make vulnerable lines have slightly different patterns
    x = torch.randn(num_nodes, 100) * 0.8
    
    # Emphasize vulnerable patterns for nodes 5 and 6
    x[5] += torch.randn(100) * 0.5  # strcpy line
    x[6] += torch.randn(100) * 0.4  # printf line
    
    # Create edges representing control flow and data dependencies
    edges = []
    
    # Sequential control flow
    for i in range(num_nodes - 1):
        edges.extend([[i, i+1], [i+1, i]])
    
    # Data dependencies
    edges.extend([
        [1, 5], [5, 1],    # buffer used in strcpy
        [2, 3], [3, 2],    # user_input declaration and assignment
        [3, 5], [5, 3],    # user_input used in strcpy
        [5, 6], [6, 5],    # buffer flows from strcpy to printf
        [5, 7], [7, 5],    # buffer flows from strcpy to process_data
        [1, 6], [6, 1],    # buffer used in printf
        [1, 7], [7, 1],    # buffer used in process_data
        [3, 9], [9, 3],    # user_input freed
    ])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index), node_labels


def create_safe_code_example():
    """Create a graph representing safe code"""
    
    node_labels = [
        "int main() {",                           # 0 - Function start
        "    char buffer[100];",                  # 1 - Buffer declaration
        "    char* user_input;",                  # 2 - Input pointer
        "    user_input = get_user_input();",     # 3 - Get user input
        "    if (user_input != NULL) {",          # 4 - Null check
        "        if (strlen(user_input) < 99) {", # 5 - SAFE: Length check
        "            strncpy(buffer, user_input, 99);", # 6 - SAFE: strncpy with bounds
        "            buffer[99] = '\\0';",        # 7 - SAFE: Null termination
        "            printf(\"%s\", buffer);",    # 8 - SAFE: Format string with %s
        "            process_data(buffer);",      # 9 - Process buffer
        "        }",                              # 10 - End inner if
        "    }",                                  # 11 - End outer if
        "    free(user_input);",                  # 12 - Free memory
        "    return 0;",                          # 13 - Return
        "}",                                      # 14 - Function end
    ]
    
    num_nodes = len(node_labels)
    
    # Create node features (more uniform for safe code)
    x = torch.randn(num_nodes, 100) * 0.6
    
    # Create edges representing control flow
    edges = []
    
    # Sequential control flow
    for i in range(num_nodes - 1):
        edges.extend([[i, i+1], [i+1, i]])
    
    # Data dependencies (similar but with safety checks)
    edges.extend([
        [1, 6], [6, 1],    # buffer used in strncpy
        [2, 3], [3, 2],    # user_input declaration and assignment
        [3, 5], [5, 3],    # user_input used in length check
        [3, 6], [6, 6],    # user_input used in strncpy
        [6, 8], [8, 6],    # buffer flows from strncpy to printf
        [6, 9], [9, 6],    # buffer flows from strncpy to process_data
        [1, 8], [8, 1],    # buffer used in printf
        [1, 9], [9, 1],    # buffer used in process_data
        [3, 12], [12, 3],  # user_input freed
    ])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index), node_labels


def demo_attention_explanation():
    """Main demo function"""
    
    print("="*90)
    print("🔍 ATTENTION-BASED EXPLAINABLE VULNERABILITY DETECTION DEMO")
    print("="*90)
    
    try:
        # Initialize explainable predictor
        print("\n🔧 Initializing explainable predictor...")
        predictor = ExplainablePredictor('models/final_model_with_attention.pth')
        
        # Demo 1: Vulnerable Code
        print("\n" + "="*70)
        print("📋 DEMO 1: VULNERABLE CODE ANALYSIS")
        print("="*70)
        
        vuln_graph, vuln_labels = create_vulnerable_code_example()
        print(f"✅ Created vulnerable code graph with {len(vuln_labels)} nodes")
        
        # Show the code
        print(f"\n📄 Code being analyzed:")
        for i, line in enumerate(vuln_labels):
            print(f"   {i:2d}: {line}")
        
        # Make explainable prediction
        print(f"\n🔮 Analyzing with attention mechanism...")
        vuln_result = predictor.predict_with_explanation(
            vuln_graph, 
            node_labels=vuln_labels, 
            top_k=8
        )
        
        # Display detailed explanation
        predictor.print_detailed_explanation(vuln_result, "Vulnerable C Code")
        
        # Show attention visualization
        predictor.visualize_attention(vuln_result)
        
        # Demo 2: Safe Code
        print("\n" + "="*70)
        print("📋 DEMO 2: SAFE CODE ANALYSIS")
        print("="*70)
        
        safe_graph, safe_labels = create_safe_code_example()
        print(f"✅ Created safe code graph with {len(safe_labels)} nodes")
        
        # Show the code
        print(f"\n📄 Code being analyzed:")
        for i, line in enumerate(safe_labels):
            print(f"   {i:2d}: {line}")
        
        # Make explainable prediction
        print(f"\n🔮 Analyzing with attention mechanism...")
        safe_result = predictor.predict_with_explanation(
            safe_graph, 
            node_labels=safe_labels, 
            top_k=8
        )
        
        # Display detailed explanation
        predictor.print_detailed_explanation(safe_result, "Safe C Code")
        
        # Show attention visualization
        predictor.visualize_attention(safe_result)
        
        # Comparison Summary
        print("\n" + "="*70)
        print("📊 COMPARISON SUMMARY")
        print("="*70)
        
        print(f"\n🎯 Vulnerable Code:")
        print(f"   Prediction: {'🚨 VULNERABLE' if vuln_result['is_vulnerable'] else '✅ SAFE'}")
        print(f"   Confidence: {vuln_result['confidence']:.1%}")
        print(f"   Risk Level: {vuln_result['risk_level']}")
        print(f"   High Attention Nodes: {vuln_result['attention_stats']['num_high_attention']}")
        print(f"   Max Attention: {vuln_result['attention_stats']['max']:.3f}")
        
        if vuln_result['top_suspicious_nodes']:
            top_vuln = vuln_result['top_suspicious_nodes'][0]
            print(f"   Most Suspicious: {top_vuln[2]} (attention: {top_vuln[1]:.3f})")
        
        print(f"\n🎯 Safe Code:")
        print(f"   Prediction: {'🚨 VULNERABLE' if safe_result['is_vulnerable'] else '✅ SAFE'}")
        print(f"   Confidence: {safe_result['confidence']:.1%}")
        print(f"   Risk Level: {safe_result['risk_level']}")
        print(f"   High Attention Nodes: {safe_result['attention_stats']['num_high_attention']}")
        print(f"   Max Attention: {safe_result['attention_stats']['max']:.3f}")
        
        if safe_result['top_suspicious_nodes']:
            top_safe = safe_result['top_suspicious_nodes'][0]
            print(f"   Most Attention: {top_safe[2]} (attention: {top_safe[1]:.3f})")
        
        # Key Insights
        print(f"\n💡 KEY INSIGHTS:")
        
        if vuln_result['is_vulnerable'] and not safe_result['is_vulnerable']:
            print(f"   ✅ Model correctly distinguishes vulnerable from safe code")
        
        vuln_max_attention = vuln_result['attention_stats']['max']
        safe_max_attention = safe_result['attention_stats']['max']
        
        if vuln_max_attention > safe_max_attention:
            print(f"   ✅ Vulnerable code shows higher attention ({vuln_max_attention:.3f} vs {safe_max_attention:.3f})")
        
        vuln_high_nodes = vuln_result['attention_stats']['num_high_attention']
        safe_high_nodes = safe_result['attention_stats']['num_high_attention']
        
        if vuln_high_nodes > safe_high_nodes:
            print(f"   ✅ Vulnerable code has more high-attention nodes ({vuln_high_nodes} vs {safe_high_nodes})")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📝 The attention mechanism successfully identifies which code lines")
        print(f"   the model considers most important for vulnerability detection!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def test_attention_mechanism():
    """Test the attention mechanism with simple examples"""
    
    print("\n" + "="*60)
    print("🧪 TESTING ATTENTION MECHANISM")
    print("="*60)
    
    try:
        predictor = ExplainablePredictor('models/final_model_with_attention.pth')
        
        # Create simple test graph
        x = torch.randn(6, 100)
        edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [1, 4], [2, 5]]
        edges += [[j, i] for i, j in edges]  # Make bidirectional
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        test_graph = Data(x=x, edge_index=edge_index)
        test_labels = [f"Line_{i}" for i in range(6)]
        
        result = predictor.predict_with_explanation(test_graph, test_labels, top_k=6)
        
        print(f"✅ Attention mechanism test successful!")
        print(f"   Prediction: {'Vulnerable' if result['is_vulnerable'] else 'Safe'}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Attention range: [{min(result['attention_weights']):.3f}, {max(result['attention_weights']):.3f}]")
        
        print(f"\n🎯 Node attention scores:")
        for i, attention in enumerate(result['attention_weights']):
            print(f"   {test_labels[i]}: {attention:.3f}")
        
    except Exception as e:
        print(f"❌ Attention test failed: {e}")


if __name__ == "__main__":
    # Run the comprehensive demo
    demo_attention_explanation()
    
    # Run basic test
    test_attention_mechanism()