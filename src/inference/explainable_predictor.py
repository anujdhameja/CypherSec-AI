#!/usr/bin/env python3
"""
Explainable Vulnerability Predictor with Attention Mechanism

This predictor uses the AttentionDevignModel to provide explainable predictions
by showing which nodes (code lines) the model pays attention to.

Key Features:
- Identifies most suspicious code areas
- Provides attention scores for each node
- Gives human-readable explanations
- Risk level assessment
- Actionable recommendations

Usage:
    predictor = ExplainablePredictor('models/final_model_with_attention.pth')
    result = predictor.predict_with_explanation(graph_data)
    predictor.print_detailed_explanation(result)
"""

import torch
import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.process.attention_devign_model import AttentionDevignModel
from torch_geometric.data import Data
from gensim.models import Word2Vec


class ExplainablePredictor:
    """
    Explainable Vulnerability Predictor with Attention Mechanism
    
    This predictor not only predicts vulnerabilities but also explains
    WHY it made the prediction by showing attention weights.
    """
    
    def __init__(self, model_path='models/final_model_with_attention.pth', 
                 w2v_path='data/w2v/w2v.model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.w2v_path = w2v_path
        
        print("="*70)
        print("EXPLAINABLE VULNERABILITY PREDICTOR INITIALIZATION")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Model path: {os.path.abspath(model_path)}")
        print(f"Word2Vec path: {os.path.abspath(w2v_path)}")
        
        # Load attention-enhanced model
        print(f"\n🔧 Loading attention-enhanced model...")
        self.model = AttentionDevignModel(
            input_dim=100,      # Same as training
            output_dim=2,       # Binary classification
            hidden_dim=256,     # Same as training
            num_steps=5,        # Same as training
            dropout=0.2,        # Same as training
            pooling='mean_max'  # Dual pooling
        )
        
        # Load trained weights
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✅ Attention model weights loaded successfully")
        else:
            print(f"❌ Model file not found: {model_path}")
            print(f"💡 Tip: Run convert_model_to_attention() first")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load Word2Vec model
        print(f"\n📚 Loading Word2Vec model...")
        if os.path.exists(w2v_path):
            self.w2v = Word2Vec.load(w2v_path)
            print(f"✅ Word2Vec loaded (vocabulary: {len(self.w2v.wv)} words)")
        else:
            print(f"❌ Word2Vec file not found: {w2v_path}")
            raise FileNotFoundError(f"Word2Vec file not found: {w2v_path}")
        
        print(f"\n🎯 Explainable predictor ready!")
        print("="*70 + "\n")
    
    def predict_with_explanation(self, graph_data: Data, node_labels=None, top_k=10) -> dict:
        """
        Predict vulnerability with detailed explanation
        
        Args:
            graph_data: PyTorch Geometric Data object
            node_labels: Optional list of node descriptions/code lines
            top_k: Number of top attention nodes to include
        
        Returns:
            dict: Comprehensive prediction result with explanations
        """
        # Ensure graph is on correct device
        graph_data = graph_data.to(self.device)
        
        # Add batch dimension if not present
        if not hasattr(graph_data, 'batch'):
            graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            # Get predictions and attention weights
            output, attention_weights = self.model(graph_data, return_attention=True)
            
            # Convert to probabilities
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            confidence = probs[0][pred].item()
            
            prob_safe = probs[0][0].item()
            prob_vulnerable = probs[0][1].item()
            
            # Get detailed explanation
            explanation = self.model.get_attention_explanation(
                graph_data, node_labels=node_labels, top_k=top_k
            )
        
        # Combine basic prediction with explanation
        result = {
            # Basic prediction info
            'is_vulnerable': pred,
            'confidence': confidence,
            'prob_safe': prob_safe,
            'prob_vulnerable': prob_vulnerable,
            
            # Attention-based explanation
            'attention_weights': attention_weights.cpu().numpy(),
            'top_suspicious_nodes': explanation['top_nodes'],
            'attention_stats': explanation['attention_stats'],
            'risk_level': explanation['risk_assessment'],
            
            # Detailed analysis
            'explanation': self._generate_explanation(explanation, pred, confidence),
            'recommendations': self._generate_recommendations(explanation, pred),
            
            # Graph info
            'graph_info': {
                'num_nodes': graph_data.x.size(0),
                'num_edges': graph_data.edge_index.size(1) if graph_data.edge_index.numel() > 0 else 0,
                'node_labels': node_labels
            }
        }
        
        return result
    
    def _generate_explanation(self, explanation, pred, confidence):
        """Generate human-readable explanation"""
        
        is_vulnerable = pred == 1
        risk_level = explanation['risk_assessment']
        stats = explanation['attention_stats']
        top_nodes = explanation['top_nodes']
        
        explanation_text = []
        
        # Main prediction
        if is_vulnerable:
            explanation_text.append(f"🚨 The model predicts this code is VULNERABLE with {confidence:.1%} confidence.")
        else:
            explanation_text.append(f"✅ The model predicts this code is SAFE with {confidence:.1%} confidence.")
        
        # Risk assessment
        explanation_text.append(f"🎯 Risk Level: {risk_level}")
        
        # Attention analysis
        if stats['num_high_attention'] > 0:
            explanation_text.append(f"⚠️  The model found {stats['num_high_attention']} highly suspicious code areas (attention > 0.7).")
        
        if stats['max'] > 0.8:
            explanation_text.append(f"🔍 Maximum attention score: {stats['max']:.3f} - indicating strong focus on specific code patterns.")
        
        # Top suspicious areas
        if is_vulnerable and len(top_nodes) > 0:
            explanation_text.append(f"🎯 Most suspicious code areas:")
            for i, (node_id, attention, label) in enumerate(top_nodes[:3]):
                risk_emoji = "🔴" if attention > 0.7 else "🟡" if attention > 0.4 else "🟢"
                explanation_text.append(f"   {risk_emoji} Rank #{i+1}: {label} (attention: {attention:.3f})")
        
        return "\n".join(explanation_text)
    
    def _generate_recommendations(self, explanation, pred):
        """Generate actionable recommendations"""
        
        is_vulnerable = pred == 1
        risk_level = explanation['risk_assessment']
        stats = explanation['attention_stats']
        top_nodes = explanation['top_nodes']
        
        recommendations = []
        
        if is_vulnerable:
            if risk_level == "Critical":
                recommendations.append("🚨 CRITICAL: Immediate code review required!")
                recommendations.append("🔍 Focus on the highest attention areas first")
                recommendations.append("🛡️  Consider implementing additional security measures")
            
            elif risk_level == "High":
                recommendations.append("⚠️  HIGH PRIORITY: Schedule code review within 24 hours")
                recommendations.append("🔍 Examine flagged code areas for potential vulnerabilities")
            
            elif risk_level == "Medium":
                recommendations.append("📋 MEDIUM PRIORITY: Include in next security review cycle")
                recommendations.append("🔍 Consider additional testing for flagged areas")
            
            else:
                recommendations.append("📝 LOW PRIORITY: Monitor during routine code reviews")
            
            # Specific recommendations based on attention patterns
            if stats['num_high_attention'] >= 3:
                recommendations.append("🎯 Multiple high-attention areas detected - check for complex interactions")
            
            if stats['max'] > 0.9:
                recommendations.append("🔍 Extremely high attention on specific code - investigate immediately")
            
            # Top node recommendations
            if len(top_nodes) > 0:
                top_attention = top_nodes[0][1]
                if top_attention > 0.8:
                    recommendations.append(f"🎯 Pay special attention to: {top_nodes[0][2]}")
        
        else:
            recommendations.append("✅ Code appears safe based on current analysis")
            recommendations.append("📝 Continue following secure coding practices")
            
            if stats['max'] > 0.5:
                recommendations.append("🔍 Some areas show moderate attention - consider periodic review")
        
        return recommendations
    
    def print_detailed_explanation(self, result: dict, title: str = ""):
        """Print comprehensive explanation of the prediction"""
        
        print("\n" + "="*80)
        print("🔍 EXPLAINABLE VULNERABILITY ANALYSIS")
        print("="*80)
        
        if title:
            print(f"📄 Analysis: {title}")
        
        # Basic prediction
        status = "🚨 VULNERABLE" if result['is_vulnerable'] else "✅ SAFE"
        print(f"\n🎯 PREDICTION: {status} ({result['confidence']:.1%} confidence)")
        
        # Risk level
        risk_colors = {
            "Critical": "🔴",
            "High": "🟠", 
            "Medium": "🟡",
            "Low": "🟢"
        }
        risk_emoji = risk_colors.get(result['risk_level'], "⚪")
        print(f"🎯 RISK LEVEL: {risk_emoji} {result['risk_level']}")
        
        # Probability breakdown
        print(f"\n📊 PROBABILITY BREAKDOWN:")
        print(f"   Safe:       {result['prob_safe']:.1%}")
        print(f"   Vulnerable: {result['prob_vulnerable']:.1%}")
        
        # Top suspicious areas
        if result['top_suspicious_nodes']:
            print(f"\n🎯 MOST SUSPICIOUS CODE AREAS:")
            for i, (node_id, attention, label) in enumerate(result['top_suspicious_nodes'][:5]):
                if attention > 0.7:
                    risk_indicator = "🔴 HIGH RISK"
                elif attention > 0.4:
                    risk_indicator = "🟡 MEDIUM RISK"
                else:
                    risk_indicator = "🟢 LOW RISK"
                
                print(f"   Rank #{i+1}: {risk_indicator} - {label} (Attention: {attention:.3f})")
        
        # Attention statistics
        stats = result['attention_stats']
        print(f"\n📈 ATTENTION ANALYSIS:")
        print(f"   Average attention: {stats['mean']:.3f}")
        print(f"   Maximum attention: {stats['max']:.3f}")
        print(f"   High attention nodes (>0.7): {stats['num_high_attention']}")
        print(f"   Medium attention nodes (0.3-0.7): {stats['num_medium_attention']}")
        print(f"   Low attention nodes (<0.3): {stats['num_low_attention']}")
        
        # Explanation
        print(f"\n💡 EXPLANATION:")
        for line in result['explanation'].split('\n'):
            if line.strip():
                print(f"   {line}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in result['recommendations']:
            print(f"   {rec}")
        
        # Graph info
        graph_info = result['graph_info']
        print(f"\n📊 GRAPH DETAILS:")
        print(f"   Nodes: {graph_info['num_nodes']}")
        print(f"   Edges: {graph_info['num_edges']}")
        
        print("="*80 + "\n")
    
    def visualize_attention(self, result: dict, save_path=None):
        """
        Create a simple text-based visualization of attention weights
        
        Args:
            result: Prediction result from predict_with_explanation()
            save_path: Optional path to save visualization
        """
        attention_weights = result['attention_weights']
        node_labels = result['graph_info']['node_labels']
        
        print("\n" + "="*60)
        print("🎨 ATTENTION VISUALIZATION")
        print("="*60)
        
        # Create attention bars
        max_attention = np.max(attention_weights)
        
        for i, attention in enumerate(attention_weights):
            # Create bar visualization
            bar_length = int((attention / max_attention) * 20) if max_attention > 0 else 0
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            # Get node label
            if node_labels and i < len(node_labels):
                label = node_labels[i][:30]  # Truncate long labels
            else:
                label = f"Node_{i}"
            
            # Color coding
            if attention > 0.7:
                color_indicator = "🔴"
            elif attention > 0.4:
                color_indicator = "🟡"
            else:
                color_indicator = "🟢"
            
            print(f"{color_indicator} {label:30} |{bar}| {attention:.3f}")
        
        print("="*60)
        print("Legend: 🔴 High Risk  🟡 Medium Risk  🟢 Low Risk")
        print("="*60 + "\n")
        
        if save_path:
            # Save to file (simple text format)
            with open(save_path, 'w') as f:
                f.write("Attention Visualization\n")
                f.write("="*50 + "\n")
                for i, attention in enumerate(attention_weights):
                    label = node_labels[i] if node_labels and i < len(node_labels) else f"Node_{i}"
                    f.write(f"{label}: {attention:.3f}\n")
            print(f"💾 Visualization saved to: {save_path}")


def demo_explainable_prediction():
    """Demo function for explainable prediction"""
    
    print("="*80)
    print("EXPLAINABLE VULNERABILITY PREDICTOR DEMO")
    print("="*80)
    
    try:
        # Initialize predictor
        predictor = ExplainablePredictor()
        
        # Create demo graph with node labels
        print("🔄 Creating demo graph with labeled nodes...")
        
        num_nodes = 12
        x = torch.randn(num_nodes, 100)
        
        # Create edges (complex pattern)
        edges = []
        for i in range(num_nodes - 1):
            edges.extend([[i, i+1], [i+1, i]])
        
        # Add vulnerability-like patterns
        edges.extend([
            [0, 5], [5, 0],    # Jump
            [2, 8], [8, 2],    # Cross connection
            [3, 9], [9, 3],    # Complex flow
            [1, 10], [10, 1],  # More complexity
        ])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Create node labels (simulating code lines)
        node_labels = [
            "int buffer[100];",
            "char* input = get_user_input();",
            "if (strlen(input) > 0) {",
            "    strcpy(buffer, input);",  # Potential vulnerability
            "    printf(buffer);",         # Format string vulnerability
            "}",
            "return process_buffer(buffer);",
            "void cleanup() {",
            "    free(buffer);",           # Potential double-free
            "    buffer = NULL;",
            "}",
            "exit(0);"
        ]
        
        graph_data = Data(x=x, edge_index=edge_index)
        
        print(f"✅ Demo graph created with {len(node_labels)} labeled nodes")
        
        # Make explainable prediction
        print(f"\n🔮 Making explainable prediction...")
        result = predictor.predict_with_explanation(
            graph_data, 
            node_labels=node_labels, 
            top_k=8
        )
        
        # Display detailed explanation
        predictor.print_detailed_explanation(result, "Demo Vulnerable Code")
        
        # Show attention visualization
        predictor.visualize_attention(result)
        
        print(f"\n✅ Explainable demo completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run explainable demo
    demo_explainable_prediction()