#!/usr/bin/env python3
"""
Stage 5: GNN Prediction
Predict vulnerability using the trained final_model.pth
"""

import torch
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.process.balanced_training_config import BalancedDevignModel
from torch_geometric.data import Data
from gensim.models import Word2Vec
import numpy as np


class VulnerabilityPredictor:
    """
    Predict vulnerability using the trained final_model.pth
    
    Usage:
        predictor = VulnerabilityPredictor('models/final_model.pth')
        result = predictor.predict(graph_data)
        # Returns: {'is_vulnerable': 1, 'confidence': 0.92, ...}
    """
    
    def __init__(self, model_path='models/final_model.pth', w2v_path='data/w2v/w2v.model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.w2v_path = w2v_path
        
        print("="*60)
        print("VULNERABILITY PREDICTOR INITIALIZATION")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Model path: {os.path.abspath(model_path)}")
        print(f"Word2Vec path: {os.path.abspath(w2v_path)}")
        
        # Load model with exact same architecture as training
        print(f"\n🔧 Loading model...")
        self.model = BalancedDevignModel(
            input_dim=100,      # Same as training
            output_dim=2,       # Binary classification
            hidden_dim=256,     # Same as training
            num_steps=5,        # Same as training
            dropout=0.2         # Same as training
        )
        
        # Load trained weights
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✅ Model weights loaded successfully")
        else:
            print(f"❌ Model file not found: {model_path}")
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
        
        print(f"\n🎯 Predictor ready for inference!")
        print("="*60 + "\n")
    
    def predict(self, graph_data: Data) -> dict:
        """
        Predict vulnerability for a single graph
        
        Args:
            graph_data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, feature_dim]
                - edge_index: Edge connectivity [2, num_edges]
        
        Returns:
            dict: {
                'is_vulnerable': 0 or 1,
                'confidence': float (0-1),
                'prob_safe': float (0-1),
                'prob_vulnerable': float (0-1),
                'prediction_details': dict
            }
        """
        # Ensure graph is on correct device
        graph_data = graph_data.to(self.device)
        
        # Add batch dimension if not present
        if not hasattr(graph_data, 'batch'):
            graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            # Forward pass
            output = self.model(graph_data)  # Shape: [1, 2]
            
            # Convert to probabilities
            probs = torch.softmax(output, dim=1)  # Shape: [1, 2]
            
            # Get prediction
            pred = output.argmax(dim=1).item()  # 0 or 1
            confidence = probs[0][pred].item()  # Confidence in prediction
            
            prob_safe = probs[0][0].item()
            prob_vulnerable = probs[0][1].item()
        
        # Create detailed result
        result = {
            'is_vulnerable': pred,
            'confidence': confidence,
            'prob_safe': prob_safe,
            'prob_vulnerable': prob_vulnerable,
            'prediction_details': {
                'raw_logits': output[0].cpu().numpy().tolist(),
                'probabilities': probs[0].cpu().numpy().tolist(),
                'predicted_class': 'Vulnerable' if pred == 1 else 'Safe',
                'confidence_level': self._get_confidence_level(confidence),
                'num_nodes': graph_data.x.size(0),
                'num_edges': graph_data.edge_index.size(1) if graph_data.edge_index.numel() > 0 else 0
            }
        }
        
        return result
    
    def predict_batch(self, graph_list: list) -> list:
        """
        Predict vulnerabilities for multiple graphs
        
        Args:
            graph_list: List of PyTorch Geometric Data objects
        
        Returns:
            list: List of prediction dictionaries
        """
        results = []
        
        print(f"🔄 Processing {len(graph_list)} graphs...")
        
        for i, graph_data in enumerate(graph_list):
            try:
                result = self.predict(graph_data)
                result['graph_id'] = i
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(graph_list)} graphs")
                    
            except Exception as e:
                print(f"❌ Error processing graph {i}: {e}")
                results.append({
                    'graph_id': i,
                    'error': str(e),
                    'is_vulnerable': None,
                    'confidence': 0.0
                })
        
        print(f"✅ Completed batch prediction")
        return results
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to human-readable level"""
        if confidence >= 0.95:
            return "Very High"
        elif confidence >= 0.85:
            return "High"
        elif confidence >= 0.70:
            return "Medium"
        elif confidence >= 0.55:
            return "Low"
        else:
            return "Very Low"
    
    def print_prediction(self, result: dict, graph_info: str = ""):
        """Pretty print prediction result"""
        print("\n" + "="*50)
        print("VULNERABILITY PREDICTION RESULT")
        print("="*50)
        
        if graph_info:
            print(f"Graph: {graph_info}")
        
        is_vuln = result['is_vulnerable']
        confidence = result['confidence']
        
        status = "🚨 VULNERABLE" if is_vuln else "✅ SAFE"
        print(f"\nPrediction: {status}")
        print(f"Confidence: {confidence:.2%} ({result['prediction_details']['confidence_level']})")
        
        print(f"\nProbability Breakdown:")
        print(f"  Safe:       {result['prob_safe']:.2%}")
        print(f"  Vulnerable: {result['prob_vulnerable']:.2%}")
        
        details = result['prediction_details']
        print(f"\nGraph Details:")
        print(f"  Nodes: {details['num_nodes']}")
        print(f"  Edges: {details['num_edges']}")
        
        print("="*50 + "\n")


def demo_prediction():
    """Demo function showing how to use the predictor"""
    
    print("="*80)
    print("VULNERABILITY PREDICTOR DEMO")
    print("="*80)
    
    try:
        # Initialize predictor
        predictor = VulnerabilityPredictor()
        
        # Create dummy graph data for demo
        print("🔄 Creating demo graph data...")
        
        # Simulate a small code graph
        num_nodes = 10
        feature_dim = 100
        
        # Random node features (in practice, these would be Word2Vec embeddings)
        x = torch.randn(num_nodes, feature_dim)
        
        # Simple edge connectivity (linear chain + some connections)
        edge_list = []
        for i in range(num_nodes - 1):
            edge_list.append([i, i + 1])  # Sequential connections
            edge_list.append([i + 1, i])  # Bidirectional
        
        # Add some additional connections
        edge_list.extend([[0, 5], [5, 0], [2, 7], [7, 2]])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Create PyG Data object
        graph_data = Data(x=x, edge_index=edge_index)
        
        print(f"✅ Demo graph created:")
        print(f"   Nodes: {graph_data.x.shape[0]}")
        print(f"   Edges: {graph_data.edge_index.shape[1]}")
        print(f"   Features: {graph_data.x.shape[1]}")
        
        # Make prediction
        print(f"\n🔮 Making prediction...")
        result = predictor.predict(graph_data)
        
        # Display result
        predictor.print_prediction(result, "Demo Graph")
        
        # Show raw result
        print("Raw prediction result:")
        for key, value in result.items():
            if key != 'prediction_details':
                print(f"  {key}: {value}")
        
        print(f"\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run demo
    demo_prediction()