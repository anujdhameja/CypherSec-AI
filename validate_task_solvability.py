#!/usr/bin/env python3
"""
Task Solvability Investigation
Determines if vulnerability detection is actually possible with current data
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from collections import Counter, defaultdict
import pickle
from datetime import datetime

# Add src to path
sys.path.append('src')
from src.utils.objects.input_dataset import InputDataset
from src.process.balanced_training_config import BalancedDevignModel
import configs

class TaskSolvabilityInvestigator:
    def __init__(self):
        # Get correct data path from configs
        paths = configs.Paths()
        self.data_dir = str(Path(paths.input))
        self.results = {}
        
    def load_dataset_with_predictions(self):
        """Load dataset and get model predictions"""
        print("Loading dataset and generating predictions...")
        
        # Load dataset
        dataset = InputDataset(self.data_dir)
        print(f"Total graphs loaded: {len(dataset)}")
        
        # Load trained model (from previous comparison)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BalancedDevignModel(
            input_dim=100,
            hidden_dim=128,
            output_dim=2,
            num_steps=3,
            dropout=0.3
        ).to(device)
        
        try:
            model.load_state_dict(torch.load('best_gnn_model.pth', map_location=device))
            print("✓ Loaded trained model")
        except:
            print("⚠️ No trained model found, using random initialization")
        
        # Get predictions for all samples
        model.eval()
        predictions = []
        true_labels = []
        confidences = []
        
        from torch_geometric.loader import DataLoader
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch)
                probs = torch.softmax(out, dim=1)
                pred = out.argmax(dim=1)
                
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(batch.y.long().cpu().numpy())
                confidences.extend(probs.max(dim=1)[0].cpu().numpy())
        
        return dataset, predictions, true_labels, confidences
    
    def analyze_misclassifications(self, dataset, predictions, true_labels, confidences):
        """Analyze the 20 worst misclassifications"""
        print("\n=== ANALYZING MISCLASSIFICATIONS ===")
        
        misclassified = []
        for i, (pred, true, conf) in enumerate(zip(predictions, true_labels, confidences)):
            if pred != true:
                misclassified.append({
                    'index': i,
                    'predicted': pred,
                    'true_label': true,
                    'confidence': conf,
                    'data': dataset[i]
                })
        
        # Sort by confidence (most confident wrong predictions)
        misclassified.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"Total misclassifications: {len(misclassified)}")
        print(f"Accuracy: {(len(predictions) - len(misclassified)) / len(predictions):.2%}")
        
        # Analyze top 20 most confident wrong predictions
        top_misclassified = misclassified[:20]
        
        analysis = {
            'total_misclassified': len(misclassified),
            'accuracy': (len(predictions) - len(misclassified)) / len(predictions),
            'top_20_analysis': []
        }
        
        print(f"\n📊 Top 20 Most Confident Misclassifications:")
        print(f"{'Index':<8} {'True':<6} {'Pred':<6} {'Conf':<8} {'Nodes':<8} {'Edges':<8} {'Feature Stats'}")
        print("-" * 80)
        
        for item in top_misclassified:
            data = item['data']
            feat_mean = data.x.mean().item()
            feat_std = data.x.std().item()
            feat_zero_ratio = (data.x == 0).float().mean().item()
            
            print(f"{item['index']:<8} {item['true_label']:<6} {item['predicted']:<6} "
                  f"{item['confidence']:<8.3f} {data.x.shape[0]:<8} {data.edge_index.shape[1]:<8} "
                  f"μ={feat_mean:.3f}, σ={feat_std:.3f}, zero={feat_zero_ratio:.2%}")
            
            analysis['top_20_analysis'].append({
                'index': item['index'],
                'true_label': item['true_label'],
                'predicted': item['predicted'],
                'confidence': item['confidence'],
                'num_nodes': data.x.shape[0],
                'num_edges': data.edge_index.shape[1],
                'feature_mean': feat_mean,
                'feature_std': feat_std,
                'zero_feature_ratio': feat_zero_ratio
            })
        
        return analysis
    
    def inspect_feature_quality(self, dataset):
        """Deep dive into feature quality issues"""
        print("\n=== FEATURE QUALITY INSPECTION ===")
        
        zero_feature_graphs = []
        identical_feature_graphs = []
        normal_graphs = []
        
        for i, data in enumerate(dataset):
            # Check for zero features
            zero_ratio = (data.x == 0).float().mean().item()
            
            # Check for identical features (all nodes have same features)
            if data.x.shape[0] > 1:
                feature_variance = data.x.var(dim=0).mean().item()
                if feature_variance < 1e-6:  # Essentially identical
                    identical_feature_graphs.append({
                        'index': i,
                        'data': data,
                        'zero_ratio': zero_ratio,
                        'variance': feature_variance
                    })
                elif zero_ratio > 0.9:  # >90% zeros
                    zero_feature_graphs.append({
                        'index': i,
                        'data': data,
                        'zero_ratio': zero_ratio
                    })
                else:
                    normal_graphs.append({
                        'index': i,
                        'data': data,
                        'zero_ratio': zero_ratio
                    })
        
        print(f"Zero feature graphs: {len(zero_feature_graphs)}")
        print(f"Identical feature graphs: {len(identical_feature_graphs)}")
        print(f"Normal graphs: {len(normal_graphs)}")
        
        # Show examples
        feature_analysis = {
            'zero_feature_count': len(zero_feature_graphs),
            'identical_feature_count': len(identical_feature_graphs),
            'normal_count': len(normal_graphs),
            'zero_examples': [],
            'identical_examples': [],
            'normal_examples': []
        }
        
        # Zero feature examples
        print(f"\n🔍 Zero Feature Examples (showing first 5):")
        for item in zero_feature_graphs[:5]:
            data = item['data']
            print(f"Graph {item['index']}: {data.x.shape[0]} nodes, "
                  f"{item['zero_ratio']:.1%} zeros, label={data.y.item()}")
            
            feature_analysis['zero_examples'].append({
                'index': item['index'],
                'num_nodes': data.x.shape[0],
                'zero_ratio': item['zero_ratio'],
                'label': data.y.item()
            })
        
        # Identical feature examples
        print(f"\n🔍 Identical Feature Examples (showing first 5):")
        for item in identical_feature_graphs[:5]:
            data = item['data']
            print(f"Graph {item['index']}: {data.x.shape[0]} nodes, "
                  f"variance={item['variance']:.2e}, label={data.y.item()}")
            
            feature_analysis['identical_examples'].append({
                'index': item['index'],
                'num_nodes': data.x.shape[0],
                'variance': item['variance'],
                'label': data.y.item()
            })
        
        # Normal examples
        print(f"\n🔍 Normal Feature Examples (showing first 5):")
        for item in normal_graphs[:5]:
            data = item['data']
            feat_mean = data.x.mean().item()
            feat_std = data.x.std().item()
            print(f"Graph {item['index']}: {data.x.shape[0]} nodes, "
                  f"μ={feat_mean:.3f}, σ={feat_std:.3f}, label={data.y.item()}")
            
            feature_analysis['normal_examples'].append({
                'index': item['index'],
                'num_nodes': data.x.shape[0],
                'feature_mean': feat_mean,
                'feature_std': feat_std,
                'label': data.y.item()
            })
        
        return feature_analysis
    
    def analyze_label_distribution(self, dataset, predictions, true_labels):
        """Analyze label patterns and distribution"""
        print("\n=== LABEL DISTRIBUTION ANALYSIS ===")
        
        # Basic distribution
        true_dist = Counter(true_labels)
        pred_dist = Counter(predictions)
        
        print(f"True label distribution: {dict(true_dist)}")
        print(f"Predicted distribution: {dict(pred_dist)}")
        
        # Analyze by graph characteristics
        vulnerable_graphs = []
        safe_graphs = []
        
        for i, data in enumerate(dataset):
            label = true_labels[i]
            graph_info = {
                'index': i,
                'label': label,
                'num_nodes': data.x.shape[0],
                'num_edges': data.edge_index.shape[1],
                'feature_mean': data.x.mean().item(),
                'feature_std': data.x.std().item(),
                'zero_ratio': (data.x == 0).float().mean().item()
            }
            
            if label == 1:  # Vulnerable
                vulnerable_graphs.append(graph_info)
            else:  # Safe
                safe_graphs.append(graph_info)
        
        # Compare characteristics
        vuln_nodes = [g['num_nodes'] for g in vulnerable_graphs]
        safe_nodes = [g['num_nodes'] for g in safe_graphs]
        
        vuln_edges = [g['num_edges'] for g in vulnerable_graphs]
        safe_edges = [g['num_edges'] for g in safe_graphs]
        
        print(f"\n📊 Graph Characteristics by Label:")
        print(f"Vulnerable graphs: {len(vulnerable_graphs)}")
        print(f"  - Avg nodes: {np.mean(vuln_nodes):.1f} ± {np.std(vuln_nodes):.1f}")
        print(f"  - Avg edges: {np.mean(vuln_edges):.1f} ± {np.std(vuln_edges):.1f}")
        
        print(f"Safe graphs: {len(safe_graphs)}")
        print(f"  - Avg nodes: {np.mean(safe_nodes):.1f} ± {np.std(safe_nodes):.1f}")
        print(f"  - Avg edges: {np.mean(safe_edges):.1f} ± {np.std(safe_edges):.1f}")
        
        # Show examples
        print(f"\n🔍 Vulnerable Graph Examples (first 5):")
        for graph in vulnerable_graphs[:5]:
            print(f"Graph {graph['index']}: {graph['num_nodes']} nodes, "
                  f"{graph['num_edges']} edges, zero_ratio={graph['zero_ratio']:.2%}")
        
        print(f"\n🔍 Safe Graph Examples (first 5):")
        for graph in safe_graphs[:5]:
            print(f"Graph {graph['index']}: {graph['num_nodes']} nodes, "
                  f"{graph['num_edges']} edges, zero_ratio={graph['zero_ratio']:.2%}")
        
        return {
            'true_distribution': dict(true_dist),
            'predicted_distribution': dict(pred_dist),
            'vulnerable_examples': vulnerable_graphs[:5],
            'safe_examples': safe_graphs[:5],
            'vulnerable_stats': {
                'count': len(vulnerable_graphs),
                'avg_nodes': float(np.mean(vuln_nodes)),
                'std_nodes': float(np.std(vuln_nodes)),
                'avg_edges': float(np.mean(vuln_edges)),
                'std_edges': float(np.std(vuln_edges))
            },
            'safe_stats': {
                'count': len(safe_graphs),
                'avg_nodes': float(np.mean(safe_nodes)),
                'std_nodes': float(np.std(safe_nodes)),
                'avg_edges': float(np.mean(safe_edges)),
                'std_edges': float(np.std(safe_edges))
            }
        }
    
    def investigate_data_pipeline(self, dataset):
        """Investigate where the data pipeline might be failing"""
        print("\n=== DATA PIPELINE INVESTIGATION ===")
        
        # Try to trace back to original data
        try:
            # Check if we can access raw CPG data
            paths = configs.Paths()
            raw_path = Path(paths.raw) if hasattr(paths, 'raw') else None
            cpg_path = Path(paths.cpg) if hasattr(paths, 'cpg') else None
            
            print(f"Raw data path: {raw_path}")
            print(f"CPG data path: {cpg_path}")
            
            # Check input files
            input_files = list(Path(self.data_dir).glob('*.pkl'))
            print(f"Input files found: {len(input_files)}")
            
            # Sample one file to inspect
            if input_files:
                sample_file = input_files[0]
                print(f"Inspecting: {sample_file}")
                
                df = pd.read_pickle(sample_file)
                print(f"DataFrame shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                
                if 'input' in df.columns:
                    sample_graph = df['input'].iloc[0]
                    print(f"Sample graph type: {type(sample_graph)}")
                    print(f"Sample graph attributes: {dir(sample_graph)}")
                    
                    if hasattr(sample_graph, 'x'):
                        print(f"Features shape: {sample_graph.x.shape}")
                        print(f"Feature range: [{sample_graph.x.min():.3f}, {sample_graph.x.max():.3f}]")
                        print(f"Feature mean: {sample_graph.x.mean():.3f}")
                        print(f"Zero ratio: {(sample_graph.x == 0).float().mean():.2%}")
        
        except Exception as e:
            print(f"Error investigating pipeline: {e}")
        
        return {
            'input_files_count': len(input_files) if 'input_files' in locals() else 0,
            'pipeline_status': 'investigated'
        }
    
    def run_full_investigation(self):
        """Run complete task solvability investigation"""
        print("=" * 80)
        print("TASK SOLVABILITY INVESTIGATION")
        print("=" * 80)
        
        # Load data and predictions
        dataset, predictions, true_labels, confidences = self.load_dataset_with_predictions()
        
        # Run all analyses
        misclass_analysis = self.analyze_misclassifications(dataset, predictions, true_labels, confidences)
        feature_analysis = self.inspect_feature_quality(dataset)
        label_analysis = self.analyze_label_distribution(dataset, predictions, true_labels)
        pipeline_analysis = self.investigate_data_pipeline(dataset)
        
        # Compile results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(dataset),
            'misclassification_analysis': misclass_analysis,
            'feature_quality_analysis': feature_analysis,
            'label_distribution_analysis': label_analysis,
            'pipeline_investigation': pipeline_analysis
        }
        
        # Generate conclusions
        self.generate_conclusions()
        
        # Save results
        self.save_results()
    
    def generate_conclusions(self):
        """Generate actionable conclusions"""
        print("\n" + "=" * 80)
        print("CONCLUSIONS & RECOMMENDATIONS")
        print("=" * 80)
        
        feature_analysis = self.results['feature_quality_analysis']
        misclass_analysis = self.results['misclassification_analysis']
        
        conclusions = []
        
        # Feature quality assessment
        zero_ratio = feature_analysis['zero_feature_count'] / self.results['dataset_size']
        identical_ratio = feature_analysis['identical_feature_count'] / self.results['dataset_size']
        
        if zero_ratio > 0.1:  # >10% zero features
            conclusions.append(f"🚨 CRITICAL: {zero_ratio:.1%} of graphs have mostly zero features")
        
        if identical_ratio > 0.05:  # >5% identical features
            conclusions.append(f"🚨 CRITICAL: {identical_ratio:.1%} of graphs have identical node features")
        
        # Performance assessment
        accuracy = misclass_analysis['accuracy']
        if accuracy < 0.55:  # <55% accuracy
            conclusions.append(f"🚨 CRITICAL: Model accuracy ({accuracy:.1%}) barely above random")
        
        # Print conclusions
        print("\n🎯 KEY FINDINGS:")
        for i, conclusion in enumerate(conclusions, 1):
            print(f"{i}. {conclusion}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if zero_ratio > 0.1 or identical_ratio > 0.05:
            print("1. 🔧 FIX FEATURE PIPELINE: Features are corrupted/missing")
            print("   - Check Word2Vec model quality")
            print("   - Verify token extraction from CPG")
            print("   - Debug NodesEmbedding class")
        
        if accuracy < 0.55:
            print("2. 📊 TASK DIFFICULTY: Current features insufficient")
            print("   - Add domain-specific features (API calls, patterns)")
            print("   - Consider different code representations")
            print("   - Verify label quality")
        
        print("3. 🔍 IMMEDIATE ACTIONS:")
        print("   - Run investigate_zero_features.py to trace corruption")
        print("   - Manually inspect 10 vulnerable vs safe examples")
        print("   - Check inter-annotator agreement on labels")
        
        self.results['conclusions'] = conclusions
    
    def save_results(self):
        """Save investigation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"task_solvability_report.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Full investigation report saved to: {filename}")

def main():
    """Main execution"""
    investigator = TaskSolvabilityInvestigator()
    investigator.run_full_investigation()

if __name__ == "__main__":
    main()