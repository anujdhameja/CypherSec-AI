#!/usr/bin/env python3
"""
Task Solvability Validation
Investigate if vulnerability detection is actually possible with current data
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
import json
import pickle
from datetime import datetime

# Add src to path
sys.path.append('src')
from src.utils.objects.input_dataset import InputDataset
from src.process.balanced_training_config import BalancedDevignModel
import configs

class TaskValidator:
    def __init__(self):
        # Get data path
        paths = configs.Paths()
        self.data_dir = str(Path(paths.input))
        self.results = {}
        
    def load_dataset_with_metadata(self):
        """Load dataset and extract metadata for analysis"""
        print("="*80)
        print("LOADING DATASET WITH METADATA")
        print("="*80)
        
        dataset = InputDataset(self.data_dir, max_files=5)  # Limit for faster analysis
        print(f"Loaded {len(dataset)} samples for analysis")
        
        # Extract comprehensive metadata
        metadata = []
        for i, data in enumerate(dataset):
            meta = {
                'index': i,
                'label': int(data.y.item()),
                'num_nodes': data.x.shape[0],
                'num_edges': data.edge_index.shape[1],
                'node_features': data.x,
                'edge_index': data.edge_index,
                'avg_node_feature': data.x.mean().item(),
                'max_node_feature': data.x.max().item(),
                'min_node_feature': data.x.min().item(),
                'feature_std': data.x.std().item(),
            }
            
            # Graph structure metrics
            if data.edge_index.shape[1] > 0:
                degrees = torch.bincount(data.edge_index[0])
                meta['avg_degree'] = degrees.float().mean().item()
                meta['max_degree'] = degrees.max().item()
                meta['degree_std'] = degrees.float().std().item()
            else:
                meta['avg_degree'] = 0
                meta['max_degree'] = 0
                meta['degree_std'] = 0
            
            metadata.append(meta)
        
        return dataset, metadata
    
    def analyze_label_distribution(self, metadata):
        """Analyze label distribution and basic statistics"""
        print("\n" + "="*80)
        print("LABEL DISTRIBUTION ANALYSIS")
        print("="*80)
        
        labels = [m['label'] for m in metadata]
        label_counts = Counter(labels)
        
        print(f"Total samples: {len(labels)}")
        print(f"Label distribution: {dict(label_counts)}")
        print(f"Class balance: {label_counts[0]/len(labels):.1%} vs {label_counts[1]/len(labels):.1%}")
        
        # Analyze by label
        vulnerable = [m for m in metadata if m['label'] == 1]
        safe = [m for m in metadata if m['label'] == 0]
        
        print(f"\n📊 STRUCTURAL DIFFERENCES:")
        print(f"{'Metric':<20} {'Vulnerable':<15} {'Safe':<15} {'Difference':<15}")
        print("-" * 65)
        
        metrics = ['num_nodes', 'num_edges', 'avg_degree', 'max_degree', 'avg_node_feature', 'feature_std']
        
        for metric in metrics:
            vuln_avg = np.mean([m[metric] for m in vulnerable])
            safe_avg = np.mean([m[metric] for m in safe])
            diff = abs(vuln_avg - safe_avg)
            
            print(f"{metric:<20} {vuln_avg:<15.2f} {safe_avg:<15.2f} {diff:<15.2f}")
        
        return vulnerable, safe
    
    def get_model_predictions(self, dataset, metadata):
        """Get model predictions for analysis"""
        print("\n" + "="*80)
        print("GETTING MODEL PREDICTIONS")
        print("="*80)
        
        # Load or train a simple model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get feature dimension
        input_dim = dataset[0].x.shape[1]
        
        model = BalancedDevignModel(
            input_dim=input_dim,
            hidden_dim=64,  # Smaller for faster training
            output_dim=2,
            num_steps=2,
            dropout=0.3
        ).to(device)
        
        # Quick training on subset
        from torch_geometric.loader import DataLoader
        
        train_indices = list(range(int(0.8 * len(dataset))))
        val_indices = list(range(int(0.8 * len(dataset)), len(dataset)))
        
        train_data = [dataset[i] for i in train_indices]
        val_data = [dataset[i] for i in val_indices]
        
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        print("Quick training for 20 epochs...")
        model.train()
        for epoch in range(20):
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch.y.long())
                loss.backward()
                optimizer.step()
        
        # Get predictions on validation set
        model.eval()
        predictions = []
        true_labels = []
        confidences = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                probs = torch.softmax(out, dim=1)
                pred = out.argmax(dim=1)
                
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(batch.y.cpu().numpy())
                confidences.extend(probs.max(dim=1)[0].cpu().numpy())
        
        return predictions, true_labels, confidences
    
    def analyze_misclassifications(self, dataset, metadata, predictions, true_labels, confidences):
        """Analyze misclassified samples"""
        print("\n" + "="*80)
        print("MISCLASSIFICATION ANALYSIS")
        print("="*80)
        
        val_start_idx = int(0.8 * len(dataset))
        val_metadata = metadata[val_start_idx:]
        
        # Find misclassifications
        misclassified = []
        correctly_classified = []
        
        for i, (pred, true, conf, meta) in enumerate(zip(predictions, true_labels, confidences, val_metadata)):
            sample_info = {
                'val_index': i,
                'global_index': meta['index'],
                'predicted': int(pred),
                'true': int(true),
                'confidence': float(conf),
                'metadata': meta
            }
            
            if pred != true:
                misclassified.append(sample_info)
            else:
                correctly_classified.append(sample_info)
        
        print(f"Validation samples: {len(predictions)}")
        print(f"Correctly classified: {len(correctly_classified)} ({len(correctly_classified)/len(predictions):.1%})")
        print(f"Misclassified: {len(misclassified)} ({len(misclassified)/len(predictions):.1%})")
        
        # Analyze misclassification patterns
        print(f"\n🔍 MISCLASSIFICATION PATTERNS:")
        
        # False positives (predicted vulnerable, actually safe)
        false_positives = [m for m in misclassified if m['predicted'] == 1 and m['true'] == 0]
        # False negatives (predicted safe, actually vulnerable)
        false_negatives = [m for m in misclassified if m['predicted'] == 0 and m['true'] == 1]
        
        print(f"False Positives: {len(false_positives)} (model thinks safe code is vulnerable)")
        print(f"False Negatives: {len(false_negatives)} (model misses actual vulnerabilities)")
        
        # Show examples
        print(f"\n📋 TOP 10 MISCLASSIFIED SAMPLES (by confidence):")
        print(f"{'Type':<15} {'Confidence':<12} {'Nodes':<8} {'Edges':<8} {'Avg Degree':<12}")
        print("-" * 65)
        
        sorted_misclassified = sorted(misclassified, key=lambda x: x['confidence'], reverse=True)[:10]
        
        for sample in sorted_misclassified:
            sample_type = "False Pos" if sample['predicted'] == 1 else "False Neg"
            meta = sample['metadata']
            print(f"{sample_type:<15} {sample['confidence']:<12.3f} {meta['num_nodes']:<8} "
                  f"{meta['num_edges']:<8} {meta['avg_degree']:<12.2f}")
        
        return misclassified, correctly_classified
    
    def inspect_sample_graphs(self, dataset, metadata, misclassified):
        """Inspect actual graph structures of misclassified samples"""
        print("\n" + "="*80)
        print("DETAILED GRAPH INSPECTION")
        print("="*80)
        
        # Take top 5 misclassified samples
        top_misclassified = sorted(misclassified, key=lambda x: x['confidence'], reverse=True)[:5]
        
        print("🔬 DETAILED ANALYSIS OF TOP 5 MISCLASSIFIED SAMPLES:")
        
        for i, sample in enumerate(top_misclassified, 1):
            print(f"\n--- SAMPLE {i} ---")
            print(f"Global Index: {sample['global_index']}")
            print(f"Predicted: {sample['predicted']} ({'Vulnerable' if sample['predicted'] == 1 else 'Safe'})")
            print(f"True Label: {sample['true']} ({'Vulnerable' if sample['true'] == 1 else 'Safe'})")
            print(f"Confidence: {sample['confidence']:.3f}")
            
            # Get the actual graph data
            graph_data = dataset[sample['global_index']]
            
            print(f"Graph Structure:")
            print(f"  - Nodes: {graph_data.x.shape[0]}")
            print(f"  - Edges: {graph_data.edge_index.shape[1]}")
            print(f"  - Feature dim: {graph_data.x.shape[1]}")
            
            # Analyze node features
            node_features = graph_data.x
            print(f"Node Features Analysis:")
            print(f"  - Mean: {node_features.mean():.3f}")
            print(f"  - Std: {node_features.std():.3f}")
            print(f"  - Min: {node_features.min():.3f}")
            print(f"  - Max: {node_features.max():.3f}")
            
            # Check for patterns in features
            # Look for zero features (might indicate missing data)
            zero_features = (node_features == 0).sum().item()
            total_features = node_features.numel()
            print(f"  - Zero features: {zero_features}/{total_features} ({zero_features/total_features:.1%})")
            
            # Look for identical nodes (might indicate poor feature extraction)
            unique_nodes = torch.unique(node_features, dim=0).shape[0]
            print(f"  - Unique nodes: {unique_nodes}/{node_features.shape[0]} ({unique_nodes/node_features.shape[0]:.1%})")
            
            # Edge analysis
            if graph_data.edge_index.shape[1] > 0:
                edge_index = graph_data.edge_index
                degrees = torch.bincount(edge_index[0], minlength=node_features.shape[0])
                print(f"Degree Distribution:")
                print(f"  - Min degree: {degrees.min().item()}")
                print(f"  - Max degree: {degrees.max().item()}")
                print(f"  - Avg degree: {degrees.float().mean().item():.2f}")
                print(f"  - Isolated nodes: {(degrees == 0).sum().item()}")
    
    def compare_vulnerable_vs_safe_samples(self, dataset, vulnerable_meta, safe_meta):
        """Compare vulnerable vs safe samples in detail"""
        print("\n" + "="*80)
        print("VULNERABLE vs SAFE COMPARISON")
        print("="*80)
        
        print("🔍 DETAILED COMPARISON OF SAMPLE GRAPHS:")
        
        # Take 3 vulnerable and 3 safe samples
        vuln_samples = vulnerable_meta[:3]
        safe_samples = safe_meta[:3]
        
        print(f"\n--- VULNERABLE SAMPLES ---")
        for i, meta in enumerate(vuln_samples, 1):
            print(f"\nVulnerable Sample {i} (Index: {meta['index']}):")
            graph_data = dataset[meta['index']]
            
            print(f"  Structure: {meta['num_nodes']} nodes, {meta['num_edges']} edges")
            print(f"  Avg degree: {meta['avg_degree']:.2f}")
            print(f"  Feature stats: mean={meta['avg_node_feature']:.3f}, std={meta['feature_std']:.3f}")
            
            # Check feature patterns
            node_features = graph_data.x
            feature_ranges = []
            for dim in range(min(5, node_features.shape[1])):  # Check first 5 dimensions
                dim_values = node_features[:, dim]
                feature_ranges.append(f"dim{dim}: [{dim_values.min():.2f}, {dim_values.max():.2f}]")
            print(f"  Feature ranges: {', '.join(feature_ranges)}")
        
        print(f"\n--- SAFE SAMPLES ---")
        for i, meta in enumerate(safe_samples, 1):
            print(f"\nSafe Sample {i} (Index: {meta['index']}):")
            graph_data = dataset[meta['index']]
            
            print(f"  Structure: {meta['num_nodes']} nodes, {meta['num_edges']} edges")
            print(f"  Avg degree: {meta['avg_degree']:.2f}")
            print(f"  Feature stats: mean={meta['avg_node_feature']:.3f}, std={meta['feature_std']:.3f}")
            
            # Check feature patterns
            node_features = graph_data.x
            feature_ranges = []
            for dim in range(min(5, node_features.shape[1])):  # Check first 5 dimensions
                dim_values = node_features[:, dim]
                feature_ranges.append(f"dim{dim}: [{dim_values.min():.2f}, {dim_values.max():.2f}]")
            print(f"  Feature ranges: {', '.join(feature_ranges)}")
    
    def check_data_quality_issues(self, dataset, metadata):
        """Check for common data quality issues"""
        print("\n" + "="*80)
        print("DATA QUALITY ASSESSMENT")
        print("="*80)
        
        issues = []
        
        # Check for duplicate graphs
        print("🔍 Checking for duplicate graphs...")
        feature_hashes = []
        for i, data in enumerate(dataset):
            # Create a simple hash of the graph structure
            node_hash = hash(tuple(data.x.flatten().numpy().round(4)))
            edge_hash = hash(tuple(data.edge_index.flatten().numpy()))
            graph_hash = hash((node_hash, edge_hash, data.y.item()))
            feature_hashes.append(graph_hash)
        
        hash_counts = Counter(feature_hashes)
        duplicates = sum(1 for count in hash_counts.values() if count > 1)
        if duplicates > 0:
            issues.append(f"Found {duplicates} potential duplicate graphs")
            print(f"⚠️  Found {duplicates} potential duplicate graphs")
        else:
            print("✅ No obvious duplicates found")
        
        # Check for degenerate graphs
        print("\n🔍 Checking for degenerate graphs...")
        degenerate_count = 0
        for meta in metadata:
            if meta['num_nodes'] <= 1 or meta['num_edges'] == 0:
                degenerate_count += 1
        
        if degenerate_count > 0:
            issues.append(f"Found {degenerate_count} degenerate graphs (≤1 node or 0 edges)")
            print(f"⚠️  Found {degenerate_count} degenerate graphs")
        else:
            print("✅ No degenerate graphs found")
        
        # Check feature quality
        print("\n🔍 Checking feature quality...")
        all_zero_features = 0
        identical_feature_graphs = 0
        
        for i, data in enumerate(dataset):
            # Check for all-zero features
            if (data.x == 0).all():
                all_zero_features += 1
            
            # Check for identical features across all nodes
            if data.x.shape[0] > 1:
                unique_features = torch.unique(data.x, dim=0).shape[0]
                if unique_features == 1:
                    identical_feature_graphs += 1
        
        if all_zero_features > 0:
            issues.append(f"Found {all_zero_features} graphs with all-zero features")
            print(f"⚠️  Found {all_zero_features} graphs with all-zero features")
        
        if identical_feature_graphs > 0:
            issues.append(f"Found {identical_feature_graphs} graphs where all nodes have identical features")
            print(f"⚠️  Found {identical_feature_graphs} graphs with identical node features")
        
        if not issues:
            print("✅ No major data quality issues detected")
        
        return issues
    
    def generate_solvability_report(self, all_results):
        """Generate final solvability assessment"""
        print("\n" + "="*80)
        print("TASK SOLVABILITY ASSESSMENT")
        print("="*80)
        
        # Collect all findings
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(all_results.get('metadata', [])),
            'findings': all_results,
            'assessment': {}
        }
        
        # Make assessment
        assessment = report['assessment']
        
        # Check if task appears solvable
        if 'data_quality_issues' in all_results and all_results['data_quality_issues']:
            assessment['data_quality'] = 'POOR - Multiple issues detected'
            assessment['solvability'] = 'LOW'
        else:
            assessment['data_quality'] = 'ACCEPTABLE'
        
        # Check feature discriminability
        vulnerable_meta = all_results.get('vulnerable_samples', [])
        safe_meta = all_results.get('safe_samples', [])
        
        if vulnerable_meta and safe_meta:
            # Compare average metrics
            vuln_nodes = np.mean([m['num_nodes'] for m in vulnerable_meta])
            safe_nodes = np.mean([m['num_nodes'] for m in safe_meta])
            node_diff = abs(vuln_nodes - safe_nodes) / max(vuln_nodes, safe_nodes)
            
            vuln_features = np.mean([m['avg_node_feature'] for m in vulnerable_meta])
            safe_features = np.mean([m['avg_node_feature'] for m in safe_meta])
            feature_diff = abs(vuln_features - safe_features) / max(abs(vuln_features), abs(safe_features))
            
            if node_diff > 0.1 or feature_diff > 0.1:
                assessment['feature_discriminability'] = 'MODERATE'
            else:
                assessment['feature_discriminability'] = 'LOW'
        
        # Overall assessment
        if assessment.get('data_quality') == 'POOR':
            assessment['overall'] = 'NOT SOLVABLE - Fix data quality first'
        elif assessment.get('feature_discriminability') == 'LOW':
            assessment['overall'] = 'DIFFICULT - Need better features'
        else:
            assessment['overall'] = 'POTENTIALLY SOLVABLE - Investigate further'
        
        print(f"📋 FINAL ASSESSMENT:")
        print(f"Data Quality: {assessment.get('data_quality', 'UNKNOWN')}")
        print(f"Feature Discriminability: {assessment.get('feature_discriminability', 'UNKNOWN')}")
        print(f"Overall Solvability: {assessment.get('overall', 'UNKNOWN')}")
        
        # Save report
        with open('task_solvability_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✅ Full report saved to: task_solvability_report.json")
        
        return assessment
    
    def run_full_validation(self):
        """Run complete task validation"""
        print("🔬 STARTING COMPREHENSIVE TASK SOLVABILITY VALIDATION")
        print("="*80)
        
        all_results = {}
        
        # 1. Load data
        dataset, metadata = self.load_dataset_with_metadata()
        all_results['metadata'] = metadata
        
        # 2. Analyze labels
        vulnerable_samples, safe_samples = self.analyze_label_distribution(metadata)
        all_results['vulnerable_samples'] = vulnerable_samples
        all_results['safe_samples'] = safe_samples
        
        # 3. Get model predictions
        predictions, true_labels, confidences = self.get_model_predictions(dataset, metadata)
        all_results['predictions'] = {
            'predictions': predictions,
            'true_labels': true_labels,
            'confidences': confidences
        }
        
        # 4. Analyze misclassifications
        misclassified, correctly_classified = self.analyze_misclassifications(
            dataset, metadata, predictions, true_labels, confidences
        )
        all_results['misclassifications'] = {
            'misclassified': misclassified,
            'correctly_classified': correctly_classified
        }
        
        # 5. Inspect sample graphs
        self.inspect_sample_graphs(dataset, metadata, misclassified)
        
        # 6. Compare vulnerable vs safe
        self.compare_vulnerable_vs_safe_samples(dataset, vulnerable_samples, safe_samples)
        
        # 7. Check data quality
        data_issues = self.check_data_quality_issues(dataset, metadata)
        all_results['data_quality_issues'] = data_issues
        
        # 8. Generate final assessment
        assessment = self.generate_solvability_report(all_results)
        
        return assessment

def main():
    """Main execution"""
    validator = TaskValidator()
    assessment = validator.run_full_validation()
    
    print(f"\n🎯 SUMMARY:")
    print(f"The vulnerability detection task appears to be: {assessment.get('overall', 'UNKNOWN')}")

if __name__ == "__main__":
    main()