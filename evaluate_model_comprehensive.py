#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Provides detailed performance analysis with all metrics
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve, precision_recall_curve,
    matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader
import sys
import os

# Add src to path
sys.path.append('src')
from src.utils.objects.input_dataset import InputDataset
from src.process.devign import Devign
import configs


def comprehensive_evaluation():
    """Run comprehensive evaluation of the trained model"""
    
    print("="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    
    # Load configurations
    context = configs.Process()
    devign_config = configs.Devign()
    PATHS = configs.Paths()
    FILES = configs.Files()
    DEVICE = FILES.get_device()
    
    # Load trained model
    model_path = PATHS.model + FILES.model
    print(f"Loading model from: {model_path}")
    print(f"Full path: {os.path.abspath(model_path)}")
    
    model = Devign(
        path=model_path, 
        device=DEVICE, 
        model=devign_config.model, 
        learning_rate=devign_config.learning_rate,
        weight_decay=devign_config.weight_decay,
        loss_lambda=devign_config.loss_lambda
    )
    
    try:
        model.load()
        print("✅ Model loaded successfully")
    except:
        print("⚠️  No saved model found, using current state")
    
    # Load test data
    print(f"\n📊 Loading test dataset...")
    dataset = InputDataset('data/input')
    
    # Split data (same as training)
    from sklearn.model_selection import train_test_split
    indices = list(range(len(dataset)))
    labels = [int(dataset[i].y.item()) for i in indices]
    
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, 
        stratify=[labels[i] for i in temp_idx]
    )
    
    test_dataset = [dataset[i] for i in test_idx]
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Run evaluation
    model.model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    print(f"\n🔍 Running model evaluation...")
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            output = model.model(batch)
            
            # Get probabilities and predictions
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.long().cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    # Calculate comprehensive metrics
    print_comprehensive_metrics(all_predictions, all_probabilities, all_labels)
    
    # Create visualizations
    create_evaluation_plots(all_predictions, all_probabilities, all_labels)
    
    return all_predictions, all_probabilities, all_labels


def print_comprehensive_metrics(predictions, probabilities, labels):
    """Print all possible performance metrics"""
    
    print(f"\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    mcc = matthews_corrcoef(labels, predictions)
    kappa = cohen_kappa_score(labels, predictions)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    # Averaged metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
    
    # ROC AUC
    try:
        auc = roc_auc_score(labels, probabilities[:, 1])
    except:
        auc = None
    
    print(f"🎯 OVERALL PERFORMANCE METRICS")
    print("-" * 50)
    print(f"Accuracy:                    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Balanced Accuracy:           {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
    print(f"Matthews Correlation Coef:   {mcc:.4f}")
    print(f"Cohen's Kappa:               {kappa:.4f}")
    if auc is not None:
        print(f"ROC AUC Score:               {auc:.4f}")
    
    print(f"\n📊 PER-CLASS PERFORMANCE")
    print("-" * 50)
    class_names = ['Non-Vulnerable', 'Vulnerable']
    for i, class_name in enumerate(class_names):
        print(f"{class_name:15s}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")
    
    print(f"\n📈 AVERAGED METRICS")
    print("-" * 50)
    print(f"Macro Average:      Precision={precision_macro:.4f}, Recall={recall_macro:.4f}, F1={f1_macro:.4f}")
    print(f"Weighted Average:   Precision={precision_weighted:.4f}, Recall={recall_weighted:.4f}, F1={f1_weighted:.4f}")
    
    print(f"\n🔍 CONFUSION MATRIX ANALYSIS")
    print("-" * 50)
    print(f"Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Non-Vuln  Vulnerable")
    print(f"  Actual Non-Vuln    {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"  Actual Vulnerable  {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    print(f"\nDetailed Confusion Matrix Metrics:")
    print(f"True Positives (TP):         {tp:4d}  (Correctly identified vulnerabilities)")
    print(f"True Negatives (TN):         {tn:4d}  (Correctly identified non-vulnerabilities)")
    print(f"False Positives (FP):        {fp:4d}  (False alarms)")
    print(f"False Negatives (FN):        {fn:4d}  (Missed vulnerabilities)")
    
    print(f"\n🔬 CLINICAL/DIAGNOSTIC METRICS")
    print("-" * 50)
    print(f"Sensitivity (True Pos Rate): {sensitivity:.4f}  (% of vulnerabilities detected)")
    print(f"Specificity (True Neg Rate): {specificity:.4f}  (% of non-vulns correctly identified)")
    print(f"Positive Predictive Value:   {ppv:.4f}  (% of positive predictions that are correct)")
    print(f"Negative Predictive Value:   {npv:.4f}  (% of negative predictions that are correct)")
    
    # Error rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    
    print(f"\n⚠️  ERROR RATES")
    print("-" * 50)
    print(f"False Positive Rate:         {fpr:.4f}  (% of non-vulns incorrectly flagged)")
    print(f"False Negative Rate:         {fnr:.4f}  (% of vulnerabilities missed)")
    print(f"Overall Error Rate:          {1-accuracy:.4f}  (% of all predictions wrong)")
    
    print(f"\n💡 PERFORMANCE INTERPRETATION")
    print("-" * 50)
    
    # Overall performance
    if accuracy >= 0.85:
        print(f"🎉 EXCELLENT: Outstanding model performance!")
    elif accuracy >= 0.80:
        print(f"✅ VERY GOOD: Model exceeds baseline performance!")
    elif accuracy >= 0.75:
        print(f"👍 GOOD: Solid model performance!")
    elif accuracy >= 0.70:
        print(f"⚠️  FAIR: Acceptable but room for improvement!")
    else:
        print(f"❌ POOR: Significant improvement needed!")
    
    # Vulnerability detection analysis
    vuln_recall = recall[1] if len(recall) > 1 else 0
    vuln_precision = precision[1] if len(precision) > 1 else 0
    
    print(f"\n🛡️  VULNERABILITY DETECTION ANALYSIS")
    print("-" * 50)
    print(f"Vulnerability Detection Rate: {vuln_recall:.4f} ({vuln_recall*100:.1f}%)")
    print(f"Vulnerability Precision:      {vuln_precision:.4f} ({vuln_precision*100:.1f}%)")
    
    if vuln_recall >= 0.85:
        print(f"🎯 EXCELLENT vulnerability detection!")
    elif vuln_recall >= 0.75:
        print(f"✅ GOOD vulnerability detection!")
    elif vuln_recall >= 0.65:
        print(f"⚠️  MODERATE vulnerability detection!")
    else:
        print(f"❌ POOR vulnerability detection - missing too many!")
    
    if vuln_precision >= 0.80:
        print(f"🎯 LOW false alarm rate!")
    elif vuln_precision >= 0.60:
        print(f"⚠️  MODERATE false alarm rate!")
    else:
        print(f"❌ HIGH false alarm rate!")
    
    # Business impact analysis
    print(f"\n💼 BUSINESS IMPACT ANALYSIS")
    print("-" * 50)
    total_vulns = support[1] if len(support) > 1 else 0
    missed_vulns = fn
    false_alarms = fp
    
    print(f"Total Vulnerabilities in Test: {total_vulns}")
    print(f"Vulnerabilities Detected:      {tp} ({tp/total_vulns*100:.1f}%)")
    print(f"Vulnerabilities Missed:        {missed_vulns} ({missed_vulns/total_vulns*100:.1f}%)")
    print(f"False Alarms Generated:        {false_alarms}")
    
    if missed_vulns == 0:
        print(f"🎉 PERFECT: No vulnerabilities missed!")
    elif missed_vulns <= total_vulns * 0.1:
        print(f"✅ EXCELLENT: Very few vulnerabilities missed!")
    elif missed_vulns <= total_vulns * 0.2:
        print(f"👍 GOOD: Acceptable number of missed vulnerabilities!")
    else:
        print(f"⚠️  CONCERNING: Too many vulnerabilities being missed!")


def create_evaluation_plots(predictions, probabilities, labels):
    """Create visualization plots for model evaluation"""
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comprehensive Model Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix Heatmap
        cm = confusion_matrix(labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Vulnerable', 'Vulnerable'],
                   yticklabels=['Non-Vulnerable', 'Vulnerable'],
                   ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_ylabel('True Label')
        axes[0,0].set_xlabel('Predicted Label')
        
        # 2. ROC Curve
        try:
            fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
            auc = roc_auc_score(labels, probabilities[:, 1])
            axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, 
                          label=f'ROC curve (AUC = {auc:.3f})')
            axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[0,1].set_xlim([0.0, 1.0])
            axes[0,1].set_ylim([0.0, 1.05])
            axes[0,1].set_xlabel('False Positive Rate')
            axes[0,1].set_ylabel('True Positive Rate')
            axes[0,1].set_title('ROC Curve')
            axes[0,1].legend(loc="lower right")
        except:
            axes[0,1].text(0.5, 0.5, 'ROC Curve\nNot Available', 
                          ha='center', va='center', transform=axes[0,1].transAxes)
        
        # 3. Precision-Recall Curve
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(labels, probabilities[:, 1])
            axes[1,0].plot(recall_curve, precision_curve, color='blue', lw=2)
            axes[1,0].set_xlabel('Recall')
            axes[1,0].set_ylabel('Precision')
            axes[1,0].set_title('Precision-Recall Curve')
            axes[1,0].grid(True)
        except:
            axes[1,0].text(0.5, 0.5, 'Precision-Recall\nCurve Not Available', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
        
        # 4. Prediction Distribution
        axes[1,1].hist(probabilities[:, 1], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1,1].set_xlabel('Predicted Probability (Vulnerable)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Prediction Probability Distribution')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_evaluation_plots.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Evaluation plots saved to: model_evaluation_plots.png")
        
    except ImportError:
        print(f"\n⚠️  Matplotlib not available, skipping plots")
    except Exception as e:
        print(f"\n⚠️  Error creating plots: {e}")


if __name__ == '__main__':
    predictions, probabilities, labels = comprehensive_evaluation()
    
    print(f"\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"✅ Comprehensive evaluation finished!")
    print(f"📊 Check model_evaluation_plots.png for visualizations")
    print(f"📋 All metrics have been calculated and displayed above")