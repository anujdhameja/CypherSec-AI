from ..utils.objects.metrics import Metrics
import torch
import time
from ..utils import log as logger


class Train(object):
    def __init__(self, step, epochs, verbose=True):
        self.epochs = epochs
        self.step = step
        self.history = History()
        self.verbose = verbose

    def __call__(self, train_loader_step, val_loader_step=None, early_stopping=None):
        for epoch in range(self.epochs):
            self.step.train()
            train_stats = train_loader_step(self.step)
            self.history(train_stats, epoch + 1)

            if val_loader_step is not None:
                with torch.no_grad():
                    self.step.eval()
                    val_stats = val_loader_step(self.step)
                    self.history(val_stats, epoch + 1)

                print(self.history)
                
                # Show detailed metrics every 20 epochs
                if (epoch + 1) % 20 == 0:
                    print_epoch_metrics(val_stats, epoch + 1)
                
                # Step the learning rate scheduler if available
                if hasattr(self.step, 'scheduler'):
                    self.step.scheduler.step(val_stats.loss())

                if early_stopping is not None:
                    valid_loss = val_stats.loss()
                    # early_stopping needs the validation loss to check if it has decreased,
                    # and if it has, it will make a checkpoint of the current model
                    if early_stopping(valid_loss):
                        self.history.log()
                        return
            else:
                print(self.history)
        self.history.log()


def predict(step, test_loader_step):
    """Enhanced prediction with comprehensive metrics"""
    print(f"\n" + "="*80)
    print("COMPREHENSIVE TEST EVALUATION")
    print("="*80)
    
    with torch.no_grad():
        step.eval()
        stats = test_loader_step(step)
        
        # Get predictions and labels
        predictions = stats.outs()
        labels = stats.labels()
        
        # Convert predictions to proper format for Metrics class
        if isinstance(predictions, list) and len(predictions) > 0:
            if isinstance(predictions[0], list):
                # If predictions are lists of probabilities, take the positive class probability
                predictions = [pred[1] if len(pred) > 1 else pred[0] for pred in predictions]
            elif hasattr(predictions[0], '__len__') and len(predictions[0]) > 1:
                # If predictions are arrays/tensors with multiple values
                predictions = [float(pred[1]) if len(pred) > 1 else float(pred[0]) for pred in predictions]
        
        # Create comprehensive metrics
        metrics = Metrics(predictions, labels)
        print(metrics)
        metrics.log()
        
        # Additional detailed metrics (temporarily disabled due to data conversion issue)
        # print_detailed_metrics(predictions, labels)
        print(f"\n🎉 FINAL RESULT: Your model achieved {metrics()['Accuracy']:.4f} ({metrics()['Accuracy']*100:.2f}%) test accuracy!")
        
    return metrics()["Accuracy"]


def print_detailed_metrics(predictions, labels):
    """Print comprehensive performance metrics"""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, 
        confusion_matrix, classification_report, roc_auc_score,
        matthews_corrcoef, balanced_accuracy_score
    )
    import torch
    
    # Convert to numpy if needed
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    elif isinstance(predictions, list):
        predictions = np.array(predictions)
    
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)
    
    # Convert predictions to class labels if they're probabilities
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        pred_classes = np.argmax(predictions, axis=1)
        pred_probs = predictions[:, 1]  # Probability of positive class
    else:
        pred_classes = predictions
        pred_probs = None
    
    # Final check: ensure both are integers
    pred_classes = pred_classes.astype(int)
    labels = labels.astype(int)
    
    # Basic metrics
    accuracy = accuracy_score(labels, pred_classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, pred_classes, average=None, zero_division=0
    )
    
    # Weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, pred_classes, average='weighted', zero_division=0
    )
    
    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, pred_classes, average='macro', zero_division=0
    )
    
    # Additional metrics
    balanced_acc = balanced_accuracy_score(labels, pred_classes)
    mcc = matthews_corrcoef(labels, pred_classes)
    
    # Confusion matrix
    cm = confusion_matrix(labels, pred_classes)
    
    print(f"\n📊 DETAILED PERFORMANCE METRICS")
    print("-" * 50)
    
    print(f"\n🎯 Overall Performance:")
    print(f"  Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
    print(f"  Matthews Corr Coef: {mcc:.4f}")
    
    print(f"\n📈 Per-Class Metrics:")
    class_names = ['Non-Vulnerable', 'Vulnerable']
    for i, class_name in enumerate(class_names):
        print(f"  {class_name:15s}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")
    
    print(f"\n📊 Averaged Metrics:")
    print(f"  Weighted Avg: Precision={precision_weighted:.4f}, Recall={recall_weighted:.4f}, F1={f1_weighted:.4f}")
    print(f"  Macro Avg:    Precision={precision_macro:.4f}, Recall={recall_macro:.4f}, F1={f1_macro:.4f}")
    
    print(f"\n🔍 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Non-Vuln  Vulnerable")
    print(f"  Actual Non-Vuln    {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"  Actual Vulnerable  {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    # Calculate additional confusion matrix metrics
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle non-2x2 confusion matrix
        print(f"⚠️  Unexpected confusion matrix shape: {cm.shape}")
        print(f"Confusion matrix:\n{cm}")
        # Try to extract values assuming binary classification
        if cm.shape[0] >= 2 and cm.shape[1] >= 2:
            tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        else:
            tn = fp = fn = tp = 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n🔬 Confusion Matrix Analysis:")
    print(f"  True Positives (TP):  {tp:4d}  (Correctly identified vulnerabilities)")
    print(f"  True Negatives (TN):  {tn:4d}  (Correctly identified non-vulnerabilities)")
    print(f"  False Positives (FP): {fp:4d}  (False alarms)")
    print(f"  False Negatives (FN): {fn:4d}  (Missed vulnerabilities)")
    print(f"  Sensitivity (Recall): {sensitivity:.4f}  (% of vulnerabilities caught)")
    print(f"  Specificity:          {specificity:.4f}  (% of non-vulns correctly identified)")
    
    # ROC AUC if probabilities available
    if pred_probs is not None:
        try:
            auc = roc_auc_score(labels, pred_probs)
            print(f"  ROC AUC Score:        {auc:.4f}")
        except:
            print(f"  ROC AUC Score:        N/A (single class in predictions)")
    
    # Classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(labels, pred_classes, target_names=class_names, digits=4))
    
    # Performance interpretation
    print(f"\n💡 Performance Interpretation:")
    if accuracy >= 0.85:
        print(f"  🎉 EXCELLENT: Model performance is outstanding!")
    elif accuracy >= 0.80:
        print(f"  ✅ VERY GOOD: Model performance exceeds baseline!")
    elif accuracy >= 0.75:
        print(f"  👍 GOOD: Model performance is solid!")
    elif accuracy >= 0.70:
        print(f"  ⚠️  FAIR: Model performance is acceptable but could improve!")
    else:
        print(f"  ❌ POOR: Model performance needs significant improvement!")
    
    # Vulnerability detection specific insights
    vuln_recall = recall[1] if len(recall) > 1 else 0
    vuln_precision = precision[1] if len(precision) > 1 else 0
    
    print(f"\n🛡️  Vulnerability Detection Analysis:")
    if vuln_recall >= 0.85:
        print(f"  🎯 EXCELLENT vulnerability detection: Catching {vuln_recall*100:.1f}% of vulnerabilities!")
    elif vuln_recall >= 0.75:
        print(f"  ✅ GOOD vulnerability detection: Catching {vuln_recall*100:.1f}% of vulnerabilities!")
    elif vuln_recall >= 0.65:
        print(f"  ⚠️  MODERATE vulnerability detection: Only catching {vuln_recall*100:.1f}% of vulnerabilities!")
    else:
        print(f"  ❌ POOR vulnerability detection: Missing {(1-vuln_recall)*100:.1f}% of vulnerabilities!")
    
    if vuln_precision >= 0.80:
        print(f"  🎯 LOW false alarms: {vuln_precision*100:.1f}% of vulnerability alerts are real!")
    elif vuln_precision >= 0.60:
        print(f"  ⚠️  MODERATE false alarms: {vuln_precision*100:.1f}% of vulnerability alerts are real!")
    else:
        print(f"  ❌ HIGH false alarms: Only {vuln_precision*100:.1f}% of vulnerability alerts are real!")
    
    print(f"\n" + "="*80)


class History:
    def __init__(self):
        self.history = {}
        self.epoch = 0
        self.timer = time.time()

    def __call__(self, stats, epoch):
        self.epoch = epoch

        if epoch in self.history:
            self.history[epoch].append(stats)
        else:
            self.history[epoch] = [stats]

    def __str__(self):
        epoch = f"\nEpoch {self.epoch};"
        stats = ' - '.join([f"{res}" for res in self.current()])
        timer = f"Time: {(time.time() - self.timer)}"

        return f"{epoch} - {stats} - {timer}"

    def current(self):
        return self.history[self.epoch]

    def log(self):
        msg = f"(Epoch: {self.epoch}) {' - '.join([f'({res})' for res in self.current()])}"
        logger.log_info("history", msg)


def print_epoch_metrics(val_stats, epoch):
    """Print detailed metrics during training every 20 epochs"""
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    import torch
    
    predictions = val_stats.outs()
    labels = val_stats.labels()
    
    # Convert to numpy if needed
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    elif isinstance(predictions, list):
        predictions = np.array(predictions)
    
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)
    
    # Convert predictions to class labels
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        # Multi-class probabilities - take argmax
        pred_classes = np.argmax(predictions, axis=1)
    else:
        # Raw logits or single values - convert to binary classes
        pred_classes = (predictions > 0).astype(int)  # Positive logits = class 1
    
    # Ensure both are clean integer arrays
    pred_classes = np.array(pred_classes, dtype=int)
    labels = np.array(labels, dtype=int)
    
    # Remove any potential NaN or inf values
    valid_mask = np.isfinite(pred_classes) & np.isfinite(labels)
    pred_classes = pred_classes[valid_mask]
    labels = labels[valid_mask].astype(int)  # Ensure integer type
    
    # Ensure labels are integers
    labels = labels.astype(int)
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, pred_classes, average=None, zero_division=0
    )
    
    cm = confusion_matrix(labels, pred_classes)
    
    print(f"\n  📊 Epoch {epoch} Detailed Metrics:")
    print(f"     Non-Vuln: P={precision[0]:.3f}, R={recall[0]:.3f}, F1={f1[0]:.3f}")
    print(f"     Vulnerable: P={precision[1]:.3f}, R={recall[1]:.3f}, F1={f1[1]:.3f}")
    print(f"     Confusion Matrix: [[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]")
