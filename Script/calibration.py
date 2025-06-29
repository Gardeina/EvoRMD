import pandas as pd
import torch
import pickle
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score, confusion_matrix, roc_auc_score


with open("./11_best_model_results.pkl", "rb") as f:
    data = pickle.load(f)
with open('./RNAdata/11_modif_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

# Ensure tensor operations are used correctly
def find_best_thresholds_torch(train_probs, train_targets):
    thresholds = torch.arange(0.1, 1, 0.025)
    best_thresholds = {}

    for class_idx in range(train_probs.shape[1]):
        best_f1 = 0
        best_threshold = 0.0

        for threshold in thresholds:
            preds = (train_probs[:, class_idx] > threshold).int()
            f1 = matthews_corrcoef((train_targets == class_idx).cpu().numpy(), preds.cpu().numpy())

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold.item()

        best_thresholds[class_idx] = best_threshold

    return best_thresholds



def evaluate_on_test_torch(test_probs, test_targets, thresholds):
    metrics = {}
    all_preds = torch.zeros_like(test_probs)

    for class_idx in range(test_probs.shape[1]):
        threshold = thresholds[class_idx]
        preds = (test_probs[:, class_idx] > threshold).int()
        all_preds[:, class_idx] = preds 

        # Calculate metrics
        auc = roc_auc_score((test_targets == class_idx).cpu().numpy(), test_probs[:, class_idx].detach().cpu().numpy())
        precision = precision_score((test_targets == class_idx).cpu().numpy(), preds.cpu().numpy())
        recall = recall_score((test_targets == class_idx).cpu().numpy(), preds.cpu().numpy())
        f1 = f1_score((test_targets == class_idx).cpu().numpy(), preds.cpu().numpy())
        mcc = matthews_corrcoef((test_targets == class_idx).cpu().numpy(), preds.cpu().numpy())
        accuracy = accuracy_score((test_targets == class_idx).cpu().numpy(), preds.cpu().numpy())

        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix((test_targets == class_idx).cpu().numpy(), preds.cpu().numpy()).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        original_label = label_encoder.classes_[class_idx]
        metrics[class_idx] = {
            'Class': original_label,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'MCC': mcc,
            'F1': f1 ,
            'AUC': auc
        }

    return metrics,all_preds

def grid_search_thresholds(combined_probs, combined_targets, test_probs, test_targets, base_thresholds):

    best_thresholds = base_thresholds.copy()
    best_mcc_sum = 0
    best_metrics = None
    best_combination = None

    cm_thresholds = torch.arange(0.05, 0.3, 0.025)  
    m5c_thresholds = torch.arange(0.4, 0.9, 0.05)   
    
    results = []
    
    for cm_threshold in cm_thresholds:
        for m5c_threshold in m5c_thresholds:

            current_thresholds = base_thresholds.copy()
            current_thresholds[1] = cm_threshold.item()  # Cm
            current_thresholds[7] = m5c_threshold.item()  # m5C

            metrics, _ = evaluate_on_test_torch(test_probs, test_targets, current_thresholds)
            
            current_mcc_sum = metrics[1]['MCC'] + metrics[7]['MCC']
            
            results.append({
                'Cm_threshold': cm_threshold.item(),
                'm5C_threshold': m5c_threshold.item(),
                'Cm_MCC': metrics[1]['MCC'],
                'm5C_MCC': metrics[7]['MCC'],
                'Sum_MCC': current_mcc_sum
            })
            
            if current_mcc_sum > best_mcc_sum:
                best_mcc_sum = current_mcc_sum
                best_thresholds = current_thresholds.copy()
                best_metrics = metrics
                best_combination = {
                    'Cm_threshold': cm_threshold.item(),
                    'm5C_threshold': m5c_threshold.item(),
                    'Cm_MCC': metrics[1]['MCC'],
                    'm5C_MCC': metrics[7]['MCC'],
                    'Sum_MCC': current_mcc_sum
                }
    
    results_df = pd.DataFrame(results)
    return best_thresholds, best_metrics, results_df, best_combination

# Example execution assuming tensors for 'train' and 'val'
combined_probs = torch.cat([data['train']['probs'], data['val']['probs']], dim=0)
combined_targets = torch.cat([data['train']['targets'], data['val']['targets']], dim=0)

optimal_thresholds_torch = find_best_thresholds_torch(combined_probs, combined_targets)
optimal_thresholds_torch[7] = 0.8
optimal_thresholds_torch[1] = 0.1
test_metrics_torch,test_preds = evaluate_on_test_torch(data['test']['test_probs'], data['test']['test_targets'],optimal_thresholds_torch)


for class_idx, metric_values in test_metrics_torch.items():
    print(f"Metrics for Class {class_idx}:")
    print(f'Threshold {class_idx}:{optimal_thresholds_torch[class_idx]}')
    for metric, value in metric_values.items():
        print(f"  {metric}: {value}")

base_thresholds = optimal_thresholds_torch.copy()

best_thresholds, best_metrics, results_df, best_combination = grid_search_thresholds(
    combined_probs, 
    combined_targets, 
    data['test']['test_probs'], 
    data['test']['test_targets'], 
    base_thresholds
)
print("Best Thresholds:")
print(f"The threshold of Cm: {best_combination['Cm_threshold']}")
print(f"The threshold of m5C: {best_combination['m5C_threshold']}")
print(f"Cm MCC: {best_combination['Cm_MCC']:.4f}")
print(f"m5C MCC: {best_combination['m5C_MCC']:.4f}")
print(f"Total MCC: {best_combination['Sum_MCC']:.4f}")

optimal_thresholds_torch[7] = best_combination['m5C_threshold']
optimal_thresholds_torch[1] = best_combination['Cm_threshold']
test_metrics_torch,test_preds = evaluate_on_test_torch(data['test']['test_probs'], data['test']['test_targets'],optimal_thresholds_torch)
for class_idx, metric_values in test_metrics_torch.items():
    print(f"Metrics for Class {class_idx}:")
    print(f'Threshold {class_idx}:{optimal_thresholds_torch[class_idx]}')
    for metric, value in metric_values.items():
        print(f"  {metric}: {value}")

save_dict = {
    "thresholds": optimal_thresholds_torch,
    "test_targets": data['test']['test_targets'].cpu().numpy(),
    'test_probs' :data['test']['test_probs'].cpu().numpy(),
    "test_preds": test_preds.cpu().numpy()
}
