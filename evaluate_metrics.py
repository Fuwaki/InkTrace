
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from models import ModelFactory
from datasets import InkTraceDataset
import argparse

def get_matcher_indices(pred_strokes, pred_logits, tgt_strokes, tgt_classes):
    """
    Hungarian matching for evaluation.
    Computes cost based on L1 distance of control points and negative class probability.
    """
    # pred_strokes: [K, 10]
    # pred_logits: [K, 3]
    # tgt_strokes: [M, 10]
    # tgt_classes: [M] (0=Null, 1=New, 2=Cont)
    
    # Filter only valid targets for matching cost calculation geometry part
    # But for DETR matching we usually match everything to everything (including padding)
    # Here we simplify: We match K preds to M targets.
    
    K = pred_strokes.shape[0]
    M = tgt_strokes.shape[0]
    
    # Cost Matrix
    # 1. Class Cost: -log(p(class))
    pred_probs = pred_logits.softmax(-1) # [K, 3]
    
    # We want indices for [K, M]
    # cost_class[i, j] = -prob[i, tgt_class[j]]
    cost_class = -pred_probs[:, tgt_classes.long()] # [K, M]
    
    # 2. Coord Cost: L1 distance
    # Expand for broadcasting: [K, 1, 10] - [1, M, 10]
    diff = torch.abs(pred_strokes.unsqueeze(1) - tgt_strokes.unsqueeze(0))
    cost_coord = diff.sum(-1) # [K, M]
    
    # Combine
    C = 2.0 * cost_class + 5.0 * cost_coord
    
    C = C.cpu().numpy()
    indices = linear_sum_assignment(C)
    
    return indices

def evaluate(model_path, device_name="cuda"):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading model from {model_path}...")
    try:
        model = ModelFactory.load_vectorization_model(model_path, device=device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Dataset (Independent Mode - Phase 2)
    # Generate 50 batches of 64 samples = 3200 samples
    dataset_size = 3200
    batch_size = 64
    
    dataset = InkTraceDataset(
        mode="independent",
        img_size=64,
        batch_size=batch_size,
        epoch_length=dataset_size,
        max_strokes=8,
        fixed_count=None, # Random count
        for_detr=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    
    # Metrics
    total_samples = 0
    total_l1_error = 0.0
    total_l2_error = 0.0
    matched_strokes_count = 0
    
    # Detection Metrics
    total_gt_strokes = 0
    total_pred_strokes = 0 # Predicted as non-null
    true_positives = 0
    
    # Class Confusion: GT vs Pred (for matched pairs)
    # Rows: GT (Null, New, Cont), Cols: Pred (Null, New, Cont)
    confusion_matrix = np.zeros((3, 3), dtype=int)

    print("Starting evaluation...")
    
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            # targets: [B, 8, 11]
            
            # Forward
            outputs = model(imgs, mode="vectorize")
            # outputs is tuple: (strokes, logits, ...)
            pred_strokes_batch = outputs[0] # [B, 8, 10]
            pred_logits_batch = outputs[1]  # [B, 8, 3]
            
            batch_num = imgs.shape[0]
            
            for i in range(batch_num):
                # Per sample processing
                p_strokes = pred_strokes_batch[i] # [8, 10]
                p_logits = pred_logits_batch[i]   # [8, 3]
                t_target = targets[i]             # [8, 11]
                
                # Split Target
                t_strokes = t_target[:, :10]
                t_valid = t_target[:, 10] # 1.0 = Valid (New), 0.0 = Pad (Null)
                
                # Convert valid flag to class index for matching
                # In independent mode, valid=1 means Class 1 (New), valid=0 means Class 0 (Null)
                t_classes = t_valid.long() 
                
                # Get Assignments
                row_idx, col_idx = get_matcher_indices(p_strokes, p_logits, t_strokes, t_classes)
                
                # Collect Metrics
                
                # 1. Detection Stats (Based on Argmax)
                p_classes_pred = p_logits.argmax(-1) # [8]
                
                # Actual GT strokes count (Class != 0)
                gt_count = (t_classes != 0).sum().item()
                # Predicted strokes count (Class != 0)
                pred_count = (p_classes_pred != 0).sum().item()
                
                total_gt_strokes += gt_count
                total_pred_strokes += pred_count
                
                # 2. Analyze Matches
                for r, c in zip(row_idx, col_idx):
                    # r is pred index, c is target index
                    gt_cls = t_classes[c].item()
                    pred_cls = p_classes_pred[r].item()
                    
                    # Update Confusion Matrix
                    # Note: This is conditioned on Hungarian Matching being "correct"
                    confusion_matrix[gt_cls, pred_cls] += 1
                    
                    if gt_cls != 0:
                        # This matches a valid GT stroke
                        
                        # Check Geometry Error
                        dist_l1 = torch.abs(p_strokes[r] - t_strokes[c]).sum().item()
                        dist_l2 = torch.norm(p_strokes[r] - t_strokes[c], p=2).item()
                        
                        # Only count geometry error if class was correctly predicted as Non-Null
                        # (Or should we count it anyway? Usually depends on task. Let's count if matched to GT)
                        total_l1_error += dist_l1
                        total_l2_error += dist_l2
                        matched_strokes_count += 1
                        
                        if pred_cls != 0:
                            true_positives += 1

            total_samples += batch_num
            print(f"\rProcessed {total_samples}/{dataset_size}", end="")
            
    print("\n\n=== Evaluation Results ===")
    print(f"Total Samples: {total_samples}")
    print(f"Total GT Strokes: {total_gt_strokes}")
    print(f"Total Pred Strokes: {total_pred_strokes}")
    print("-" * 30)
    
    # Precision / Recall / F1
    # TP: Predicted Non-Null AND Matched to GT Non-Null
    # FP: Predicted Non-Null BUT Matched to GT Null (or unmatched - handled by fixed slots)
    # FN: GT Non-Null BUT Matched to Pred Null
    
    # Logic adjustment:
    # Hungarian matcher matches everyone.
    # If GT is Class 1, and Pred is Class 1 -> TP
    # If GT is Class 1, and Pred is Class 0 -> FN
    # If GT is Class 0, and Pred is Class 1 -> FP
    # If GT is Class 0, and Pred is Class 0 -> TN
    
    # From Confusion Matrix
    # GT \ Pred | 0 (Null) | 1 (New) | 2 (Cont)
    # 0 (Null)  | TN       | FP      | FP
    # 1 (New)   | FN       | TP      | TP (Labelled distinct)
    # 2 (Cont)  | FN       | TP      | TP
    
    # In independent mode, we mainly care about Class 1 (New) vs 0 (Null).
    
    # Recalculate from Confusion Matrix for Robustness
    TP = confusion_matrix[1:, 1:].sum() # GT != 0 and Pred != 0
    FP = confusion_matrix[0, 1:].sum()  # GT == 0 and Pred != 0
    FN = confusion_matrix[1:, 0].sum()  # GT != 0 and Pred == 0
    TN = confusion_matrix[0, 0].sum()
    
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("-" * 30)
    print("Confusion Matrix:")
    print("Pred ->  Null   New   Cont")
    print(f"GT Null: {confusion_matrix[0]}")
    print(f"GT New:  {confusion_matrix[1]}")
    print(f"GT Cont: {confusion_matrix[2]}")
    
    print("-" * 30)
    if matched_strokes_count > 0:
        avg_l1 = total_l1_error / matched_strokes_count
        avg_l2 = total_l2_error / matched_strokes_count
        print(f"Matched Strokes Geometry Error (Normalized [0-1] scale approx):")
        print(f"MAE (L1): {avg_l1:.4f}")
        print(f"MSE (L2): {avg_l2:.4f}")
        # Note: Strokes are in [-0.5, 1.5] space roughly, but GT is [0, 1].
        # Error 0.05 means 5% of canvas size deviation per param.
    else:
        print("No matched strokes.")

if __name__ == "__main__":
    evaluate("best_detr_independent.pth")
