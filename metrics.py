from collections import defaultdict
import csv
import json
import numpy as np
from pathlib import Path
import argparse
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    mean_squared_error, 
    mean_absolute_error,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

def build_graph(file):
    graph = {}
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            if line.strip():
                graph[line] = line.split(' ')
    return graph

def BFS(graph, type_logits, make_logits, model_logits):
    all_probs = []
    for i in range(len(type_logits)):
        probs = []
        for k in graph.keys():
            curr = graph[k]
            prob = type_logits[i][int(curr[0])] + make_logits[i][int(curr[1])] + model_logits[i][int(curr[2])]
            probs.append((curr, prob))
        probs = sorted(probs, key=lambda x: x[1])[::-1]
        all_probs.append(probs)
    return all_probs

def analyze_errors(gts, preds, task_name, output_dir="./", top_n=20):
    total_samples = len(gts)
    
    # 1. Per-class metrics via classification report
    report = classification_report(gts, preds, zero_division=0, output_dict=True)
    class_metrics = []
    
    for cls, metrics in report.items():
        if cls not in ['accuracy', 'macro avg', 'weighted avg']:
            class_metrics.append({
                'class': str(cls),
                'f1': metrics['f1-score'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'support': int(metrics['support'])
            })

    class_metrics = sorted(class_metrics, key=lambda x: x['f1'])

    # 2. Confusion Matrix & Top Specific Errors
    labels = sorted(list(set(gts) | set(preds)))
    cm = confusion_matrix(gts, preds, labels=labels)
    
    errors = []
    tot_count = 0
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and cm[i, j] > 0:
                count = cm[i, j]
                pct = (count / total_samples) * 100
                tot_count += count
                errors.append((count, pct, str(true_label), str(pred_label)))
                
    print(f"\t\t\tTotal error on task {task_name}: {tot_count/total_samples}")
    errors = sorted(errors, key=lambda x: x[0], reverse=True)

    # --- 3. Generate Readable Visualizations ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Full Confusion Matrix with Text Annotations
    if len(labels) <= 150:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  
        cm_norm = cm.astype('float') / row_sums
        cm_pct = cm_norm * 100

        annot_matrix = np.empty(cm.shape, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = cm_pct[i, j]
                if count > 0:
                    annot_matrix[i, j] = f"{count}\n({pct:.1f}%)"
                else:
                    annot_matrix[i, j] = ""

        # Adjust font size dynamically based on number of labels to keep it legible
        font_size = max(9, min(8, int(80 / len(labels))))

        plt.figure(figsize=(max(18, len(labels)*0.4), max(15, len(labels)*0.35)))
        sns.heatmap(cm_pct, annot=annot_matrix, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels, annot_kws={"size": font_size}, vmin=0, vmax=100)
        plt.title(f'Full Confusion Matrix (% of True Class): {task_name.upper()}', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        cm_file = output_path / f"error_analysis_{task_name}_confusion_matrix.png"
        plt.savefig(cm_file, dpi=300)
        plt.close()

    # Sub-Confusion Matrix for Top-N Errors (Focused view)
    if errors:
        top_errors = errors[:top_n]
        top_error_labels = sorted(list(set([t for _, _, t, _ in top_errors] + [p for _, _, _, p in top_errors])))
        
        if len(top_error_labels) > 1:
            cm_sub = confusion_matrix(gts, preds, labels=top_error_labels)
            row_sums_sub = cm_sub.sum(axis=1, keepdims=True)
            row_sums_sub[row_sums_sub == 0] = 1
            cm_sub_norm = cm_sub.astype('float') / row_sums_sub
            cm_sub_pct = cm_sub_norm * 100

            annot_sub = np.empty(cm_sub.shape, dtype=object)
            for i in range(cm_sub.shape[0]):
                for j in range(cm_sub.shape[1]):
                    count = cm_sub[i, j]
                    pct = cm_sub_pct[i, j]
                    if count > 0:
                        annot_sub[i, j] = f"{count}\n({pct:.1f}%)"
                    else:
                        annot_sub[i, j] = ""

            plt.figure(figsize=(max(18, len(top_error_labels)*0.8), max(15, len(top_error_labels)*0.6)))
            sns.heatmap(cm_sub_pct, annot=annot_sub, fmt='', cmap='Blues', xticklabels=top_error_labels, yticklabels=top_error_labels, annot_kws={"size": 9}, vmin=0, vmax=100)
            plt.title(f'Top-{top_n} Errors Sub-Confusion Matrix: {task_name.upper()}', fontsize=14)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            sub_cm_file = output_path / f"error_analysis_{task_name}_top_errors_confusion_matrix.png"
            plt.savefig(sub_cm_file, dpi=300)
            plt.close()

        # Broader Expanded Sub-Confusion Matrix (Top 150 Errors for Combined / High-Cardinality Tasks)
        broad_errors = errors[:150]
        broad_labels = sorted(list(set([t for _, _, t, _ in broad_errors] + [p for _, _, _, p in broad_errors])))
        
        if len(broad_labels) > 1 and len(broad_labels) != len(top_error_labels):
            cm_broad = confusion_matrix(gts, preds, labels=broad_labels)
            row_sums_broad = cm_broad.sum(axis=1, keepdims=True)
            row_sums_broad[row_sums_broad == 0] = 1
            cm_broad_norm = cm_broad.astype('float') / row_sums_broad
            cm_broad_pct = cm_broad_norm * 100

            annot_broad = np.empty(cm_broad.shape, dtype=object)
            for i in range(cm_broad.shape[0]):
                for j in range(cm_broad.shape[1]):
                    count = cm_broad[i, j]
                    pct = cm_broad_pct[i, j]
                    if count > 0:
                        annot_broad[i, j] = f"{count}\n({pct:.1f}%)"
                    else:
                        annot_broad[i, j] = ""

            broad_font_size = max(9, min(7, int(60 / len(broad_labels))))

            plt.figure(figsize=(max(18, len(broad_labels)*0.4), max(15, len(broad_labels)*0.35)))
            sns.heatmap(cm_broad_pct, annot=annot_broad, fmt='', cmap='Blues', xticklabels=broad_labels, yticklabels=broad_labels, annot_kws={"size": broad_font_size}, vmin=0, vmax=100)
            plt.title(f'Expanded Top-150 Errors Confusion Matrix: {task_name.upper()}', fontsize=14)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            broad_cm_file = output_path / f"error_analysis_{task_name}_expanded_errors_confusion_matrix.png"
            plt.savefig(broad_cm_file, dpi=300)
            plt.close()

    # Plot A: Horizontal Bar Chart of Top Errors with Percentage Annotations
    if errors:
        top_errors = errors[:top_n]
        error_labels = [f"True: {t} \n-> Pred: {p}" for _, _, t, p in top_errors]
        error_counts = [c for c, _, _, _ in top_errors]
        error_pcts = [p for _, p, _, _ in top_errors]

        plt.figure(figsize=(18, 15))
        ax = sns.barplot(x=error_counts, y=error_labels, palette='rocket', hue=error_counts[::-1], legend=False)
        
        for p_bar, count, pct in zip(ax.patches, error_counts, error_pcts):
            width = p_bar.get_width()
            ax.text(width + (max(error_counts) * 0.01), p_bar.get_y() + p_bar.get_height() / 2, 
                    f'{count} ({pct:.2f}%)', 
                    ha='left', va='center', fontsize=10, color='black', fontweight='bold')

        plt.title(f'Top {top_n} Misclassification Patterns: {task_name.upper()}', fontsize=14)
        plt.xlabel('Number of Errors (and % of Total Dataset)', fontsize=12)
        plt.ylabel('Misclassification Path', fontsize=12)
        plt.xlim(0, max(error_counts) * 1.25)
        plt.tight_layout()
        err_file = output_path / f"error_analysis_{task_name}_top_errors.png"
        plt.savefig(err_file, dpi=300)
        plt.close()

        class_error_counts = defaultdict(float)
        class_error_pcts = defaultdict(float)
        
        for count, pct, t, _ in errors:
            class_error_counts[str(t)] += count
            class_error_pcts[str(t)] += pct
            
        sorted_class_errors = sorted(class_error_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        full_error_labels = [f"True: {c}" for c, _ in sorted_class_errors]
        full_error_counts = [c for _, c in sorted_class_errors]
        full_error_pcts = [class_error_pcts[c] for c, _ in sorted_class_errors]

        with open(output_path / "top_errors.txt", "w") as f:
            for label, count, pct in zip(full_error_labels[:10], full_error_counts[:10], full_error_pcts[:10]):
                f.write(f"{label} {int(count)} {pct:.2f}\n")

        plt.figure(figsize=(18, 15))
        ax = sns.barplot(x=full_error_counts, y=full_error_labels, palette='rocket', hue=full_error_counts[::-1], legend=False)
        
        for p_bar, count, pct in zip(ax.patches, full_error_counts, full_error_pcts):
            width = p_bar.get_width()
            ax.text(width + (max(full_error_counts) * 0.01), p_bar.get_y() + p_bar.get_height() / 2, 
                    f'{int(count)} ({pct:.2f}%)', 
                    ha='left', va='center', fontsize=10, color='black', fontweight='bold')

        plt.title(f'Top {top_n} misclassification: {task_name.upper()}', fontsize=14)
        plt.xlabel('Number of Errors (and % of Total Dataset)', fontsize=12)
        plt.ylabel('Class', fontsize=12)
        plt.xlim(0, max(full_error_counts) * 1.25)
        plt.tight_layout()
        err_file = output_path / f"error_analysis_{task_name}_top_full_errors.png"
        plt.savefig(err_file, dpi=300)
        plt.close()

    # Plot B: Bar Chart of the 20 Worst-Performing Classes
    if class_metrics:
        worst_classes = class_metrics[:top_n]
        worst_names = [item['class'] for item in worst_classes]
        worst_f1s = [item['f1'] for item in worst_classes]

        plt.figure(figsize=(18, 15))
        ax = sns.barplot(x=worst_f1s[::-1], y=worst_names, palette='coolwarm', hue=worst_f1s[::-1], legend=False)
        
        for p_bar, f1 in zip(ax.patches, worst_f1s):
            width = p_bar.get_width()
            ax.text(width + 0.02, p_bar.get_y() + p_bar.get_height() / 2, 
                    f'{f1:.4f}', 
                    ha='left', va='center', fontsize=10, color='black', fontweight='bold')

        plt.title(f'Bottom {top_n} Classes by F1-Score: {task_name.upper()}', fontsize=14)
        plt.xlabel('F1-Score', fontsize=12)
        plt.ylabel('Class', fontsize=12)
        plt.xlim(0, 1.2)
        plt.tight_layout()
        worst_file = output_path / f"error_analysis_{task_name}_worst_classes.png"
        plt.savefig(worst_file, dpi=300)
        plt.close()

def evaluate_results(npz_path, tasks_to_eval, vocab_path=None, split_types=["val"]):
    graph = build_graph('./valid_tuples.txt')

    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Results file not found at: {npz_path}")
    
    vocab = {}
    if vocab_path and Path(vocab_path).exists():
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)

    data = np.load(npz_path, allow_pickle=True)
    evaluation_results = {}

    for split_type in split_types:
        print(f"\n========================================")
        print(f"        SPLIT: {split_type.upper()}")
        print(f"========================================")
        
        split_preds_dict = {}
        split_gts_dict = {}
        all_logits = []

        for task in tasks_to_eval:
            logits_key = f"{split_type}_{task}_logits"
            preds_key = f"{split_type}_{task}_preds"
            gts_key = f"{split_type}_{task}_gts"
            ids_key = f"{split_type}_{task}_ids"
            
            if gts_key not in data:
                print(f"Skipping [{split_type} | {task}]: Ground truth key not found in .npz file.")
                continue
                
            task_vocab = vocab.get(task, {}).get("idx_to_class", {})
            gts = [task_vocab.get(str(gt), gt).lower().strip() for gt in data[gts_key]]
            
            if logits_key in data:
                logits = np.array(data[logits_key])
                if logits.ndim > 1:
                    pred_indices = np.argmax(logits, axis=-1)
                else:
                    pred_indices = logits.astype(int)
                all_logits.append(logits)
            else:
                if preds_key not in data:
                    print(f"Skipping [{split_type} | {task}]: Neither logits nor preds found.")
                    continue
                pred_indices = data[preds_key]
                all_logits.append(None)
            
            preds = []
            for idx in pred_indices:
                idx_str = str(int(idx))
                mapped_val = task_vocab.get(idx_str, str(idx))
                preds.append(str(mapped_val).lower().strip())
            
            # print(preds, '\n', gts)

            split_preds_dict[task] = preds
            split_gts_dict[task] = gts
            
            print(f"\n--- Task: {task.upper()} (Logits Mapped via Vocab) ---")
            print(f"Total samples: {len(preds)}")
            
            try:
                metrics = {
                    "accuracy": accuracy_score(gts, preds),
                    "f1_score": f1_score(gts, preds, average="weighted", zero_division=0),
                    "precision": precision_score(gts, preds, average="weighted", zero_division=0),
                    "recall": recall_score(gts, preds, average="weighted", zero_division=0)
                }
            except Exception:
                metrics = {
                    "accuracy": accuracy_score(gts, preds),
                    "f1_score": f1_score(gts, preds, average="weighted", zero_division=0)
                }
            
            evaluation_results[f"{split_type}_{task}"] = metrics
            for metric_name, value in metrics.items():
                print(f"  {metric_name.upper()}: {value:.4f}")

            analyze_errors(gts, preds, f'{task}_{split_type}', args.metric_path, top_n=30)

        # --- Combined Task Evaluation ---
        if len(split_preds_dict) == len(tasks_to_eval) and len(tasks_to_eval) > 1:
            combined_task_name = "_".join(tasks_to_eval)
            num_samples = len(next(iter(split_preds_dict.values())))
            
            combined_gts = [
                "_".join(str(split_gts_dict[task][i]) for task in tasks_to_eval)
                for i in range(num_samples)
            ]
            
            # Method 1: Simple AND (Concatenation of individual task predictions)
            print(f"\n--- COMBINED TASK: {combined_task_name.upper()} (Simple AND Concatenation) ---")
            combined_preds_simple = [
                "_".join(str(split_preds_dict[task][i]) for task in tasks_to_eval)
                for i in range(num_samples)
            ]
            
            combined_metrics_simple = {
                "accuracy": accuracy_score(combined_gts, combined_preds_simple),
                "f1_score": f1_score(combined_gts, combined_preds_simple, average="weighted", zero_division=0),
                "precision": precision_score(combined_gts, combined_preds_simple, average="weighted", zero_division=0),
                "recall": recall_score(combined_gts, combined_preds_simple, average="weighted", zero_division=0)
            }
            evaluation_results[f"{split_type}_{combined_task_name}_combined_simple"] = combined_metrics_simple
            for metric_name, value in combined_metrics_simple.items():
                print(f"  COMBINED {metric_name.upper()}: {value:.4f}")

            analyze_errors(combined_gts, combined_preds_simple, f"{combined_task_name}_simple", args.metric_path, top_n=30)

            # Method 2: Logit-Sum / BFS over Valid Tuples (Max Sum of Logits restricted to Valid Combinations)
            print(f"\n--- COMBINED TASK: {combined_task_name.upper()} (Logit Sum Max over Valid Tuples, Top-{args.top_k}) ---")
            if all(l is not None for l in all_logits) and len(tasks_to_eval) == 3:
                type_logits, make_logits, model_logits = all_logits
                sorted_tuples = BFS(graph, type_logits, make_logits, model_logits)

                mapped_sample_tuples = []
                for sample_tuples in sorted_tuples:
                    mapped_tuples_for_sample = []
                    for t, prob in sample_tuples:
                        mapped_elements = []
                        for idx_str, task_name in zip(t, tasks_to_eval):
                            task_vocab = vocab.get(task_name, {}).get("idx_to_class", {})
                            mapped_val = task_vocab.get(str(int(idx_str)), idx_str)
                            mapped_elements.append(str(mapped_val).lower().strip())
                        mapped_tuple_str = "_".join(mapped_elements)
                        mapped_tuples_for_sample.append(mapped_tuple_str)
                    mapped_sample_tuples.append(mapped_tuples_for_sample)

                combined_metrics_logits = {
                    "accuracy": -1,
                    "f1_score": -1,
                    "precision": -1,
                    "recall": -1
                }

                for k in range(1, args.top_k + 1):
                    current_preds = [
                        sample_preds[k-1] if len(sample_preds) >= k else sample_preds[-1] 
                        for sample_preds in mapped_sample_tuples
                    ]
                    
                    combined_metrics_logits["accuracy"] = max(combined_metrics_logits["accuracy"], accuracy_score(combined_gts, current_preds))
                    combined_metrics_logits["f1_score"] = max(combined_metrics_logits["f1_score"], f1_score(combined_gts, current_preds, average="weighted", zero_division=0))
                    combined_metrics_logits["precision"] = max(combined_metrics_logits["precision"], precision_score(combined_gts, current_preds, average="weighted", zero_division=0))
                    combined_metrics_logits["recall"] = max(combined_metrics_logits["recall"], recall_score(combined_gts, current_preds, average="weighted", zero_division=0))

                    if k == 1:
                        analyze_errors(combined_gts, current_preds, f"{combined_task_name}_logits_top1", args.metric_path, top_n=30)

                evaluation_results[f"{split_type}_{combined_task_name}_combined_logits"] = combined_metrics_logits
                for metric_name, value in combined_metrics_logits.items():
                    print(f"  COMBINED {metric_name.upper()}: {value:.4f}")
            else:
                print("Skipping Logit Sum BFS evaluation (requires exactly 3 tasks and available logits).")

    output_path = Path(args.metric_path)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_file_path = output_path / "evaluation_metrics_summary.csv"

    with open(csv_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Evaluation Key', 'Metric', 'Value'])
        for eval_key, metrics in evaluation_results.items():
            for metric_name, value in metrics.items():
                writer.writerow([eval_key, metric_name, value])

    print(f"\nMetrics successfully exported to {csv_file_path}")
    return evaluation_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation metrics using vocab mapping and dual combined strategies.")
    parser.add_argument("--metric_path", default='./metrics', type=str, help="Path to results .npz file")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to results .npz file")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocab.json file")
    parser.add_argument("--tasks", nargs="+", required=True, help="List of component tasks (e.g., type make model)")
    parser.add_argument("--splits", nargs="+", default=["val"], help="List of split types")
    parser.add_argument("--top_k", type=int, default=1, help="Test if result is in top k")
    
    args = parser.parse_args()
    
    evaluate_results(
        npz_path=args.npz_path,
        tasks_to_eval=args.tasks,
        vocab_path=args.vocab_path,
        split_types=args.splits
    )