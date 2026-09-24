import argparse
import json
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from gvlm import *
from dataset import *
import gc
import os

def parse_intervals(interval_str: str) -> list[tuple[int, int]]:
    """
    Converts a string of intervals/numbers (e.g., '0-3, 4, 6-9') 
    into a list of tuples [(0, 3), (4, 4), (6, 9)].
    """
    intervals = []
    for part in interval_str.split(","):
        part = part.strip()
        if not part:
            continue
            
        if "-" in part:
            start, end = part.split("-")
            intervals.append((int(start.strip()), int(end.strip())))
        else:
            val = int(part)
            intervals.append((val, val))
            
    return intervals

parser = argparse.ArgumentParser(description="Vehicle Make, Model, Type VLM Training & Evaluation")
parser.add_argument("--fold", type=str, required=True, default="0-9", help="Fold index (0-9)")
parser.add_argument("--model", type=str, default="arnir0/Tiny-LLM", help="Hugging Face model ID or local path")
parser.add_argument("--images_dir", type=str, default="../images", help="Path to images directory")
parser.add_argument("--splits_dir", type=str, default="../splits", help="Path to splits directory")
parser.add_argument("--annotations", type=str, default="../annotations.json", help="Path to annotations.json")
parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save checkpoints and npz")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model")
parser.add_argument("--min_limit_size", type=int, default=-1, help="Limit starting point")
parser.add_argument("--max_limit_size", type=int, default=-1, help="Limit ending point")
parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--graph_encoder", type=str, choices=["LightGCN", "GAT"], default="LightGCN", help="Graph encoder (LightGCN, GAT)")

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

with open(args.annotations, 'r') as f:
    annotations_list = json.load(f)

annotations_dict = {str(item.get("filename")): item for item in annotations_list}

makes = sorted(list(set(ann.get("make") for ann in annotations_dict.values() if ann.get("make"))))
models = sorted(list(set(ann.get("model") for ann in annotations_dict.values() if ann.get("model"))))
types = sorted(list(set(ann.get("type") for ann in annotations_dict.values() if ann.get("type"))))

vocab = {
    "type": {"classes": types},
    "make": {"classes": makes},
    "model": {"classes": models}
}

make_to_types = {}
model_to_makes = {}
for ann in annotations_dict.values():
    t = ann.get("type")
    mk = ann.get("make")
    md = ann.get("model")
    if t and mk:
        make_to_types.setdefault(mk, set()).add(t)
    if mk and md:
        model_to_makes.setdefault(md, set()).add(mk)

type_classes = vocab["type"]["classes"]
make_classes = vocab["make"]["classes"]
model_classes = vocab["model"]["classes"]

make_type_mask = torch.zeros((len(make_classes), len(type_classes)), dtype=torch.bool, device=args.device)
for mk_idx, mk in enumerate(make_classes):
    for t in make_to_types.get(mk, set()):
        if t in type_classes:
            make_type_mask[mk_idx, type_classes.index(t)] = True

model_make_mask = torch.zeros((len(model_classes), len(make_classes)), dtype=torch.bool, device=args.device)
for md_idx, md in enumerate(model_classes):
    for mk in model_to_makes.get(md, set()):
        if mk in make_classes:
            model_make_mask[md_idx, make_classes.index(mk)] = True

tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token

def get_dataloader(fold, split_name, is_train=True):
    split_txt_path = Path(args.splits_dir) / f"{fold}" / f"{split_name}.txt"
    if not split_txt_path.exists():
        split_txt_path = Path(args.splits_dir) / f"{split_name}.txt"

    with open(split_txt_path, 'r') as f:
        image_ids = [line.strip() for line in f if line.strip()]

    image_to_node_map = {img_id: idx for idx, img_id in enumerate(image_ids)}
    
    dataset = VeSVDatasetWithGraph(
        split_txt_path=split_txt_path,
        images_dir=args.images_dir,
        annotations_dict=annotations_dict,
        vocab=vocab,
        image_to_node_map=image_to_node_map,
        min_limit_size=args.min_limit_size,
        max_limit_size=args.max_limit_size
    )
    
    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        prompts = [item["prompt"] for item in batch]
        gts = [item["ground_truth"] for item in batch]
        img_ids = [item.get("image_id", "") for item in batch]
        tasks = [item.get("task", "") for item in batch]
        
        labels = []
        for task, gt in zip(tasks, gts):
            classes = vocab[task]["classes"]
            labels.append(classes.index(gt) if gt in classes else 0)
            
        encodings = tokenizer(prompts, padding=True, return_tensors="pt", truncation=True, max_length=128)
        node_indices = torch.tensor([item["node_idx"] for item in batch], dtype=torch.long)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "node_indices": node_indices,
            "ground_truth": gts,
            "image_id": img_ids,
            "task": tasks,
            "labels": torch.tensor(labels, dtype=torch.long)
        }
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=is_train, 
        collate_fn=collate_fn
    )
    return loader, len(image_to_node_map)

def run_inference_and_collect(model, dataloader, edge_index, device, split_name="test"):
    model.eval()
    tasks_list = ["type", "make", "model"]
    results = {task: {"preds": [], "gts": [], "ids": [], "logits": []} for task in tasks_list}
    
    with torch.inference_mode():
        progress_bar = tqdm(dataloader, desc=f"Inference [{split_name}]")
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            node_indices = batch["node_indices"].to(device, non_blocking=True)
            img_ids = batch["image_id"]
            gt_values = batch["ground_truth"]
            batch_tasks = batch["task"]
            
            _, logits_dict = model(
                edge_index=edge_index,
                node_indices=node_indices,
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            for i in range(pixel_values.shape[0]):
                task = batch_tasks[i]
                img_id = img_ids[i]
                gt = gt_values[i]
                
                classes = vocab[task]["classes"]
                task_logits = logits_dict[task][i].float()
                
                if task == "make":
                    type_logits = logits_dict["type"][i].float()
                    best_type_idx = torch.argmax(type_logits)
                    valid_makes_mask = make_type_mask[:, best_type_idx]
                    task_logits[~valid_makes_mask] = -float('inf')
                elif task == "model":
                    make_logits = logits_dict["make"][i].float()
                    best_make_idx = torch.argmax(make_logits)
                    valid_models_mask = model_make_mask[:, best_make_idx]
                    task_logits[~valid_models_mask] = -float('inf')
                
                class_probs = torch.softmax(task_logits, dim=0).cpu().numpy()
                best_idx = np.argmax(class_probs)
                best_pred = classes[best_idx]
                
                results[task]["preds"].append(best_pred)
                results[task]["gts"].append(gt)
                results[task]["ids"].append(img_id)
                results[task]["logits"].append(class_probs)

    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    return results

output_path = Path(args.output_dir)
output_path.mkdir(parents=True, exist_ok=True)

folds = parse_intervals(args.fold)

num_types = len(vocab["type"]["classes"])
num_makes = len(vocab["make"]["classes"])
num_models = len(vocab["model"]["classes"])

make_offset = num_types
model_offset = num_types + num_makes
total_class_nodes = num_types + num_makes + num_models

edges_u = []
edges_v = []

tuples_path = Path("../valid_tuples.txt")
if tuples_path.exists():
    with open(tuples_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                t_idx, mk_idx, model_idx = map(int, parts)
                
                t_node = t_idx
                mk_node = mk_idx + make_offset
                model_node = model_idx + model_offset
                
                edges_u.extend([t_node, mk_node, mk_node, model_node])
                edges_v.extend([mk_node, t_node, model_node, mk_node])

for i in range(total_class_nodes):
    edges_u.append(i)
    edges_v.append(i)

edge_index = torch.tensor([edges_u, edges_v], dtype=torch.long).to(args.device)

for s, e in folds:
    for fold in range(s, e+1):
        train_loader, num_train_nodes = get_dataloader(str(fold), "train", is_train=True)
        val_loader, _ = get_dataloader(str(fold), "val", is_train=False)
        test_loader, _ = get_dataloader(str(fold), "test", is_train=False)

        # edge_index = torch.stack([
        #     torch.arange(num_train_nodes, dtype=torch.long),
        #     torch.arange(num_train_nodes, dtype=torch.long)
        # ], dim=0).to(args.device)

        model = TinyMultimodalGNNViTLLM(
            num_nodes=total_class_nodes, 
            vocab=vocab,
            embedding_dim=64, 
            llm_model_id=args.model
        ).to(args.device)

        for param in model.llm.parameters():
            param.requires_grad = False

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        print("\n--- Running Zero-Shot Evaluation on Validation Split ---")
        zeroshot_test_res = run_inference_and_collect(model, test_loader, edge_index, args.device, split_name="zeroshot_test")

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr, 
            weight_decay=0.01
        )

        training_history = {"epoch": [], "train_loss": [], "learning_rate": []}

        model.train()
        for epoch in range(args.epochs):
            total_loss = 0.0
            num_batches = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
            
            for batch_idx, batch in enumerate(progress_bar):
                pixel_values = batch["pixel_values"].to(args.device)
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                node_indices = batch["node_indices"].to(args.device)
                labels = batch["labels"].to(args.device)
                tasks = batch["task"]

                optimizer.zero_grad()
                
                loss, logits_dict = model(
                    edge_index=edge_index,
                    node_indices=node_indices,
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tasks=tasks,
                    labels=labels
                )
                
                loss_fct = nn.CrossEntropyLoss()
                pred_indices = {t: torch.argmax(logits_dict[t], dim=-1) for t in ["type", "make", "model"]}
                
                hierarchical_penalty = 0.0
                for i, task_name in enumerate(tasks):
                    if task_name == "make":
                        mk_idx = pred_indices["make"][i]
                        t_idx = pred_indices["type"][i]
                        if not make_type_mask[mk_idx, t_idx]:
                            hierarchical_penalty += 1.5
                    elif task_name == "model":
                        md_idx = pred_indices["model"][i]
                        mk_idx = pred_indices["make"][i]
                        if not model_make_mask[md_idx, mk_idx]:
                            hierarchical_penalty += 2.0
                            
                loss = loss + (0.1 * hierarchical_penalty / len(tasks))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix(loss=f"{total_loss/num_batches:.4f}")

            avg_loss = total_loss / num_batches
            current_lr = optimizer.param_groups[0]['lr']
            
            training_history["epoch"].append(epoch + 1)
            training_history["train_loss"].append(avg_loss)
            training_history["learning_rate"].append(current_lr)

        model_save_path = output_path / f"fold_{fold}_model.pt"
        torch.save({
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vocab': vocab,
            'num_nodes': num_train_nodes
        }, model_save_path)
        print(f"Trained model checkpoint saved to {model_save_path}")

        history_file = output_path / f"fold_{fold}_training_history.json"
        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=4)
        print(f"Training history saved to {history_file}")

        print("\n--- Running Post-Training Evaluation ---")
        val_res = run_inference_and_collect(model, val_loader, edge_index, args.device, split_name="val")
        test_res = run_inference_and_collect(model, test_loader, edge_index, args.device, split_name="test")

        print("\nExporting all results to .npz...")
        npz_data = {}

        for split_prefix, res_dict in [("zero", zeroshot_test_res), ("val", val_res), ("test", test_res)]:
            for task, metrics in res_dict.items():
                npz_data[f"{split_prefix}_{task}_preds"] = np.array(metrics["preds"], dtype=object)
                npz_data[f"{split_prefix}_{task}_gts"] = np.array(metrics["gts"], dtype=object)
                npz_data[f"{split_prefix}_{task}_ids"] = np.array(metrics["ids"], dtype=object)
                npz_data[f"{split_prefix}_{task}_logits"] = np.array(metrics["logits"], dtype=object)

        npz_filepath = output_path / f"fold_{fold}_results.npz"
        np.savez(npz_filepath, **npz_data)
        print(f"All split results saved to {npz_filepath}")