import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from torch_geometric.nn import LightGCN, GAT
import timm

def create_collate_fn(tokenizer):
    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        prompts = [item["prompt"] + " " + item["ground_truth"] for item in batch]
        
        encodings = tokenizer(
            prompts, 
            padding=True, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128
        )
        
        node_indices = torch.tensor([item["node_idx"] for item in batch], dtype=torch.long)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "node_indices": node_indices
        }
    return collate_fn

class TinyMultimodalGNNViTLLM(nn.Module):
    def __init__(self, num_nodes, vocab, graph_encoder="LightGCN", embedding_dim=64, llm_model_id="HuggingFaceTB/SmolLM-135M"):
        super().__init__()
        
        self.graph_encoder = graph_encoder
        if graph_encoder == "GAT":
            self.node_embedding = nn.Embedding(num_nodes, embedding_dim)

            self.gnn = GAT(in_channels=embedding_dim, hidden_channels=embedding_dim, num_layers=2, out_channels=embedding_dim)
        else:
            self.gnn = LightGCN(num_nodes=num_nodes, embedding_dim=embedding_dim, num_layers=2)
                
        self.vit = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0)
        vit_out_dim = 192
        
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_id)
        llm_hidden_size = self.llm.config.hidden_size
        
        self.gnn_projector = nn.Sequential(
            nn.Linear(embedding_dim, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        
        self.vit_projector = nn.Sequential(
            nn.Linear(vit_out_dim, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        
        self.heads = nn.ModuleDict({
            "make": nn.Linear(llm_hidden_size, len(vocab["make"]["classes"])),
            "model": nn.Linear(llm_hidden_size, len(vocab["model"]["classes"])),
            "veh_type": nn.Linear(llm_hidden_size, len(vocab["type"]["classes"]))
        })

    def forward(self, edge_index, node_indices, pixel_values, input_ids, attention_mask=None, tasks=None, labels=None):
        if self.graph_encoder == "GAT":
            x = self.node_embedding.weight
            self.gnn(x, edge_index)
        else:
            all_embeddings = self.gnn.get_embedding(edge_index)
        batch_gnn_feats = all_embeddings[node_indices].unsqueeze(1)
        
        vit_feats = self.vit.forward_features(pixel_values)

        if vit_feats.ndim == 4:
            vit_feats = vit_feats.mean(dim=[-1, -2])

        if vit_feats.ndim == 2:
            vit_feats = vit_feats.unsqueeze(1)

        text_embeddings = self.llm.get_input_embeddings()(input_ids) 
        target_dtype = text_embeddings.dtype
        
        gnn_tokens = self.gnn_projector(batch_gnn_feats).to(target_dtype)
        vit_tokens = self.vit_projector(vit_feats).to(target_dtype)
        
        combined_embeddings = torch.cat([gnn_tokens, vit_tokens, text_embeddings], dim=1)
        
        if attention_mask is not None:
            prefix_len = gnn_tokens.shape[1] + vit_tokens.shape[1]
            prefix_mask = torch.ones((attention_mask.shape[0], prefix_len), device=attention_mask.device, dtype=attention_mask.dtype)
            combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            combined_mask = None

        outputs = self.llm(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_mask,
            output_hidden_states=True
        )
        
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        
        logits_dict = {}
        for task_name, head in self.heads.items():
            ext_key = "type" if task_name == "veh_type" else task_name
            logits_dict[ext_key] = head(last_hidden_state)
            
        loss = None
        if labels is not None and tasks is not None:
            loss_fct = nn.CrossEntropyLoss()
            total_loss = 0.0
            for i, task_name in enumerate(tasks):
                internal_key = "veh_type" if task_name == "type" else task_name
                actual_logits = self.heads[internal_key](last_hidden_state[i].unsqueeze(0))
                task_label = labels[i].unsqueeze(0)
                total_loss += loss_fct(actual_logits, task_label)
            loss = total_loss / len(tasks)
            
        return loss, logits_dict