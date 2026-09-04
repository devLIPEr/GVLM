from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms

class VeSVDatasetWithGraph(Dataset):
    def __init__(self, split_txt_path, images_dir, annotations_dict, vocab, image_to_node_map, min_limit_size=-1, max_limit_size=-1):
        self.images_dir = Path(images_dir)
        self.tasks = ["make", "model", "type"]
        self.image_to_node_map = image_to_node_map
        
        with open(split_txt_path, 'r') as f:
            self.image_ids = [line.strip() for line in f if line.strip()]

        if min_limit_size != -1 and max_limit_size != -1:
            self.image_ids = self.image_ids[min_limit_size:max_limit_size]
        elif min_limit_size != -1:
            self.image_ids = self.image_ids[min_limit_size:]
        elif max_limit_size != -1:
            self.image_ids = self.image_ids[:max_limit_size]
            
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        num_types = len(vocab["type"]["classes"])
        num_makes = len(vocab["make"]["classes"])

        model_offset = num_types + num_makes
            
        self.samples = []
        for img_id in self.image_ids:
            if img_id in annotations_dict:
                ann = annotations_dict[img_id]
                task_prompts = {
                    "make": f"What is the vehicle make? Choose from: {', '.join(vocab['make']['classes'])}",
                    "model": f"What is the vehicle model? Choose from: {', '.join(vocab['model']['classes'])}",
                    "type": f"What is the vehicle type? Choose from: {', '.join(vocab['type']['classes'])}"
                }

                t = ann.get("type")
                mk = ann.get("make")
                md = ann.get("model")

                model_idx = vocab["model"]["classes"].index(md) if md in vocab["model"]["classes"] else 0

                node_idx = model_idx + model_offset

                for task in self.tasks:
                    if task in ann and ann[task] is not None:
                        self.samples.append({
                            "image_id": img_id,
                            "task": task,
                            "prompt": task_prompts[task],
                            "ground_truth": str(ann[task]).strip(),
                            "node_idx": node_idx
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_id = sample["image_id"]
        
        img_path = self.images_dir / img_id
        if not img_path.exists():
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
                if (self.images_dir / (img_id + ext)).exists():
                    img_path = self.images_dir / (img_id + ext)
                    break

        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)
        
        return {
            "pixel_values": pixel_values,
            "image_id": sample["image_id"],
            "prompt": sample["prompt"],
            "ground_truth": sample["ground_truth"],
            "node_idx": sample["node_idx"],
            "task": sample["task"]
        }