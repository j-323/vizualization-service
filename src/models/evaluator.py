import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPScorer:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def score(self, image_path: str, prompt: str) -> float:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=[prompt], images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits_per_image = self.model(**inputs).logits_per_image
        return logits_per_image.item()