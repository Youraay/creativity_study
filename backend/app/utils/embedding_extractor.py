from PIL import Image as PILImage
import torch
from utils import ModelManager

class EmbeddingExtractor:
    def __init__(self):
        self.model_manager = ModelManager()

    def extract_image_embedding(self, image: PILImage.Image) -> torch.Tensor:
        
        model, processor = self.model_manager.get_blip_model_and_processor()
        inputs = processor(images=image, return_tensors="pt").to(self.model_manager.device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            return outputs.last_hidden_state.mean(dim=1)