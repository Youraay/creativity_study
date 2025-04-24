import torch
from diffusers import DiffusionPipeline
from transformers import Blip2Model, AutoProcessor, CLIPModel, CLIPProcessor
import config
class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._sdxl_base = None
        self._sdxl_refiner = None
        self._blip_model = None
        self._blip_processor = None
        self._clip_model = None
        self._clip_processor = None
        self.use_half_precision = True
        

        self.n_steps = 50
        self.high_noise_frac = 0.8
        
        self._initialized = True
    
    def get_sdxl_base(self):
        
        self.free_memory(keep_models=['base'])
        
        if self._sdxl_base is None:
            print("Loading SDXL Base model...")
            self._sdxl_base = DiffusionPipeline.from_pretrained(
                config.SDXL_BASE_MODEL, 
                torch_dtype=torch.float16 if self.use_half_precision else torch.float32,
                variant="fp16" if self.use_half_precision else None,
                use_safetensors=True
            )
        
        
        return self._sdxl_base.to(self.device)
    
    def get_sdxl_refiner(self):
        
        self.free_memory(keep_models=['refiner'])
        
        if self._sdxl_refiner is None:
            
            base = self.get_sdxl_base()
            print("Loading SDXL Refiner model...")
            self._sdxl_refiner = DiffusionPipeline.from_pretrained(
                config.SDXL_REF_MODEL,
                text_encoder_2=base.text_encoder_2,
                vae=base.vae,
                torch_dtype=torch.float16 if self.use_half_precision else torch.float32,
                use_safetensors=True,
                variant="fp16" if self.use_half_precision else None,
            )
            
            self.free_memory(keep_models=['refiner'])
        
        
        return self._sdxl_refiner.to(self.device)
        
    def get_blip_model_and_processor(self):
        
        self.free_memory(keep_models=['blip'])
        
        if self._blip_model is None:
            print("Loading BLIP-2 model...")
            self._blip_model = Blip2Model.from_pretrained(
                config.BLIP2_MODEL,
                torch_dtype=torch.float16 if self.use_half_precision else torch.float32,
                device_map="auto", 
                low_cpu_mem_usage=True  
            )

            self._blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        
        
        return self._blip_model, self._blip_processor
        
    def get_clip_model_and_processor(self):
        
        self.free_memory(keep_models=['clip'])
        
        if self._clip_model is None:
            print("Loading CLIP model...")
            self._clip_model = CLIPModel.from_pretrained(
                config.CLIP_MODEL,
                torch_dtype=torch.float16 if self.use_half_precision else torch.float32
            )
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        
        return self._clip_model.to(self.device), self._clip_processor
    
    def calculate_coherence_score(self, image, prompt_text):
        
        model, processor = self.get_clip_model_and_processor()
        
        inputs = processor(
            text=[prompt_text], 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            coherence_score = outputs.logits_per_image[0][0].item()
        
        return coherence_score
    
    def extract_image_embeddings(self, image):
        
        model, processor = self.get_blip_model_and_processor()
        
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)
        
        return embedding
    
    def clear_cache(self):
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def free_memory(self, keep_models=None):
        
        
        keep_models = keep_models or []

        
        if 'refiner' in keep_models and 'base' not in keep_models:
            keep_models.append('base')

        
        if 'base' not in keep_models and self._sdxl_base is not None:
            print("Entlade SDXL Base-Modell...")
            del self._sdxl_base
            self._sdxl_base = None

        
        if 'refiner' not in keep_models and self._sdxl_refiner is not None:
            print("Entlade SDXL Refiner-Modell...")
            del self._sdxl_refiner
            self._sdxl_refiner = None

        
        if 'blip' not in keep_models and self._blip_model is not None:
            print("Entlade BLIP-Modell...")
            del self._blip_model
            del self._blip_processor
            self._blip_model = None
            self._blip_processor = None

        
        if 'clip' not in keep_models and self._clip_model is not None:
            print("Entlade CLIP-Modell...")
            del self._clip_model
            del self._clip_processor
            self._clip_model = None
            self._clip_processor = None

        
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        
        