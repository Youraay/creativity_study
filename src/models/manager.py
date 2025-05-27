import torch
from diffusers import StableDiffusionXLPipeline
from transformers import Blip2Model, AutoProcessor, CLIPModel, CLIPProcessor



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
        self._sdxl_turbo = None
        self.use_half_precision = True

        

        self.n_steps = 50
        self.high_noise_frac = 0.8
        
        self._initialized = True

    def load_sdxl_base(self):

        if self._sdxl_base is None:
            
            self._sdxl_base = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16, 
                use_safetensors=True,
                variant="fp16"
            )
        print("sdxl loaded")
        self._sdxl_base.to(self.device)
        return self._sdxl_base

    def load_sdxl_refiner(self):

        if self._sdxl_refiner is None:
            
            base = self.load_sdxl_base()
            print("Loading SDXL Refiner model...")
            self._sdxl_refiner = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=base.text_encoder_2,
                vae=base.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            
        self._sdxl_refiner.to(self.device)
        return self._sdxl_refiner
    
    def load_sdxl_turbo(self):

        if self._sdxl_turbo is None:
            self._sdxl_turbo = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16, 
                use_safetensors=True,
                variant="fp16"
            )
        print("sdxl turbo loaded")
        self._sdxl_turbo.to(self.device)
        return self._sdxl_turbo

    def load_blip2(self):

        if self._blip_model is None:
            print("Loading BLIP-2 model...")
            self._blip_model = Blip2Model.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16 if self.use_half_precision else torch.float32,
                device_map="cpu", 
            )

            self._blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        
        
        return self._blip_model, self._blip_processor

    def load_clip(self):

        if self._clip_model is None:
            print("Loading CLIP model...")
            self._clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                torch_dtype=torch.float16 if self.use_half_precision else torch.float32
            )
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self._clip_model.to(self.device)
        print("Clip Model Loaded")
        return self._clip_model, self._clip_processor
