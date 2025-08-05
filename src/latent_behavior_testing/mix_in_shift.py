from optimizations.crossover import ArithmeticCrossover, UniformCrossover
from optimizations.evaluators import MaxMeanDivergenceEvaluator
from models import ModelManager
from typing import List
from diffusers import StableDiffusionXLPipeline
import torch

crossover = ArithmeticCrossover()
eva = MaxMeanDivergenceEvaluator("")
manager = ModelManager()
device = manager.device
sdxl = manager.load_sdxl_base()


# Setup

prompt = "a cat"
seed_1 =4242
seed_2 =8888
path = ""



latents_shape = (
    1,
    4,
    1024 // 8, 
    1024 // 8
)

generator_1 = torch.Generator(device=device).manual_seed(seed_1)
generator_2 = torch.Generator(device=device).manual_seed(seed_2)
latents_1 = torch.randn(latents_shape, generator=generator_1, device=device, dtype=torch.float16)
latents_2 = torch.randn(latents_shape, generator=generator_2, device=device, dtype=torch.float16)

parent_img_1 = sdxl(
                    prompt=prompt,
                    latents=latents_1,
                    output_type="pil",
                    num_inference_steps=50,
                    guidance_scale =7.5,
                    ).images[0]
parent_img_1.save(path)

parent_img_2 = sdxl(
                    prompt=prompt,
                    latents=latents_2,
                    output_type="pil",
                    num_inference_steps=50,
                    guidance_scale =7.5,
                    ).images[0]
parent_img_2.save(path)





rates : List[float]= [i * 0.1 for i in range(1,10)]

for rate in rates :



