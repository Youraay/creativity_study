# types.py
from typing import TypeVar, Optional, List
import PIL.Image
import torch
from dataclasses import dataclass, field
import PIL
import random
type Device = str
type Latents = torch.Tensor
Argument = TypeVar('Argument')
Argument2 = TypeVar('Argument2')
Result = TypeVar('Result')
type Fitness = float
type Evaluation = float

@dataclass()
class Noise():
    prompt: str
    noise_embeddings: torch.Tensor
    generator: Optional[torch.Generator] 
    first_appearance: int
    last_appearance: int
    id: int = random.randint(1,2**32)
    blip_embeddings: Optional[torch.Tensor] = None
    clip_embeddings: Optional[torch.Tensor] = None
    image_embs: Optional[torch.Tensor] = None
    pil: Optional[PIL.Image.Image] = None 
    fitness: float = 0
    scores : dict[str,float] = field(default_factory=dict)
    seed: int = 0
    

    @classmethod
    def from_seed(cls, 
                  seed: int,
                  prompt= str,
                  id: int,
                  generation: int,
                  batch_size : int = 1,
                  num_channels_latents  : int =4 ,
                  vae_scale_factor : int = 8,
                  init_noise_sigma : float = 1.0,
                  height: int = 1024,
                  width: int = 1024,
                  device : Device = 'cuda',
                  dtype = torch.float16):
        
        assert seed > 0, "Seed must be grater than 0"
        generator = torch.Generator(device).manual_seed(seed)

        latents_shape = (
            batch_size,
            num_channels_latents,
            height // vae_scale_factor, 
            width // vae_scale_factor
        )

        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(latents_shape, generator=generator, device=device, dtype=dtype)
        # latents = latents * init_noise_sigma
        
        return cls(noise_embeddings=latents, prompt=prompt id=id, seed=seed, first_appearance = generation, last_appearance= generation, generator=generator)
    
