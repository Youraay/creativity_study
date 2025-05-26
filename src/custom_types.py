# types.py
from typing import TypeVar, Optional
import PIL.Image
import torch
from dataclasses import dataclass
import PIL
type Device = str
type Latents = torch.Tensor
Argument = TypeVar('Argument')
Result = TypeVar('Result')
type Fitness = float
type Evaluation = float

@dataclass()
class Noise():

    latents: torch.Tensor
    image_embs: torch.Tensor = torch.zeros(1)
    pil: PIL.Image.Image = PIL.Image.Image()
    seed: int = 0
    generator: Optional[torch.Generator] = None

    @classmethod
    def from_seed(cls, 
                  seed: int,
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
        latents = latents * init_noise_sigma
        
        return cls(latents=latents, seed=seed, generator=generator)
    
