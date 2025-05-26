# mutators.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, Tuple
import torch
from custom_types import Latents, Argument


class Mutator( Generic[Argument], ABC):

    @abstractmethod
    def mutate(self, input : Argument) -> Argument:
        raise NotImplementedError("Method is not implementet yet")
    

def uniform_gaussian_mutate(tensor: Latents,
                            mutation_rate: float = 0.05,
                            mutation_strengh: float = 0.1,
                            clamp_range: Tuple[float,float] = (-1,1)
                            ) -> Latents:
    device = tensor.device
    c_tensor = tensor.clone()

    # define the number of element to mutate in the tensor 
    n = int(torch.numel(c_tensor * mutation_rate))
    # randomly selects the element positon from the tensor 
    mutation_position = torch.randperm(torch.numel(c_tensor), device=device)[:n]

    # Selects n elements from a standard normal distribution (Gausian Noise, mean=0, std=1)
    # The mutation stregth dcontrols how far values can deviate from zero
    mutations = torch.randn(n, device=device) * mutation_strengh


    flat_tensor = c_tensor.flatten()
    flat_tensor[mutation_position] += mutations

    mutated_tensor = flat_tensor.view(c_tensor.shape)
    
    clamp_min, clamp_max = clamp_range
    mutated_tensor = torch.clamp(mutated_tensor, min = clamp_min, max = clamp_max)

    return mutated_tensor

@dataclass
class UniformGausianMutator(Mutator[Latents]):
    mutation_rate : float
    mutation_strengh: float
    clamp_range: Tuple[float, float]
    
    def mutate(self, input : Latents ) -> Latents:

        return uniform_gaussian_mutate(
            input,
            mutation_rate=self.mutation_rate,
            mutation_strengh= self.mutation_strengh,
            clamp_range= self.clamp_range
        )
