import torch
from src.optimizations.mutators import uniform_gaussian_mutate

def test_no_mutation_returns_the_same_tensor():
    t = torch.ones((5,5))
    out = uniform_gaussian_mutate(t, mutation_rate=0.0)
    assert torch.equal(out, t)




