# selector.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, List, Dict, Any
from ..custom_types import Argument, Result, Fitness, Noise
import random
import math
def _sigma_scale(sigma_k, sigma_S, fitnesses: List[float]) -> List[float]:
        mu   = sum(fitnesses) / len(fitnesses)
        sigma_square  = sum((f-mu)**2 for f in fitnesses) / len(fitnesses)
        sigma   = math.sqrt(sigma_square)

        if sigma == 0:                  
            return [1.0] * len(fitnesses)

        return [max(0.0, sigma_S + (f-mu)/(sigma_k*sigma))
                for f in fitnesses]

@dataclass
class SolutionCandidate(Generic[Argument, Result]):
    """
    s
    """

    arguments : Argument
    results : Result
    fitness : Fitness
    meta : Dict = field(default_factory=Dict)

class Selector(Generic[Argument], ABC):

    @abstractmethod
    def select(self, candidates: List[Noise]) -> Noise:

        raise NotImplementedError("Method is not implementet yet")

class RankSelector(Selector[float]):

    def __init__(self, selection_pressure : float =1.5) -> None:
        
        assert selection_pressure > 1.0, "Selection Pressure must be greater than 1."
        self.name: str = "RankSelector"
        self.selection_pressure = selection_pressure

    def select(self, candidates: List[Noise]) -> Noise:
            
            candidates.sort(key=lambda candidates: candidates.fitness, reverse=True)
            num_candicates = len(candidates)
            if num_candicates == 1:
                 return candidates[0]
            
            weights = [(self.selection_pressure -2.0 * (self.selection_pressure - 1.0) * (i/ num_candicates -1)) for i in range(num_candicates)]
            index = random.choices(range(num_candicates), weights=weights, k=1)[0]

            return candidates[index]
    
class NewRankSelector(Selector[float]):

    def __init__(self, selection_pressure : float =1.5) -> None:
        
        assert selection_pressure > 1.0, "Selection Pressure must be greater than 1."
        self.name: str = "NewRankSelector"
        self.selection_pressure = selection_pressure

    def select(self, candidates: List[Noise]) -> Noise:
        n = len(candidates)
        if n == 1:
            return candidates[0]

        # Nicht in-place sortieren
        ranked = sorted(candidates, key=lambda c: c.fitness, reverse=True)

    
        w = [2 - self.selection_pressure +
            2*(self.selection_pressure - 1)*i/(n - 1)
            for i in range(n)]

        chosen = random.choices(ranked, weights=w, k=1)[0]
        return chosen
    
class TournamentSelector(Selector[float]):
     
    def __init__(self,  tournament_size : int) -> None:
        
        
        self.name: str = "TournamentSelector"
        self.tournament_size = tournament_size

    def select(self, candidates: List[Noise]) -> Noise:

        assert 1 <= self.tournament_size <= len(candidates), "tournament_size must be in [1, pop_size]"
        selected = random.sample(candidates, self.tournament_size)
        return max(selected, key=lambda candidate: candidate.fitness)
    
class SigmaScaledRouletteSelector:
    """
    Roulette Wheel Selection + Sigma Scaling.
    Parameter:
        sigma_k   (float) – Faktor k in   f' = max(0, 1 + (f-μ)/(k·σ))
    """

    def __init__(self, sigma_k: float = 2.0, s_offset: float = 1.0) -> None:
        assert sigma_k > 0, "sigma_k must be > 0"
        self.k = sigma_k
        self.S = s_offset
        self.name = "SigmaScaledRouletteSelector"

    # ------------------------------------------------------------------
    def _sigma_scale(self, fitnesses: List[float]) -> List[float]:
        μ   = sum(fitnesses) / len(fitnesses)
        σ2  = sum((f-μ)**2 for f in fitnesses) / len(fitnesses)
        σ   = math.sqrt(σ2)

        if σ == 0:                       # alle identisch => jeder bekommt 1
            return [1.0] * len(fitnesses)

        return [max(0.0, self.S + (f-μ)/(self.k*σ))
                for f in fitnesses]

    # ------------------------------------------------------------------
    def select(self, candidates: List[Noise]) -> Noise:
        raw = [max(0.0, c.fitness) for c in candidates]   # clamp negatives
        scaled = self._sigma_scale(raw)
        total  = sum(scaled)

        # Total kann Null sein, falls ALLE clamp=0 → fallback
        if total == 0:
            return random.choice(candidates)

        pick = random.random() * total
        acc  = 0.0
        for cand, w in zip(candidates, scaled):
            acc += w
            if acc >= pick:
                return cand

        # Numerische Rundungsfalle: letzter Kandidat
        return candidates[-1]
class RouletteWheelSelector(Selector[float]):
    def __init__(self) -> None:
        self.name: str = "RouletteWheelSelector"

    def select(self, candidates: List[Noise]) -> Noise:
        total_fitness = sum(candidate.fitness for candidate in candidates if candidate.fitness)
        pick = random.uniform(0,total_fitness)
        current = 0
        for candidate in candidates:
            if candidate.fitness:
                current += candidate.fitness
                if current >= pick: 
                    return candidate
        return random.choice(candidates)