# selector.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, List, Dict, Any
from custom_types import Argument, Result, Fitness
import random

@dataclass
class SolutionCandidate(Generic[Argument, Result]):

    arguments : Argument
    results : Result
    fitness : Fitness
    meta : Dict = field(default_factory=Dict)

class Selector(Generic[Argument], ABC):

    @abstractmethod
    def select(self, candidates: List[SolutionCandidate[Any,Any]]) -> SolutionCandidate[Any,Any]:

        raise NotImplementedError("Method is not implementet yet")

class RankSelector(Selector[float]):

    def __init__(self, selection_pressure : float =1.5) -> None:
        
        assert selection_pressure > 1.0, "Selection Pressure must be greater than 1."

        self.selection_pressure = selection_pressure

    def select(self, candidates: List[SolutionCandidate]) -> SolutionCandidate:
            
            candidates.sort(key=lambda candidates: candidates.fitness, reverse=True)
            num_candicates = len(candidates)
            if num_candicates == 1:
                 return candidates[0]
            
            weights = [(self.selection_pressure -2.0 * (self.selection_pressure - 1.0) * (i/ num_candicates -1)) for i in range(num_candicates)]
            index = random.choices(range(num_candicates), weights=weights, k=1)[0]

            return candidates[index]
    
class TournamentSelector(Selector[float]):
     
    def __init__(self,  tournament_size : int) -> None:
        
        assert tournament_size > 1.0, "Selection Pressure must be greater than 1."

        self.tournament_size = tournament_size

    def select(self, candidates: List[SolutionCandidate]) -> SolutionCandidate:
        selected = random.sample(candidates, self.tournament_size)
        return max(selected, key=lambda candidate: candidate.fitness)
    
class RouletteWheelSelector(Selector[float]):
    
    def select(self, candidates: List[SolutionCandidate]) -> SolutionCandidate:
        total_fitness = sum(candidate.fitness for candidate in candidates if candidate.fitness)
        pick = random.uniform(0,total_fitness)
        current = 0
        for candidate in candidates:
            if candidate.fitness:
                current += candidate.fitness
                if current >= pick: 
                    return candidate
        return random.choice(candidates)