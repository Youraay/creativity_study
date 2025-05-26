from custom_types import Device, Noise, Argument
from typing import List, Generic, TypeVar
from evaluators import Evaluator
class GeneticOptimizaion():

    def __init__ (self
                  generations: int, 
                  population_size: int,
                  selector, 
                  evaluators: List[Evaluator[Noise]],
                  mutator, 
                  crossover_function,
                  base_population: List[Noise], 
                  initial_mutation_rate: float = 0.1,
                  crossover_rate: float = 0.1,
                  elitism_count: int = 0,
                  stric_osga: bool = False, 
                  
                  ):
        pass

    def initialize_population(self):

        pass

    def perform_generation(self):
        pass

    def calculate_fitness(self, noise: Noise) -> float:
        pass
    
   
    