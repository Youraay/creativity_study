from src.optimizations.genetic_optimization import GeneticOptimization
from src.models.manager import ModelManager
from src.optimizations.selector import TournamentSelector
from src.optimizations.evaluators import MaxMeanDivergenceEvaluator, MaxPromptCoherenceEvaluator
from src.optimizations.mutators import UniformGausianMutator
from src.optimizations.crossover import ArithmeticCrossover

prompt="courage"

mm = ModelManager()
sdxl_base = mm.load_sdxl_turbo()
selector = TournamentSelector(1)
# e1 = MaxMeanDivergenceEvaluator(prompt)
e2 = MaxPromptCoherenceEvaluator(prompt)
# evalutators = [e1,e2]
evalutators = [e2]
# evaluation_weights = [0.8,0.2]
evaluation_weights = [1]
mutator = UniformGausianMutator(0.1, 0.2,(-1,1))
crossover = ArithmeticCrossover()
go = GeneticOptimization(
    generations=2,
    population_size=10,
    prompt=prompt,
    image_pipeline=sdxl_base,
    selector=selector,
    evaluators=evalutators,
    evaluation_weights=evaluation_weights,
    mutator=mutator,
    crossover_function=crossover
)

go.run()