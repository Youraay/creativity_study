from src.optimizations.genetic_optimization import GeneticOptimization
from src.models.manager import ModelManager
from src.optimizations.selector import TournamentSelector, RankSelector, RouletteWheelSelector
from src.optimizations.evaluators import MaxMeanDivergenceEvaluator, MaxPromptCoherenceEvaluator
from src.optimizations.mutators import UniformGausianMutator
from src.optimizations.crossover import ArithmeticCrossover, UniformCrossover
import argparse
from visulisations import plotting

parser = argparse.ArgumentParser(description='Run genetic optimization with configurable components')
parser.add_argument('--selector', type=str, default='roulette', choices=['tournament', 'rank', 'roulette'], 
                    help='Type of selector to use')
parser.add_argument('--crossover', type=str, default='uniform', choices=['arithmetic', 'uniform'], 
                    help='Type of crossover to use')
parser.add_argument('--evaluators', type=str, default='both', choices=['divergence', 'coherence', 'both'], 
                    help='Evaluators to use')
parser.add_argument('--prompt', type=str, default='courage', help='Prompt to use')
args = parser.parse_args()
prompt = args.prompt

mm = ModelManager()
sdxl_base = mm.load_sdxl_base()
# selector = TournamentSelector(1)
# selector = RankSelector()
if args.selector == 'tournament':
    selector = TournamentSelector(1)
elif args.selector == 'rank':
    selector = RankSelector()
else:  # roulette
    selector = RouletteWheelSelector()

e1 = MaxMeanDivergenceEvaluator(prompt)
e2 = MaxPromptCoherenceEvaluator(prompt)
# Evaluators selection
e1 = MaxMeanDivergenceEvaluator(prompt)
e2 = MaxPromptCoherenceEvaluator(prompt)
if args.evaluators == 'divergence':
    evaluators = [e1]
    evaluation_weights = [1.0]
elif args.evaluators == 'coherence':
    evaluators = [e2]
    evaluation_weights = [1.0]
else:  # both
    evaluators = [e1, e2]
    evaluation_weights = [0.8, 0.2]
# evaluation_weights = [1]
mutator = UniformGausianMutator(0.05, 0.1,(-1,1))

if args.crossover == 'arithmetic':
    crossover = ArithmeticCrossover()
else:  # uniform
    crossover = UniformCrossover(0.5)

print(prompt)
print(type(selector))
print(type(mutator))
print(type(crossover))
for e in evaluators:
    print(type(e))
ts=datetime.now().strftime("%Y%m%d_%H%M%S")
safe_prompt = self.prompt.replace(" ", "_")
go = GeneticOptimization(
    generations=10,
    population_size=100,
    prompt=prompt,
    image_pipeline=sdxl_base,
    selector=selector,
    evaluators=evaluators,
    evaluation_weights=evaluation_weights,
    mutator=mutator,
    crossover_function=crossover, 
    timestemp = ts
)

best, generation_map = go.run()

plotting.save_noises_to_csv(generation_map, f"{safe_prompt}_{ts}.csv")
plotting.generate_umap(generation_map, f"{safe_prompt}_{ts}_u_map")
plotting.generate_pca(generation_map, f"{safe_prompt}_{ts}_pca")
plotting.generate_mean_and_standard_deviation(generation_map, f"{safe_prompt}_{ts}_m_and_sd")