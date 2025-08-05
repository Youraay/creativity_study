from src.optimizations.genetic_optimization import GeneticOptimization
from src.models.manager import ModelManager
from src.optimizations.selector import TournamentSelector, RankSelector, RouletteWheelSelector, NewRankSelector,SigmaScaledRouletteSelector
from src.optimizations.evaluators import MaxMeanDivergenceEvaluator, MaxPromptCoherenceEvaluator, MaxLocalMeanDivergenceEvaluator
from src.optimizations.mutators import UniformGausianMutator
from src.optimizations.crossover import ArithmeticCrossover, UniformCrossover, QuarteredCrossover, SlerpCrossover
import argparse
from datetime import datetime 
from src.visulisations import plotting
import random

parser = argparse.ArgumentParser(description='Run genetic optimization with configurable components')
parser.add_argument('--selector', type=str, default='roulette', choices=['tournament', 'rank','newrank', 'roulette', 'sigma'], 
                    help='Type of selector to use')
parser.add_argument('--crossover', type=str, default='uniform', choices=['arithmetic', 'uniform', 'quartered', 'slerp'], 
                    help='Type of crossover to use')
parser.add_argument('--evaluators', type=str, default='both', choices=['divergence', 'coherence', 'both', 'all'], 
                    help='Evaluators to use')
parser.add_argument('--prompt', type=str, default='courage', help='Prompt to use')
parser.add_argument('--model', type=str, default='sdxl', choices=['sdxl', 'sdxlt'], help='Prompt to use')
parser.add_argument('--guidance', type=float, default=None, 
                    help='Guidance scale for image generation (uses model default if not specified)')
args = parser.parse_args()
prompt = args.prompt

mm = ModelManager()

if args.model == 'sdxlt':
    model = mm.load_sdxl_turbo()
    num_steps = 1
    guidance = 0.0
else:
    model = mm.load_sdxl_base()    
    num_steps = 50
    guidance = 7.0
guidance = args.guidance if args.guidance is not None else guidance
if args.selector == 'tournament':
    selector = TournamentSelector(2) 
elif args.selector == 'rank':
    selector = RankSelector()
elif args.selector == 'newrank':
    selector = NewRankSelector()
elif args.selector == 'sigma':
    selector = SigmaScaledRouletteSelector()
else:  # roulette
    selector = RouletteWheelSelector()

e1 = MaxMeanDivergenceEvaluator(prompt)
e2 = MaxPromptCoherenceEvaluator(prompt)
e3 = MaxLocalMeanDivergenceEvaluator()

# Evaluators selection
evaluators = []
if args.evaluators == 'all':
    evaluators = [e1, e2, e3]
    evaluation_weights = [0.6, 0.2, 0.2]
elif args.evaluators == 'divergence':
    evaluators = [e1]
    evaluation_weights = [1.0]
elif args.evaluators == 'coherence':
    evaluators = [e2]
    evaluation_weights = [1.0]
else:  # both
    evaluators = [e1, e2]
    evaluation_weights = [0.8, 0.2] 

mutator = UniformGausianMutator(0.15, 0.2, (-1,1)) 

if args.crossover == 'arithmetic':
    crossover = ArithmeticCrossover(weight=0.8)
elif args.crossover == 'uniform':  # uniform
    crossover = UniformCrossover(0.5)
elif args.crossover == 'slerp':
    crossover = SlerpCrossover(0.5)
else:
    crossover = QuarteredCrossover() 

print(prompt)
print(type(selector))
print(type(mutator))
print(type(crossover))
for e in evaluators:
    print(type(e))
ts=datetime.now().strftime("%Y%m%d_%H%M%S")
safe_prompt = prompt.replace(" ", "_")

go = GeneticOptimization(
    generations=10,
    population_size=100,
    prompt=prompt,
    image_pipeline=model,
    selector=selector,
    evaluators=evaluators,
    evaluation_weights=evaluation_weights,
    mutator=mutator,
    crossover_function=crossover, 
    ts = ts,
    elitism_count=0,  
    strict_osga=False,
    crossover_rate=0.9, 
    initial_mutation_rate=0.05,
    random_seed=random.randint(1,2**32),
    num_steps = num_steps,
    guidance =guidance,
    sigma_scaling= False,
)

generation_map, path = go.run()
print("now saving csv")
plotting.save_noises_to_csv(generation_map, path=path, csv_name=f"population.csv")
print("csv should be saved")

plotting.generate_umap(
    generation_map, 
    f"{path}/plot_u_map")

plotting.generate_pca(
    generation_map,
    f"{path}/plot_pca")

plotting.generate_mean_and_standard_deviation(
    generation_map,
    f"{path}/plot_m_and_sd")

plotting.generate_best_fitness_curve(
    generation_map,
    f"{path}/plot_best_fitness")

plotting.generate_mutation_vs_crossover_boxplot(
    generation_map,

    f"{path}/plot_mut_vs_cross")

plotting.generate_evaluator_contributions(
    generation_map,
    f"{path}/plot_evaluator_stack")

plotting.generate_delta_to_first_generation(
    generation_map,
    f"{path}/plot_delta_to_gen0"
)

plotting.generate_diversity_metric(
    generation_map,
    f"{path}/plot_diversity")
