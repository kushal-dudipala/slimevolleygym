from training_scripts.train_neat import NEATPopulation, evaluate_genome, play_genome
import copy 
import numpy as np
import slimevolleygym

# 1. create population
pop = NEATPopulation(
    pop_size=100,
    input_dim=12,
    output_dim=3,
)

generations = 200

best_genome = None
best_fitness = -np.inf

target_score = 15.0

for g in range(generations):
    fitness = np.array([
        evaluate_genome(gen, n_episodes=3, seed=g, render=False, verbose=False)
        for gen in pop.genomes
    ])

    idx = int(np.argmax(fitness))
    if fitness[idx] > best_fitness:
        best_fitness = float(fitness[idx])
        best_genome = copy.deepcopy(pop.genomes[idx])
        print(f"Gen {g}: new best fitness = {best_fitness:.3f}")

    # STOP CONDITION
    if best_fitness >= target_score:
        print(f"\n✅ Target score {target_score} reached at generation {g}")
        play_genome(best_genome)
        break  # stop training forever

    pop.evolve(
        fitness_fn=lambda gen: evaluate_genome(gen, n_episodes=2, seed=g, render=False, verbose=False),
        generations=1
    )
