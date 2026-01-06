import numpy as np
import slimevolleygym  # registers env
import gym
import multiprocessing as mp

from training_scripts.train_neat import NEATPopulation, evaluate_genome, play_genome

WORKER_ENV = None

def _init_worker():
    global WORKER_ENV
    WORKER_ENV = gym.make("SlimeVolley-v0")

def _eval_one(args):
    genome, n_episodes, seed = args
    return evaluate_genome(genome, WORKER_ENV, n_episodes=n_episodes, seed=seed, render=False, verbose=False)

def main():
    pop = NEATPopulation(pop_size=100, input_dim=12, output_dim=3)

    generations = 200
    target_score = 8.0
    n_workers = 8

    best_genome = None
    best_fitness = -np.inf

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=n_workers, initializer=_init_worker)

    try:
        for g in range(generations):
            jobs = [(gen, 3, g) for gen in pop.genomes]
            fitness = np.array(pool.map(_eval_one, jobs), dtype=np.float32)

            idx = int(np.argmax(fitness))
            if float(fitness[idx]) > best_fitness:
                best_fitness = float(fitness[idx])
                best_genome = pop.genomes[idx].clone()
                print(f"Gen {g}: new best fitness = {best_fitness:.3f}")

            if best_fitness >= target_score:
                print(f"\nv Target score {target_score} reached at generation {g}. Stopping training and playing...")
                pool.terminate()
                pool.join()
                pool = None
                break

            pop.evolve_from_fitness(fitness)

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if best_genome is not None:
        play_genome(best_genome, seed=0, fps=50)

if __name__ == "__main__":
    main()
