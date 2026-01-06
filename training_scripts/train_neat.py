import numpy as np
import copy

def seed_everything(seed):
    import random

    np.random.seed(seed)
    random.seed(seed)
   

seed_everything(0)
class NEATGenome:
    def __init__(
        self,
        input_dim: int = 12,
        output_dim: int = 3,
        seed: int = 0,
        dtype=np.float32,
    ):
        assert (
            isinstance(input_dim, int)
            and input_dim > 0
            and isinstance(output_dim, int)
            and output_dim > 0
        ), "input_dim and output_dim must be positive integers."

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dtype = dtype
        
        self.nodes = input_dim + output_dim

        # layer enforces feedforward DAG
        self.layer = [0]*input_dim + [1]*output_dim

        # weights + enable mask
        self.W = np.zeros((self.nodes, self.nodes), dtype=dtype)
        self.enabled = np.zeros_like(self.W, dtype=bool)

        # fully connect input to output
        for i in range(input_dim):
            for j in range(input_dim, self.nodes):
                self.enabled[i, j] = True
                self.W[i, j] = np.random.uniform(-1, 1) 

    def forward(self, x):
        h = np.zeros(self.nodes, dtype=np.float32)
        h[:self.input_dim] = x

        order = np.argsort(self.layer)
        for j in order:
            if j < self.input_dim:
                continue

            incoming = self.enabled[:, j]
            z = (h[incoming] * self.W[incoming, j]).sum()
            h[j] = np.tanh(z)  
        return h[self.input_dim:self.input_dim + self.output_dim]

    def mutate_weights(self, sigma=0.1) -> None:
        mask = self.enabled
        self.W[mask] += np.random.randn(*self.W.shape)[mask] * sigma
        
        return
        
    def mutate_add_connection(self, tries=50) -> None:
        for _ in range(tries):
            i = np.random.randint(0, self.nodes)
            j = np.random.randint(self.input_dim, self.nodes)
            
            if (i == j) or (j < self.input_dim) or (self.layer[i] >= self.layer[j]) or self.enabled[i, j]:
                continue

            self.enabled[i, j] = True
            self.W[i, j] = np.random.uniform(-1, 1)
            return
    
    def mutate_add_node(self) -> None:
        ii, jj = np.where(self.enabled)
        if len(ii) == 0:
            return

        k = np.random.randint(len(ii))
        i, j = int(ii[k]), int(jj[k])

        old_w = self.W[i, j]
        self.enabled[i, j] = False
        self.W[i, j] = 0.0

        new_node = self.nodes
        self.nodes += 1

        self.W = np.pad(self.W, ((0,1),(0,1)))
        self.enabled = np.pad(self.enabled, ((0,1),(0,1)))

        new_layer = (self.layer[i] + self.layer[j]) / 2
        self.layer.append(new_layer)

        self.enabled[i, new_node] = True
        self.W[i, new_node] = 1.0

        self.enabled[new_node, j] = True
        self.W[new_node, j] = old_w
        
        return



class NEATPopulation:
    def __init__(self, pop_size, input_dim, output_dim):
        self.genomes = [
            NEATGenome(input_dim, output_dim, seed=i)
            for i in range(pop_size)
        ]

    def evolve(self, fitness_fn, generations=100):
        for g in range(generations):
            fitness = np.array([fitness_fn(gen) for gen in self.genomes])

            elite_idx = np.argsort(fitness)[-len(self.genomes)//2:]
            elites = [self.genomes[i] for i in elite_idx]

            children = []
            for parent in elites:
                child = copy.deepcopy(parent)
                child.mutate_weights()
                if np.random.rand() < 0.3:
                    child.mutate_add_connection()
                if np.random.rand() < 0.1:
                    child.mutate_add_node()
                children.append(child)

            self.genomes = elites + children

def policy(obs, genome):
    y = genome.forward(obs)
    return [
        int(y[0] > 0.0),
        int(y[1] > 0.0),
        int(y[2] > 0.0),
    ]

import gym
import slimevolleygym
def evaluate_genome(genome, n_episodes=5, seed=0, render=False, verbose=True):
    env = gym.make("SlimeVolley-v0")
    env.seed(seed)

    total_reward = 0.0

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = policy(obs, genome)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward + 0.01

            if render:
                env.render()

        if verbose:
            print(f"Episode {ep}: score = {ep_reward}")

        total_reward += ep_reward

    env.close()
    return total_reward / n_episodes

import time

def play_genome(genome, seed=0):
    env = gym.make("SlimeVolley-v0")
    env.seed(seed)

    print("Training complete — playing best genome")

    while True:  # loop forever
        obs = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = policy(obs, genome)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            env.render()
            time.sleep(1/60)  # smooth rendering

        print(f"Episode score: {ep_reward:.2f}")
