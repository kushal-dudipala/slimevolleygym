import numpy as np
import copy
import time
import gym

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
        
        self.order = np.argsort(self.layer)
        self.incoming_idx = [None] * self.nodes
        self.rebuild_incoming()

    def rebuild_incoming(self):
        # list of incoming indices for each node (only for non-input nodes)
        self.incoming_idx = []
        for j in range(self.nodes):
            if j < self.input_dim:
                self.incoming_idx.append(None)
            else:
                self.incoming_idx.append(np.where(self.enabled[:, j])[0])


    def forward(self, x):
        h = np.zeros(self.nodes, dtype=np.float32)
        h[:self.input_dim] = x
        h[self.input_dim-1] = 1.0  # bias node

        for j in self.order:
            if j < self.input_dim:
                continue
            inc = self.incoming_idx[j]
            if inc is None or inc.size == 0:
                continue
            z = (h[inc] * self.W[inc, j]).sum()
            h[j] = np.tanh(z)

        return h[self.input_dim:self.input_dim + self.output_dim]


    def mutate_weights(self, sigma=0.1) -> None:
        ii, jj = np.where(self.enabled)
        if len(ii) == 0:
            return
        self.W[ii, jj] += np.random.randn(len(ii)).astype(self.dtype) * sigma

            
    def mutate_add_connection(self, tries=50) -> None:
        for _ in range(tries):
            i = np.random.randint(0, self.nodes)
            j = np.random.randint(self.input_dim, self.nodes)

            if (i == j) or (j < self.input_dim) or (self.layer[i] >= self.layer[j]) or self.enabled[i, j]:
                continue

            self.enabled[i, j] = True
            self.W[i, j] = np.random.uniform(-1, 1)

            #  update cache for just this target node j
            self.incoming_idx[j] = np.where(self.enabled[:, j])[0].astype(np.int32)
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

        self.W = np.pad(self.W, ((0,1),(0,1))).astype(self.dtype, copy=False)
        self.enabled = np.pad(self.enabled, ((0,1),(0,1))).astype(bool, copy=False)

        new_layer = (self.layer[i] + self.layer[j]) / 2
        self.layer.append(new_layer)

        self.enabled[i, new_node] = True
        self.W[i, new_node] = 1.0

        self.enabled[new_node, j] = True
        self.W[new_node, j] = old_w

        self.order = np.argsort(self.layer).astype(np.int32)
        self.rebuild_incoming()
        return

    def clone(self):
        g = NEATGenome(self.input_dim, self.output_dim, dtype=self.dtype)
        g.nodes = self.nodes
        g.layer = list(self.layer)
        g.W = self.W.copy()
        g.enabled = self.enabled.copy()

        g.order = np.argsort(g.layer).astype(np.int32)
        g.rebuild_incoming()
        return g




class NEATPopulation:
    def __init__(self, pop_size, input_dim, output_dim):
        self.genomes = [NEATGenome(input_dim, output_dim, seed=i) for i in range(pop_size)]

    def evolve_from_fitness(self, fitness):
        fitness = np.asarray(fitness)
        elite_idx = np.argsort(fitness)[-len(self.genomes)//2:]
        elites = [self.genomes[i] for i in elite_idx]

        children = []
        for parent in elites:
            child = parent.clone()  # implement clone below (faster than deepcopy)
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

def evaluate_genome(genome, env, n_episodes=5, seed=0, render=False, verbose=False):
    total_reward = 0.0

    for ep in range(n_episodes):
        env.seed(seed + ep)
        

        obs = env.reset() 
        done = False
        ep_reward = 0.0

        while not done:
            action = policy(obs, genome)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward 
            if render:
                env.render()

        if verbose:
            print(f"Episode {ep}: score = {ep_reward}")
        total_reward += ep_reward

    return total_reward / n_episodes





def play_genome(genome, seed=0, fps=50):
    import time
    import gym
    import slimevolleygym  # noqa: F401

    env = gym.make("SlimeVolley-v0")
    env.seed(seed)

    # Force the GUI window to be created immediately (like the demo)
    env.render()

    print("Training complete — playing best genome (close window or Ctrl+C to stop)")

    try:
        while True:
            env.seed(seed)
            obs = env.reset()
            done = False
            ep_reward = 0.0

            


            while not done:
                action = policy(obs, genome)
                obs, reward, done, _ = env.step(action)

                ep_reward += reward
                if reward != 0:
                    print("POINT EVENT reward =", reward)  # +1 you scored, -1 opponent scored

                env.render()

                if fps is not None:
                    time.sleep(1.0 / fps)

            print(f"Episode score: {ep_reward:.2f}")
            seed += 1  # change seed each episode so you see variety
    finally:
        env.close()
