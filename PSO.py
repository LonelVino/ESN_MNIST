from tqdm import tqdm
import numpy as np

class Particle:
    def __init__(self, ini_params, w, c1, c2):
        self.position = ini_params           # particle position
        self.velocity = np.random.rand(len(ini_params))*2-1         # particle velocity
        self.pos_best = []          # best individual position 
        self.err_best = -1          # best individual error 
        self.err = -1               # individual error 
        self.w = w; self.c1 = c1; self.c2 = c2
    
    def evaluate(self, costFunc):
        self.err = costFunc(*self.position)
        if self.err < self.err_best or self.err_best == -1:
            self.pos_best = self.position
            self.err_best = self.err
            
    def update_velocity(self, pos_best_global):
        for i, pos_i in enumerate(self.position):
            r1 = np.random.random(); r2 = np.random.random()
            vel_cognitive = self.c1*r1*(self.pos_best[i]-pos_i)
            vel_social = self.c2*r2*(pos_best_global[i]-pos_i)
            self.velocity[i] = self.w*self.velocity[i] + vel_cognitive + vel_social
    
    def update_position(self, bounds):
        for i, vel_i in enumerate(self.velocity):
            self.position[i] = self.position[i] + vel_i
            self.position[i] = bounds[i][0] if self.position[i] < bounds[i][0] else self.position[i]
            self.position[i] = bounds[i][1] if self.position[i] > bounds[i][1] else self.position[i]
            
    
class PSO():
    def __init__(self, costFunc, ini_params, bounds, 
                 num_particles=10,  max_iter=10,
                 w=0.5, c1=0.3, c2=0.9, verbose=0):
        self.err_best_g = -1
        self.pos_best_g = np.array([])
        self.swarm = [Particle(ini_params, w, c1, c2) for _ in range(num_particles)]
        self.costFunc = costFunc
        self.params = ini_params
        self.bounds = bounds
        self.max_iter = max_iter
        self.verbose = verbose
        
    def fit(self):
        for i in tqdm(range(self.max_iter), disable=self.verbose):
            for particle in self.swarm:
                particle.evaluate(self.costFunc)
                if particle.err < self.err_best_g or self.err_best_g == -1:
                    self.pos_best_g = particle.position
                    self.err_best_g = float(particle.err)
            for particle in self.swarm:
                particle.update_velocity(self.pos_best_g)
                particle.update_position(self.bounds)
            if self.verbose == 0:
                print(f'[Iteration {i}] Error: {self.err_best_g}')
    
