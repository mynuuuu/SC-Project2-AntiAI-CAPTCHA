"""
Black-Box Optimizer for Sycophancy Attacker

Uses evolutionary/genetic algorithms to optimize behavioral parameters
purely from classifier feedback, without access to human data or code.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)


@dataclass
class OptimizedParams:
    """Optimized behavioral parameters"""
    vel_mean: float
    vel_std: float
    vel_max: float
    delay_mean: float
    delay_std: float
    idle_probability: float
    smoothness: float
    direction_change_prob: float
    micro_movement_prob: float
    hesitation_probability: float
    correction_probability: float
    min_events: int
    max_events: int
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for optimization"""
        return np.array([
            self.vel_mean,
            self.vel_std,
            self.vel_max,
            self.delay_mean,
            self.delay_std,
            self.idle_probability,
            self.smoothness,
            self.direction_change_prob,
            self.micro_movement_prob,
            self.hesitation_probability,
            self.correction_probability,
            float(self.min_events),
            float(self.max_events)
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'OptimizedParams':
        """Create from numpy array"""
        return cls(
            vel_mean=float(arr[0]),
            vel_std=float(arr[1]),
            vel_max=float(arr[2]),
            delay_mean=float(arr[3]),
            delay_std=float(arr[4]),
            idle_probability=float(np.clip(arr[5], 0, 1)),
            smoothness=float(np.clip(arr[6], 0, 1)),
            direction_change_prob=float(np.clip(arr[7], 0, 1)),
            micro_movement_prob=float(np.clip(arr[8], 0, 1)),
            hesitation_probability=float(np.clip(arr[9], 0, 1)),
            correction_probability=float(np.clip(arr[10], 0, 1)),
            min_events=int(max(20, arr[11])),
            max_events=int(max(50, arr[12]))
        )
    
    def clamp(self):
        """Clamp values to valid ranges"""
        self.vel_mean = np.clip(self.vel_mean, 30.0, 500.0)
        self.vel_std = np.clip(self.vel_std, 10.0, 200.0)
        self.vel_max = np.clip(self.vel_max, 200.0, 1000.0)
        self.delay_mean = np.clip(self.delay_mean, 5.0, 100.0)
        self.delay_std = np.clip(self.delay_std, 2.0, 50.0)
        self.idle_probability = np.clip(self.idle_probability, 0.0, 1.0)
        self.smoothness = np.clip(self.smoothness, 0.0, 1.0)
        self.direction_change_prob = np.clip(self.direction_change_prob, 0.0, 1.0)
        self.micro_movement_prob = np.clip(self.micro_movement_prob, 0.0, 1.0)
        self.hesitation_probability = np.clip(self.hesitation_probability, 0.0, 1.0)
        self.correction_probability = np.clip(self.correction_probability, 0.0, 1.0)
        self.min_events = max(20, min(self.min_events, 200))
        self.max_events = max(self.min_events + 50, min(self.max_events, 500))


class EvolutionaryOptimizer:
    """
    Evolutionary Strategy optimizer for black-box optimization
    Uses only classifier feedback, no human data
    """
    
    def __init__(self,
                 population_size: int = 20,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.7,
                 elite_size: int = 3):
        """
        Initialize evolutionary optimizer
        
        Args:
            population_size: Number of individuals in population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_size: Number of best individuals to keep
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        # Parameter bounds
        self.bounds = {
            'vel_mean': (30.0, 500.0),
            'vel_std': (10.0, 200.0),
            'vel_max': (200.0, 1000.0),
            'delay_mean': (5.0, 100.0),
            'delay_std': (2.0, 50.0),
            'idle_probability': (0.0, 1.0),
            'smoothness': (0.0, 1.0),
            'direction_change_prob': (0.0, 1.0),
            'micro_movement_prob': (0.0, 1.0),
            'hesitation_probability': (0.0, 1.0),
            'correction_probability': (0.0, 1.0),
            'min_events': (20, 200),
            'max_events': (50, 500)
        }
        
        # Initialize population
        self.population: List[Tuple[OptimizedParams, float]] = []
        self.generation = 0
        self.best_ever: Optional[Tuple[OptimizedParams, float]] = None
        
        logger.info(f"Initialized EvolutionaryOptimizer: pop_size={population_size}")
    
    def initialize_population(self):
        """Initialize population with random parameters"""
        self.population = []
        
        for _ in range(self.population_size):
            params = self._random_params()
            self.population.append((params, 0.0))
        
        logger.info(f"Initialized population of {self.population_size} individuals")
    
    def _random_params(self) -> OptimizedParams:
        """Generate random parameters within bounds"""
        return OptimizedParams(
            vel_mean=np.random.uniform(*self.bounds['vel_mean']),
            vel_std=np.random.uniform(*self.bounds['vel_std']),
            vel_max=np.random.uniform(*self.bounds['vel_max']),
            delay_mean=np.random.uniform(*self.bounds['delay_mean']),
            delay_std=np.random.uniform(*self.bounds['delay_std']),
            idle_probability=np.random.uniform(*self.bounds['idle_probability']),
            smoothness=np.random.uniform(*self.bounds['smoothness']),
            direction_change_prob=np.random.uniform(*self.bounds['direction_change_prob']),
            micro_movement_prob=np.random.uniform(*self.bounds['micro_movement_prob']),
            hesitation_probability=np.random.uniform(*self.bounds['hesitation_probability']),
            correction_probability=np.random.uniform(*self.bounds['correction_probability']),
            min_events=int(np.random.uniform(*self.bounds['min_events'])),
            max_events=int(np.random.uniform(*self.bounds['max_events']))
        )
    
    def evaluate_population(self, evaluate_fn: Callable[[OptimizedParams], float]):
        """
        Evaluate all individuals in population
        
        Args:
            evaluate_fn: Function that takes params and returns fitness (prob_human)
        """
        for i, (params, _) in enumerate(self.population):
            try:
                fitness = evaluate_fn(params)
                self.population[i] = (params, fitness)
                
                # Track best ever
                if self.best_ever is None or fitness > self.best_ever[1]:
                    self.best_ever = (params, fitness)
                    logger.info(f"New best fitness: {fitness:.3f} (generation {self.generation})")
            except Exception as e:
                logger.error(f"Error evaluating individual {i}: {e}")
                self.population[i] = (params, 0.0)
    
    def evolve(self):
        """Evolve population to next generation"""
        # Sort by fitness (descending)
        self.population.sort(key=lambda x: x[1], reverse=True)
        
        # Select elite
        elite = self.population[:self.elite_size]
        
        # Create new population
        new_population = elite.copy()  # Keep elite
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Selection (tournament selection)
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1
            
            # Mutation
            if np.random.random() < self.mutation_rate:
                child = self._mutate(child)
            
            child.clamp()
            new_population.append((child, 0.0))
        
        self.population = new_population
        self.generation += 1
        
        logger.info(f"Generation {self.generation}: Best fitness = {self.population[0][1]:.3f}")
    
    def _tournament_select(self, tournament_size: int = 3) -> OptimizedParams:
        """Tournament selection"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        tournament.sort(key=lambda x: x[1], reverse=True)
        return tournament[0][0]
    
    def _crossover(self, parent1: OptimizedParams, parent2: OptimizedParams) -> OptimizedParams:
        """Uniform crossover"""
        arr1 = parent1.to_array()
        arr2 = parent2.to_array()
        
        # Uniform crossover
        mask = np.random.random(len(arr1)) < 0.5
        child_arr = np.where(mask, arr1, arr2)
        
        return OptimizedParams.from_array(child_arr)
    
    def _mutate(self, individual: OptimizedParams, mutation_strength: float = 0.1) -> OptimizedParams:
        """Gaussian mutation"""
        arr = individual.to_array()
        
        # Adaptive mutation strength (smaller as we get better)
        if self.best_ever:
            adaptive_strength = mutation_strength * (1.0 - self.best_ever[1])
        else:
            adaptive_strength = mutation_strength
        
        # Mutate each parameter
        mutation = np.random.normal(0, adaptive_strength, size=arr.shape)
        
        # Scale by parameter ranges
        ranges = np.array([
            (self.bounds['vel_mean'][1] - self.bounds['vel_mean'][0]),
            (self.bounds['vel_std'][1] - self.bounds['vel_std'][0]),
            (self.bounds['vel_max'][1] - self.bounds['vel_max'][0]),
            (self.bounds['delay_mean'][1] - self.bounds['delay_mean'][0]),
            (self.bounds['delay_std'][1] - self.bounds['delay_std'][0]),
            1.0,  # idle_probability
            1.0,  # smoothness
            1.0,  # direction_change_prob
            1.0,  # micro_movement_prob
            1.0,  # hesitation_probability
            1.0,  # correction_probability
            float((self.bounds['min_events'][1] - self.bounds['min_events'][0])),
            float((self.bounds['max_events'][1] - self.bounds['max_events'][0]))
        ])
        
        mutation = mutation * ranges
        mutated_arr = arr + mutation
        
        return OptimizedParams.from_array(mutated_arr)
    
    def get_best(self) -> OptimizedParams:
        """Get best individual"""
        if not self.population:
            return self._random_params()
        
        self.population.sort(key=lambda x: x[1], reverse=True)
        return self.population[0][0]
    
    def get_best_ever(self) -> Optional[OptimizedParams]:
        """Get best individual ever found"""
        if self.best_ever:
            return self.best_ever[0]
        return None


class LocalRefiner:
    """
    Local refinement using gradient-free optimization
    Refines parameters around a good solution
    """
    
    def __init__(self, step_size: float = 0.05, max_iterations: int = 20):
        """
        Initialize local refiner
        
        Args:
            step_size: Initial step size for local search
            max_iterations: Maximum refinement iterations
        """
        self.step_size = step_size
        self.max_iterations = max_iterations
    
    def refine(self,
               initial_params: OptimizedParams,
               evaluate_fn: Callable[[OptimizedParams], float]) -> OptimizedParams:
        """
        Refine parameters locally
        
        Args:
            initial_params: Starting parameters
            evaluate_fn: Function to evaluate fitness
            
        Returns:
            Refined parameters
        """
        current_params = initial_params
        current_fitness = evaluate_fn(current_params)
        
        logger.info(f"Local refinement: Starting fitness = {current_fitness:.3f}")
        
        for iteration in range(self.max_iterations):
            improved = False
            
            # Try perturbations in each dimension
            for dim in range(13):  # 13 parameters
                # Try positive perturbation
                candidate = self._perturb(current_params, dim, self.step_size)
                candidate_fitness = evaluate_fn(candidate)
                
                if candidate_fitness > current_fitness:
                    current_params = candidate
                    current_fitness = candidate_fitness
                    improved = True
                    logger.debug(f"  Iteration {iteration}, dim {dim}: Improved to {current_fitness:.3f}")
                    break
                
                # Try negative perturbation
                candidate = self._perturb(current_params, dim, -self.step_size)
                candidate_fitness = evaluate_fn(candidate)
                
                if candidate_fitness > current_fitness:
                    current_params = candidate
                    current_fitness = candidate_fitness
                    improved = True
                    logger.debug(f"  Iteration {iteration}, dim {dim}: Improved to {current_fitness:.3f}")
                    break
            
            if not improved:
                # Reduce step size and try again
                self.step_size *= 0.8
                if self.step_size < 0.01:
                    break
        
        logger.info(f"Local refinement: Final fitness = {current_fitness:.3f}")
        return current_params
    
    def _perturb(self, params: OptimizedParams, dimension: int, amount: float) -> OptimizedParams:
        """Perturb one dimension of parameters"""
        arr = params.to_array()
        bounds = self._get_bounds()
        
        # Scale perturbation by parameter range
        ranges = np.array([
            bounds['vel_mean'][1] - bounds['vel_mean'][0],
            bounds['vel_std'][1] - bounds['vel_std'][0],
            bounds['vel_max'][1] - bounds['vel_max'][0],
            bounds['delay_mean'][1] - bounds['delay_mean'][0],
            bounds['delay_std'][1] - bounds['delay_std'][0],
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            float(bounds['min_events'][1] - bounds['min_events'][0]),
            float(bounds['max_events'][1] - bounds['max_events'][0])
        ])
        
        arr[dimension] += amount * ranges[dimension]
        
        result = OptimizedParams.from_array(arr)
        result.clamp()
        return result
    
    def _get_bounds(self):
        """Get parameter bounds"""
        return {
            'vel_mean': (30.0, 500.0),
            'vel_std': (10.0, 200.0),
            'vel_max': (200.0, 1000.0),
            'delay_mean': (5.0, 100.0),
            'delay_std': (2.0, 50.0),
            'idle_probability': (0.0, 1.0),
            'smoothness': (0.0, 1.0),
            'direction_change_prob': (0.0, 1.0),
            'micro_movement_prob': (0.0, 1.0),
            'hesitation_probability': (0.0, 1.0),
            'correction_probability': (0.0, 1.0),
            'min_events': (20, 200),
            'max_events': (50, 500)
        }

