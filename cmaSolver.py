import json
from simulator import Simulator
from modelController import modelController
import os
import sys
import json
from multiprocessing import Pool, cpu_count
from time import perf_counter
import numpy as np
import cma

class cmaSolver:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) solver.
    Similar interface to geneticSolver but uses CMA-ES optimization algorithm.
    """

    def __init__(self, 
                 model_path, 
                 population_object, 
                 population_size, 
                 generation_count, 
                 initial_time=0.0, 
                 final_time=2.0,
                 n_workers=None,
                 termination_function=None,
                 sigma0=0.3):
        """
        Initialize CMA-ES solver.
        
        Args:
            model_path: Path to the OpenSim model
            population_object: Class that defines the genome structure (e.g., Jumper)
            population_size: Population size (lambda in CMA-ES)
            generation_count: Number of generations to run
            initial_time: Simulation start time
            final_time: Simulation end time
            n_workers: Number of parallel workers (None = auto-detect)
            termination_function: Optional termination function for simulation
            sigma0: Initial standard deviation for CMA-ES (exploration vs exploitation)
        """
        
        self.model_path = model_path
        self.population_object = population_object
        self.population_size = population_size
        self.generation_count = generation_count
        self.initial_time = initial_time
        self.final_time = final_time
        self.best_fitness = -9999
        self.n_workers = n_workers if n_workers is not None else max(1, cpu_count() - 1)
        self.overall_best = None
        self.termination_function = termination_function
        self.sigma0 = sigma0
        
        # Initialize CMA-ES
        self.genom_length = population_object.GENOM_LENGTH
        self.genom_range = population_object.GENOM_RANGE
        
        # Initial mean in the middle of the range
        initial_mean = [(self.genom_range[0] + self.genom_range[1]) / 2] * self.genom_length
        
        # CMA-ES options
        opts = {
            'popsize': population_size,
            'bounds': [self.genom_range[0], self.genom_range[1]],
            'seed': np.random.randint(0, 10000),
            'verbose': -1  # Suppress CMA-ES output
        }
        
        self.es = cma.CMAEvolutionStrategy(initial_mean, self.sigma0, opts)
        
    def generate_population(self,gen_counter):
        """Generate population from CMA-ES."""
        genoms = self.es.ask()
        population = []
        for i, genom in enumerate(genoms):
            # Clip to ensure bounds are respected
            genom = np.clip(genom, self.genom_range[0], self.genom_range[1])
            name = f"{self.population_object.PREFIX}_gen_{gen_counter}_{i}"
            member = self.population_object(genom.tolist(), name=name)
            population.append(member)
        return population
    
    def update_cma(self, current_population):
        """Update CMA-ES with fitness values (CMA-ES minimizes, so we negate fitnesss)."""
        genoms = [np.array(member.get_genom()) for member in current_population]
        # CMA-ES minimizes, so negate fitness (we want to maximize jump height)
        fitness_values = [-member.fitness for member in current_population]
        self.es.tell(genoms, fitness_values)
    
    @staticmethod
    def _evaluate_member(args):
        """Static method to evaluate a single population member.
        
        This method runs in a separate process and simulates one individual.
        
        Args:
            args: Tuple containing (pop_object, model_path, initial_time, 
                  final_time, gen_counter, member_count, termination_function)
        
        Returns:
            Tuple of (json_data, member_count, fitness) where fitness is the fitness value
        """
        pop_object, model_path, initial_time, final_time, gen_counter, member_count, termination_function = args
        
        try:
            muscle_excitations = pop_object.genom_to_excitations()
            model_controller_instance = modelController(model_path)
            model_controller_instance.add_kinematics_analysis()
            model_controller_instance.add_states_reporter()
            model_controller_instance.setup_muscle_controller(muscle_excitations)
            model_controller_instance.initialize()
            
            Simulator.run_simulation(
                model_controller_instance, 
                initial_time, 
                final_time, 
                report_interval=0.01,
                termination_function=termination_function
            )
            
            results_path, positions_filename, states_filename = model_controller_instance.save_results(
                output_file_prefix=f'{pop_object.PREFIX}_gen{gen_counter}_{member_count}', 
                results_dir='./results/'
            )
            
            json_data = model_controller_instance.prepare_json(results_path, positions_filename, states_filename)
            fitness = pop_object.get_fitness(gen_counter, json_data)

            return (json_data, member_count, fitness)
            
        except Exception as e:
            print(f'Error in simulation for {pop_object.PREFIX}_gen{gen_counter}_{member_count}: {e}')
            return (member_count, 0.0)
    
    def solve_parallel(self):
        """Run the CMA-ES algorithm with parallel processing."""
        print(f"Starting CMA-ES optimization with population size {self.population_size}")
        print(f"Using {self.n_workers} workers for parallel processing")
        print(f"Genome length: {self.genom_length}, Initial sigma: {self.sigma0}")
        print("-" * 60)
        
        for gen_counter in range(self.generation_count):
            gen_start_time = perf_counter()
            
            # Generate population from CMA-ES
            current_population = self.generate_population(gen_counter)
            
            # Prepare arguments for parallel processing
            eval_args = [
                (current_population[i], self.model_path, self.initial_time, 
                 self.final_time, gen_counter, i, self.termination_function)
                for i in range(self.population_size)
            ]
            
            # Run simulations in parallel
            with Pool(processes=self.n_workers) as pool:
                results = pool.map(self._evaluate_member, eval_args)
            
            # Update population fitnesss
            for json_data,member_count, fitness in results:
                current_population[member_count].fitness = fitness
                current_population[member_count].json_data = json_data

            # Update CMA-ES with results
            self.update_cma(current_population)
            
            gen_end_time = perf_counter()
            
            # Report generation results
            gen_best = max(current_population, key=lambda ind: ind.fitness)
            gen_mean = np.mean([ind.fitness for ind in current_population])
            gen_std = np.std([ind.fitness for ind in current_population])
            gen_best.export_json('./CMAsolverJsonResults/') #Save the best individual's JSON data for web server
            
            print(f"Generation {gen_counter} completed in {gen_end_time - gen_start_time:.2f}s")
            print(f"  Best: {gen_best.fitness:.4f} | Mean: {gen_mean:.4f} | Std: {gen_std:.4f}")
            print(f"  Gen Best: {self.overall_best.name} with fitness {self.overall_best.fitness:.4f}")
            print(f"  CMA-ES sigma: {self.es.sigma:.4f}")
            
            
            if gen_best.fitness > self.best_fitness:
                print(f"  *** New best fitness found: {gen_best.fitness:.4f} (previous: {self.best_fitness:.4f}) ***")
                self.best_fitness = gen_best.fitness
                self.overall_best = gen_best

            if self.overall_best:
                print(f"  Overall Best: {self.overall_best.name} with fitness {self.overall_best.fitness:.4f}")
            
            print("-" * 60)
            
            # Check CMA-ES stopping criteria
            if self.es.stop():
                print("CMA-ES stopping criteria met:")
                print(self.es.stop())
                break
        
        print(f"\nOptimization complete!")
        print(f"Best fitness achieved: {self.best_fitness:.4f}")
        if self.overall_best:
            print(f"Best individual: {self.overall_best.name}")
        
        return self.overall_best
    
    def solve(self):
        """Run the CMA-ES algorithm sequentially (without parallel processing)."""
        print(f"Starting CMA-ES optimization with population size {self.population_size}")
        print(f"Sequential mode (no parallel processing)")
        print(f"Genome length: {self.genom_length}, Initial sigma: {self.sigma0}")
        print("-" * 60)
        
        for gen_counter in range(self.generation_count):
            gen_start_time = perf_counter()
            
            # Generate population from CMA-ES
            current_population = self.generate_population()
            
            # Evaluate each member sequentially
            for member_count in range(self.population_size):
                pop_object = current_population[member_count]
                
                try:
                    muscle_excitations = pop_object.genom_to_excitations()
                    model_controller_instance = modelController(self.model_path)
                    model_controller_instance.add_kinematics_analysis()
                    model_controller_instance.add_states_reporter()
                    model_controller_instance.setup_muscle_controller(muscle_excitations)
                    model_controller_instance.initialize()
                    
                    Simulator.run_simulation(
                        model_controller_instance, 
                        self.initial_time, 
                        self.final_time, 
                        report_interval=0.01,
                        termination_function=self.termination_function
                    )
                    
                    results_path, positions_filename, states_filename = model_controller_instance.save_results(
                        output_file_prefix=f'{pop_object.PREFIX}_gen{gen_counter}_{member_count}', 
                        results_dir='./results/'
                    )
                    
                    positions_csv_path = os.path.join(results_path, positions_filename)
                    json_data = model_controller_instance.convert_kinematics_to_json(positions_csv_path)
                    pop_object.fitness = max(json_data['data']['center_of_mass_Y']) - json_data['data']['center_of_mass_Y'][0]
                    
                except Exception as e:
                    print(f'Error in simulation for {pop_object.PREFIX} Error: {e}')
                    pop_object.fitness = 0.0
                    continue
            
            # Update CMA-ES with results
            self.update_cma(current_population)
            
            gen_end_time = perf_counter()
            
            # Report generation results
            gen_best = max(current_population, key=lambda ind: ind.fitness)
            gen_mean = np.mean([ind.fitness for ind in current_population])
            gen_std = np.std([ind.fitness for ind in current_population])
            
            print(f"Generation {gen_counter} completed in {gen_end_time - gen_start_time:.2f}s")
            print(f"  Best: {gen_best.fitness:.4f} | Mean: {gen_mean:.4f} | Std: {gen_std:.4f}")
            print(f"  CMA-ES sigma: {self.es.sigma:.4f}")
            
            if self.overall_best:
                print(f"  Overall Best: {self.overall_best.name} with fitness {self.overall_best.fitness:.4f}")
            
            if gen_best.fitness > self.best_fitness:
                print(f"  *** New best fitness found: {gen_best.fitness:.4f} (previous: {self.best_fitness:.4f}) ***")
                self.best_fitness = gen_best.fitness
                self.overall_best = gen_best
            
            print("-" * 60)
            
            # Check CMA-ES stopping criteria
            if self.es.stop():
                print("CMA-ES stopping criteria met:")
                print(self.es.stop())
                break
        
        print(f"\nOptimization complete!")
        print(f"Best fitness achieved: {self.best_fitness:.4f}")
        if self.overall_best:
            print(f"Best individual: {self.overall_best.name}")
        
        return self.overall_best
