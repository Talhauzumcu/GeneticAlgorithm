import json
from simulator import Simulator
from modelController import modelController
import os
import sys
import json
from multiprocessing import Pool, cpu_count
from time import perf_counter
import numpy as np

class geneticSolver:

    def __init__(self, 
                 model_path, 
                 population_object, 
                 population_size, 
                 generation_count, 
                 mutation_rate, 
                 overlap,
                 random,
                 initial_time=0.0, 
                 final_time=2.0,
                 n_workers=None,
                 termination_function=None):
        
        self.model_path = model_path
        self.population_object = population_object
        self.population_size = population_size
        self.generation_count = generation_count
        self.mutation_rate = mutation_rate
        self.overlap = overlap
        self.random = random
        self.initial_time = initial_time
        self.final_time = final_time
        self.best_fitness = 0
        self.n_workers = n_workers if n_workers is not None else max(1, cpu_count() - 1)
        self.overall_best = None
        self.termination_function = termination_function

    def generate_population(self):
        return [self.population_object.generate_member() for _ in range(self.population_size)]
    
    def crossover_population(self, current_population, gen_counter):
        return self.population_object.crossover(self.population_size, current_population, self.mutation_rate, self.overlap, self.random, gen_counter)
    
    def _evaluate_member(self,args):
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
            return (None, member_count, 0.0)

    
    def solve_parallel(self):
        current_population = self.generate_population()
        for gen_counter in range(self.generation_count):
            gen_start_time = perf_counter()
            if gen_counter > 0:
                current_population = self.crossover_population(current_population, gen_counter)
            eval_args = [
                (current_population[i], self.model_path, self.initial_time, 
                 self.final_time, gen_counter, i, self.termination_function)
                for i in range(self.population_size)
            ]

            with Pool(processes=self.n_workers) as pool:
                results = pool.map(self._evaluate_member, eval_args)
            
            gen_end_time = perf_counter()
            for json_data, member_count, fitness in results:
                current_population[member_count].fitness = fitness
                current_population[member_count].json_data = json_data

            gen_best = max(current_population, key=lambda ind: ind.fitness)
            gen_best.export_json('./GeneticSolverJsonResults/') #Save the best individual's JSON data for web server
            gen_mean = np.mean([ind.fitness for ind in current_population])
            gen_std = np.std([ind.fitness for ind in current_population])

            print(f"Generation {gen_counter} completed in {gen_end_time - gen_start_time:.2f}s")
            print(f"Best Name: {gen_best.name} | Fitness: {gen_best.fitness:.4f} | Mean: {gen_mean:.4f} | Std: {gen_std:.4f}")
            if gen_best.fitness > self.best_fitness:
                print(f"  *** New best fitness found: {gen_best.fitness:.4f} (previous: {self.best_fitness:.4f}) ***")
                self.best_fitness = gen_best.fitness
                self.overall_best = gen_best
                

            if self.overall_best:
                print(f"  Overall Best: {self.overall_best.name} with fitness {self.overall_best.fitness:.4f}")
            print("-" * 60)
            
        print(f"\nOptimization complete!")
        print(f"Best fitness achieved: {self.best_fitness:.4f}")
        if self.overall_best:
            print(f"Best individual: {self.overall_best.name}")
        
        return self.overall_best
    
    # DEPRECATED
    def solve(self):
        """Run the genetic algorithm sequentially (original implementation)."""
        current_population = self.generate_population()
        for gen_counter in range(self.generation_count):
            if gen_counter > 0:
                current_population = self.crossover_population(current_population, gen_counter)
            
            gen_start_time = perf_counter()
            for member_count in range(self.population_size):
                pop_object = current_population[member_count]
                muscle_excitations = pop_object.genom_to_excitations()
                model_controller_instance = modelController(self.model_path)
                model_controller_instance.add_kinematics_analysis()
                model_controller_instance.add_states_reporter()
                model_controller_instance.setup_muscle_controller(muscle_excitations)
                model_controller_instance.initialize()
                try:
                    Simulator.run_simulation(model_controller_instance, 
                                             self.initial_time, 
                                             self.final_time, 
                                             report_interval=0.01,
                                            termination_function=self.termination_function)
                    
                    results_path, positions_filename, states_filename = model_controller_instance.save_results(
                    output_file_prefix=f'{pop_object.PREFIX}_gen{gen_counter}_{member_count}', 
                    results_dir='./results/')
                    positions_csv_path = os.path.join(results_path, positions_filename)
                    json_data = model_controller_instance.convert_kinematics_to_json(positions_csv_path)
                    pop_object.fitness = max(json_data['data']['center_of_mass_Y'])
                except Exception as e:
                    print(f'Error in simulation for {pop_object.PREFIX} Error: {e}')
                    continue
            
            gen_end_time = perf_counter()
            gen_best = max(current_population, key=lambda ind: ind.fitness)
            print(f"Generation {gen_counter} completed. Best fitness: {gen_best.fitness} it took {gen_end_time - gen_start_time:.2f} seconds")
            if gen_best.fitness > self.best_fitness:
                print(f"New best fitness found {gen_best.fitness}, previous best was {self.best_fitness}")
                self.best_fitness = gen_best.fitness
                # gen_best.export_genom()

        return self.overall_best