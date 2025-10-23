import opensim as osim
import numpy as np
from simulator import Simulator
from geneticSolver import geneticSolver
from objects import Jumper, cmjJumper, Flipper
from cmaSolver import cmaSolver
import os
import sys
from modelController import modelController
osim.Logger.setLevelString("Error")


if __name__ == "__main__":
    solution_files = [f for f in os.listdir('./GeneticSolverJsonResults/') if f.endswith('.json')]
    POPULATION_SIZE = len(solution_files)
    INTEGRATION_DURATION = 2
    model_path = './models/H0918v3_web_backflip_20x.osim'
    pop_object = Flipper  
    solution_pop = pop_object.generate_population_from_json(POPULATION_SIZE, folder='./GeneticSolverJsonResults/')
    simulator = Simulator()
    for i, (member, json_file) in enumerate(zip(solution_pop, solution_files)):
        muscle_excitations = member.genom_to_excitations()
        model_controller_instance = modelController(model_path)
        model_controller_instance.add_kinematics_analysis()
        model_controller_instance.add_states_reporter()
        model_controller_instance.setup_muscle_controller(muscle_excitations)
        model_controller_instance.initialize()
        simulator.run_simulation(
            model_controller=model_controller_instance,
            initial_time=0.0,
            final_time=INTEGRATION_DURATION,
            termination_function=None
        )
        results_path, positions_filename, states_filename = model_controller_instance.save_results(
                output_file_prefix=f'{member.name}', 
                results_dir='./results/'
            )
        
        member.name = os.path.splitext(json_file)[0]
        json_data = model_controller_instance.prepare_json(results_path, positions_filename, states_filename)
        member.json_data = json_data
        member.export_json(f'./resimulatedJsonResults/')
        print(f'Resimulated and saved results for {member.name}')