import opensim as osim
from time import perf_counter
import numpy as np
from modelController import modelController

class Simulator:

    @staticmethod
    def run_simulation(model_controller: modelController,
                       initial_time: float, final_time: float,
                       output_file_name: str = "simulation",
                       results_dir: str = None,
                       report_interval: float = 0.01,
                       termination_function=None):

        model = model_controller.get_model()
        state = model_controller.get_state()
        state.setTime(initial_time) 
        manager = osim.Manager(model)
        manager.initialize(state)
        current_time = 0.0
        try:
            while current_time < final_time:
                current_state = manager.integrate(min(current_time + report_interval, final_time))
                current_time = current_state.getTime()  
                if termination_function is not None and termination_function(model, current_state):
                    break
        except Exception as e:
            print(f'Error in simulation for  Error: {e}')

        model_controller.set_manager(manager)