import opensim as osim
from geneticSolver import geneticSolver
from objects import Jumper, cmjJumper, Flipper
from cmaSolver import cmaSolver

osim.Logger.setLevelString("Error")

def termination_function(model,state):
    pelvis_ty = model.getCoordinateSet().get('pelvis_ty').getValue(state)
    return pelvis_ty < 0.25

if __name__ == "__main__":
    POPULATION_SIZE = 200
    GENERATION_COUNT = 1000
    MUTATION_RATE = 0.02
    INTEGRATION_DURATION = 2
    OVERLAP = 5  # overlap between generations (select the best X directly for the next generation)
    RANDOM = 0  # number of random individuals to introduce each generation
    N_WORKERS = 8  # None = auto-detect (uses cpu_count - 1), or set to specific number
    model_path = './models/H0918v3_web_squat_5x.osim'
    pop_object = Jumper  # or Jumper
    # initial_pop = pop_object.generate_population_from_json(POPULATION_SIZE, folder='./initial_population/') # You can provide an initial population here if desired if None it will be generated randomly
    initial_pop = None
    genetic_solver = geneticSolver(
        model_path=model_path,
        population_object=pop_object,
        initial_population=initial_pop,
        population_size=POPULATION_SIZE,
        generation_count=GENERATION_COUNT,
        mutation_rate=MUTATION_RATE,
        overlap=OVERLAP,
        random=RANDOM,
        initial_time=0.0,
        final_time=INTEGRATION_DURATION,
        n_workers=N_WORKERS,
        termination_function=termination_function
    )

    cma_solver = cmaSolver(
        model_path=model_path,
        population_object=pop_object,
        population_size=POPULATION_SIZE,
        generation_count=GENERATION_COUNT,
        initial_time=0.0,
        final_time=INTEGRATION_DURATION,
        n_workers=N_WORKERS,
        termination_function=termination_function,
        sigma0=0.3
    )

    solver = genetic_solver
    best_solution = solver.solve_parallel()

    if best_solution:
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Best Jump Height: {best_solution.fitness:.4f} m")
        print(f"Best Individual: {best_solution.name}")
