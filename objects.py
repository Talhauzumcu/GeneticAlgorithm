import numpy as np
from utils import *
import os
import json

class Jumper:

    PREFIX = 'jumper'
    START_TIME = 0.0
    END_TIME = 1
    GENOM_RANGE = (0,1) # Min and max value for each gene 
    GENE_EXPRESSION_LENGTH = 20 # Number of time points for each muscle group
    MUSCLE_GROUPS = {'Quadriceps': ['vasti_l', 'vasti_r', 'rect_fem_l', 'rect_fem_r'],
                    'Hamstrings': ['hamstrings_l', 'hamstrings_r','bifemsh_l','bifemsh_r'],
                    'Glutes': ['glut_max_r', 'glut_max_l'],
                    'Calves': ['gastroc_r', 'gastroc_l', 'soleus_r', 'soleus_l'],
                    'Tibialis Anterior': ['tib_ant_r', 'tib_ant_l'],
                    'Trunk Flexors': ['intobl_r', 'intobl_l', 'extobl_r', 'extobl_l'],
                    'Trunk Extensors': ['ercspn_r', 'ercspn_l'],
                    'Hip Flexors': ['iliopsoas_r', 'iliopsoas_l']}
    
    GENOM_LENGTH = GENE_EXPRESSION_LENGTH * len(MUSCLE_GROUPS)

    def __init__(self, genom, name=None):
        self.genom = genom
        self.timing = np.linspace(self.START_TIME, self.END_TIME, self.GENE_EXPRESSION_LENGTH)
        self.fitness = 0
        self.name = name if name is not None else '0_0'

    @classmethod
    def generate_member(cls):
        genom = [np.random.uniform(cls.GENOM_RANGE[0], cls.GENOM_RANGE[1]) for _ in range(cls.GENOM_LENGTH)]
        return cls(genom)
    
    def genom_to_excitations(self):
        muscle_excitations = {}
        for i, group in enumerate(self.MUSCLE_GROUPS.keys()):
            muscle_excitations[group] = {'value': self.genom[i*self.GENE_EXPRESSION_LENGTH:(i+1)*self.GENE_EXPRESSION_LENGTH],
                                         'time': self.timing.tolist()}
        parsed_muscle_excitations = parse_muscle_excitations(muscle_excitations, self.MUSCLE_GROUPS)
        return parsed_muscle_excitations
    
    def get_genom(self):
        return self.genom

    def export_genom(self):
        with open(f'./Genoms/{self.name}_{self.score:.2f}.txt', 'w') as f:
            for param in self.genom:
                f.write(f'{param[0]}, {param[1]},')
    
    @classmethod
    def crossover(cls, population_count, jumpers, mutation_rate, overlap, random, gen_counter):
        jumpers = sorted(jumpers, key=lambda jumper: jumper.fitness, reverse=True)
        fitness_scores = np.array([jumper.fitness for jumper in jumpers])
        probabilities = fitness_scores / np.sum(fitness_scores)
        new_gen = [jumper for jumper in jumpers[:overlap]]  # Carry over the best individuals directly
        new_gen += [cls.generate_member() for _ in range(random)]  # Introduce random individuals
        for i in range(population_count - overlap - random):
            parents = np.random.choice(jumpers, size=2, p=probabilities)
            parent1, parent2 = parents[0], parents[1]
            child_genom = []
            for p1_gene, p2_gene in zip(parent1.get_genom(), parent2.get_genom()):
                roll = np.random.random()
                child_param = p1_gene if roll <= 0.5 else p2_gene
                if np.random.random() < mutation_rate:
                    child_param = np.random.uniform(cls.GENOM_RANGE[0], cls.GENOM_RANGE[1], 1).tolist()[0]
                child_genom.append(child_param)

            name = f"{cls.PREFIX}_gen{gen_counter}_{i + overlap + random}"
            child = cls(child_genom, name=name)
            new_gen.append(child)

        return new_gen
    
    def export_json(self, dir_path='./json_data/'):
        if self.json_data:
            output_path = f'{dir_path}{self.name}_{self.fitness:.2f}.json'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(self.json_data, f, indent=4)
            
    
    def get_fitness(self, gen_counter, json_data=None):
        if json_data:
            max_height = max(json_data['data']['center_of_mass_Y'])
            jump_height = max_height - json_data['data']['center_of_mass_Y'][0]
            self.fitness = jump_height
            return self.fitness
        return 0
    

class cmjJumper:

    PREFIX = 'cmj_jumper'
    START_TIME = 0.0
    END_TIME = 1
    GENOM_RANGE = (0,1) # Min and max value for each gene 
    GENE_EXPRESSION_LENGTH = 20 # Number of time points for each muscle group
    MUSCLE_GROUPS = {'Quadriceps': ['vasti_l', 'vasti_r', 'rect_fem_l', 'rect_fem_r'],
                    'Hamstrings': ['hamstrings_l', 'hamstrings_r','bifemsh_l','bifemsh_r'],
                    'Glutes': ['glut_max_r', 'glut_max_l'],
                    'Calves': ['gastroc_r', 'gastroc_l', 'soleus_r', 'soleus_l'],
                    'Tibialis Anterior': ['tib_ant_r', 'tib_ant_l'],
                    'Trunk Flexors': ['intobl_r', 'intobl_l', 'extobl_r', 'extobl_l'],
                    'Trunk Extensors': ['ercspn_r', 'ercspn_l'],
                    'Hip Flexors': ['iliopsoas_r', 'iliopsoas_l']}
    
    GENOM_LENGTH = GENE_EXPRESSION_LENGTH * len(MUSCLE_GROUPS)

    def __init__(self, genom, name=None):
        self.genom = genom
        self.timing = np.linspace(self.START_TIME, self.END_TIME, self.GENE_EXPRESSION_LENGTH)
        self.fitness = 0
        self.name = name if name is not None else '0_0'

    @classmethod
    def generate_member(cls):
        genom = [np.random.uniform(cls.GENOM_RANGE[0], cls.GENOM_RANGE[1]) for _ in range(cls.GENOM_LENGTH)]
        return cls(genom)
    
    def genom_to_excitations(self):
        muscle_excitations = {}
        for i, group in enumerate(self.MUSCLE_GROUPS.keys()):
            muscle_excitations[group] = {'value': self.genom[i*self.GENE_EXPRESSION_LENGTH:(i+1)*self.GENE_EXPRESSION_LENGTH],
                                         'time': self.timing.tolist()}
        parsed_muscle_excitations = parse_muscle_excitations(muscle_excitations, self.MUSCLE_GROUPS)
        return parsed_muscle_excitations
    
    def get_genom(self):
        return self.genom

    def export_genom(self):
        with open(f'./Genoms/{self.name}_{self.score:.2f}.txt', 'w') as f:
            for param in self.genom:
                f.write(f'{param[0]}, {param[1]},')
    
    @classmethod
    def crossover(cls, population_count, jumpers, mutation_rate, overlap, random, gen_counter):
        jumpers = sorted(jumpers, key=lambda jumper: jumper.fitness, reverse=True)
        fitness_scores = np.array([jumper.fitness for jumper in jumpers])
        probabilities = fitness_scores / np.sum(fitness_scores)
        new_gen = [jumper for jumper in jumpers[:overlap]]  # Carry over the best individuals directly
        new_gen += [cls.generate_member() for _ in range(random)]  # Introduce random individuals
        for i in range(population_count - overlap):
            parents = np.random.choice(jumpers, size=2, p=probabilities)
            parent1, parent2 = parents[0], parents[1]
            child_genom = []
            for p1_gene, p2_gene in zip(parent1.get_genom(), parent2.get_genom()):
                roll = np.random.random()
                child_param = p1_gene if roll <= 0.5 else p2_gene
                if np.random.random() < mutation_rate:
                    child_param = np.random.uniform(cls.GENOM_RANGE[0], cls.GENOM_RANGE[1], 1).tolist()[0]
                child_genom.append(child_param)
            
            name = f"{cls.PREFIX}_gen{gen_counter}_{i + overlap}"
            child = cls(child_genom, name=name)
            new_gen.append(child)

        return new_gen
    
    def export_json(self, dir_path='./json_data/'):
        if hasattr(self, 'json_data') and self.json_data:
            output_path = f'{dir_path}{self.name}_{self.fitness:.2f}.json'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(self.json_data, f, indent=4)
                
    def get_fitness(self, gen_counter, json_data=None):

        if not json_data:
            return 0

        self.fitness = 100  # Start from 100 to avoid zero fitness individuals

        weights = {
            'jump_height': 100,
            'crouch_depth': 25,
            'torso_deviation': 2,
            'minimum_knee_angle': 2}

        com_y = json_data['data']['center_of_mass_Y']
        torso_Ox = json_data['data']['torso_Oz'] #Torso medio-lateral orientation
        knee_angle = json_data['joint_angles']['data']['knee_angle_r']  # Right knee angle
        initial_height = com_y[0]
        time = json_data['data']['time']
        time_threshold_deviation = 0.5  # seconds
        threshold_index_deviation = next((i for i, t in enumerate(time) if t >= time_threshold_deviation), None)
        time_threshold_crouch = .3
        threshold_index_crouch = next((i for i, t in enumerate(time) if t >= time_threshold_crouch), None)
        torso_deviation = max(abs(np.array(torso_Ox[:threshold_index_deviation])))  # Max deviation from vertical. Only consider first 0.5 seconds
        min_height = min(com_y[:threshold_index_crouch])
        crouch_depth = abs(initial_height - min_height)
        minimum_knee_angle = min(abs(min(np.rad2deg(knee_angle[:threshold_index_crouch]))), 90)  # Minimum knee flexion angle, capped at 90 degrees

        jump_height = max(com_y) - initial_height
        # torso_cumulative = np.sum(torso_deviation)  # Cumulative deviation over time
        phase_1 = gen_counter < 40
        phase_2 = 40 <= gen_counter < 60
        phase_3 = gen_counter >= 60
        #Early training phase (learn to crouch)
        if phase_1:
            self.fitness += crouch_depth * weights['crouch_depth']  # Scale up the crouch depth
            self.fitness -= torso_deviation * weights['torso_deviation']  # Penalize torso lean to learn crouching upright
            self.fitness += minimum_knee_angle * weights['minimum_knee_angle']  # reward for knee flexion
        #Later training phase (Start rewarding jump height)
        elif phase_2:
            self.fitness += crouch_depth * weights['crouch_depth']  # Scale up the crouch depth
            self.fitness -= torso_deviation * weights['torso_deviation']  # Penalize torso lean to learn crouching upright
            self.fitness += minimum_knee_angle * weights['minimum_knee_angle']  # reward for knee flexion
            self.fitness += jump_height * weights['jump_height']  # Introduce jump height reward
        # Final training phase (Jump height is the ultimate goal)
        elif phase_3:
            self.fitness += jump_height * weights['jump_height'] * 100 # Hacky way of insuring overall score keeps increasing will fix later

        # print(f"Min Knee angle score {minimum_knee_angle * weights['minimum_knee_angle']:.3f}, Crouch Depth score (m): {crouch_depth * weights['crouch_depth']:.3f}, "
        #       f"Jump Height (m): {jump_height * weights['jump_height']:.3f}, torso deviation {torso_deviation * weights['torso_deviation']:.3f}, Fitness: {self.fitness:.3f}")

        return self.fitness if self.fitness > 0 else 0 # Ensure non-negative fitness
        