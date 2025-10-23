import numpy as np
from utils import *
import os
import json

class ObjectBase:
    PREFIX = 'BaseObject'
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
    
    @classmethod
    def generate_population_from_json(cls, population_size, folder='./initial_population/'):
        population = []
        json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
        for i in range(min(len(json_files), population_size)):
            file_path = os.path.join(folder, json_files[i])
            with open(file_path, 'r') as f:
                json_data = json.load(f)

            muscle_excitations = json_data['muscle_activations'] #wrongly named in the json files as activations
            genom = cls.excitations_to_genom(muscle_excitations)
            name = f"{cls.PREFIX}_init_{i}"
            member = cls(genom, name=name)
            population.append(member)
        
        if len(population) < population_size:
            for _ in range(population_size - len(population)):
                population.append(cls.generate_member())

        return population
    
    @classmethod
    def excitations_to_genom(cls, muscle_excitations):
        genom = []
        for group in cls.MUSCLE_GROUPS.keys():
            muscle_from_group = cls.MUSCLE_GROUPS[group][0]
            excitation = muscle_excitations['data'][muscle_from_group]
            excitation_length = len(excitation)
            step = excitation_length // cls.GENE_EXPRESSION_LENGTH
            excitations = [excitation[i * step] for i in range(cls.GENE_EXPRESSION_LENGTH)]
            genom.extend(excitations)
        
        return genom

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

class Jumper(ObjectBase):
    PREFIX = 'jumper'

    def __init__(self, genom, name=None):
        super().__init__(genom, name)
    
    def get_fitness(self, gen_counter, json_data=None):
        if json_data:
            max_height = max(json_data['data']['center_of_mass_Y'])
            jump_height = max_height - json_data['data']['center_of_mass_Y'][0]
            self.fitness = jump_height
            return self.fitness
        return 0
    

class cmjJumper(ObjectBase):
    PREFIX = 'cmjJumper'

    def __init__(self, genom, name=None):
        super().__init__(genom, name)

    def get_fitness(self, gen_counter, json_data=None):
        if not json_data:
            return 0

        self.fitness = 100  # Start from 100 to avoid zero fitness individuals

        # Weights are tuned depending on the importance of each metric
        weights = {
            'jump_height': 100,
            'crouch_depth': 25,
            'torso_deviation': 2,
            'minimum_knee_angle': 2}

        # This section is just extracting the relevant data. I was going to refactor it later. But you know how it goes.
        com_y = json_data['data']['center_of_mass_Y']
        torso_Ox = json_data['data']['torso_Oz'] #Torso medio-lateral orientation
        knee_angle = json_data['joint_angles']['data']['knee_angle_r']  # Right knee angle
        initial_height = com_y[0]
        time = json_data['data']['time']

        # I used timings to define phases of the jump. So the model wouldn't get punished for falling at the end of the jump for example
        time_threshold_deviation = 0.5  # 0.5 seconds to assess torso deviation
        threshold_index_deviation = next((i for i, t in enumerate(time) if t >= time_threshold_deviation), None)
        time_threshold_crouch = .3  # 0.3 seconds to assess crouch
        threshold_index_crouch = next((i for i, t in enumerate(time) if t >= time_threshold_crouch), None)
        
        torso_deviation = max(abs(np.array(torso_Ox[:threshold_index_deviation])))  # Max deviation from vertical. Only consider first 0.5 seconds
        min_height = min(com_y[:threshold_index_crouch])
        crouch_depth = abs(initial_height - min_height)
        minimum_knee_angle = min(abs(min(np.rad2deg(knee_angle[:threshold_index_crouch]))), 90)  # Minimum knee flexion angle, capped at 90 degrees
        jump_height = max(com_y) - initial_height

        # Define training phases based on generation count
        phase_1 = gen_counter < 40
        phase_2 = 40 <= gen_counter < 60
        phase_3 = gen_counter >= 60

        #Early training phase (learn to crouch)
        if phase_1:
            self.fitness += crouch_depth * weights['crouch_depth']  
            self.fitness -= torso_deviation * weights['torso_deviation']  # Penalize torso lean to learn crouching upright
            self.fitness += minimum_knee_angle * weights['minimum_knee_angle']  # reward for knee flexion

        #Later training phase (Start rewarding jump height)
        elif phase_2:
            self.fitness += crouch_depth * weights['crouch_depth']  
            self.fitness -= torso_deviation * weights['torso_deviation']  # Penalize torso lean to learn crouching upright
            self.fitness += minimum_knee_angle * weights['minimum_knee_angle']  # reward for knee flexion
            self.fitness += jump_height * weights['jump_height']  # Introduce jump height reward
            
        # Final training phase (Jump height is the ultimate goal)
        elif phase_3:
            self.fitness += jump_height * weights['jump_height'] * 100 # Hacky way of insuring overall score keeps increasing will fix later

        # print(f"Min Knee angle score {minimum_knee_angle * weights['minimum_knee_angle']:.3f}, Crouch Depth score (m): {crouch_depth * weights['crouch_depth']:.3f}, "
        #       f"Jump Height (m): {jump_height * weights['jump_height']:.3f}, torso deviation {torso_deviation * weights['torso_deviation']:.3f}, Fitness: {self.fitness:.3f}")

        return self.fitness if self.fitness > 0 else 0 # Ensure non-negative fitness
    

class Flipper(ObjectBase):

    PREFIX = 'flipper'
    def __init__(self, genom, name=None):
        super().__init__(genom, name)
                
    def get_fitness(self, gen_counter, json_data=None):
        if not json_data:
            return 0

        def get_airborn_idx(calc_pos):
            take_off_threshold = 0.05  # 20 cm threshold to start giving points for being airborn
            start_pos = calc_pos[0]
            calc_pos = np.array(calc_pos)
            airborn_idx = np.where(calc_pos - start_pos > take_off_threshold)[0]
            if len(airborn_idx) == 0:
                return -1, -1  # Never airborn
            to_idx = airborn_idx[0]  # Take-off index
            td_idx = np.where(calc_pos[to_idx:] - start_pos <= take_off_threshold)[0] # Find touchdown after take-off
            if len(td_idx) == 0:
                td_idx = -1 # Never touches down again
            else:
                td_idx = td_idx[0] + to_idx
            return to_idx, td_idx 

        def calculate_knee_flexion_score():
            if to_idx == -1:
                return 0  # No airborn phase detected
            air_kf = knee_flexion[to_idx:td_idx] #Knee flexion during airborn phase
            min_air_kf = np.min(air_kf)
            target_knee_flexion = 150  # Target knee flexion angle for a good backflip
            knee_flexion_score = min(1, -min_air_kf/target_knee_flexion)
            return knee_flexion_score
    
        def calculate_hip_flexion_score():
            if to_idx == -1:
                return 0  # No airborn phase detected
            air_hf = hip_flexion[to_idx:td_idx] #Hip flexion during airborn phase
            max_air_hf = np.max(air_hf)
            target_hip_flexion = 150  # Target hip flexion angle for a good backflip
            hip_flexion_score = min(1, max_air_hf/target_hip_flexion)
            return hip_flexion_score
        
        def calculate_rotation_score(): 
            if to_idx == -1:
                return 0  # No airborn phase detected
            pelvis_rotation = pelvis_tilt[to_idx:td_idx]
            total_rotation = pelvis_rotation[-1] - pelvis_rotation[0]   #total rotation during airborn phase
            target_rotation = 400  # Target rotation for a full backflip
            rotation_score = min(1.0, total_rotation / target_rotation)
            return rotation_score
        
        def calculate_height_score():
            max_height = np.max(com_y)
            starting_height = com_y[0]
            jump_height = max_height - starting_height
            
            target_height = 0.70  # Expected height for a good backflip
            height_score = min(1.0, jump_height / target_height)
            return height_score
        

        # Extracting the relevant data from the json
        com_y = json_data['data']['center_of_mass_Y'] #Center of mass vertical position
        pelvis_tilt = np.rad2deg(json_data['joint_angles']['data']['pelvis_tilt']) #Pelvis medio-lateral orientation
        knee_flexion = np.rad2deg(json_data['joint_angles']['data']['knee_angle_r'])
        hip_flexion = np.rad2deg(json_data['joint_angles']['data']['hip_flexion_r'])
        lumbar_extension = np.rad2deg(json_data['joint_angles']['data']['lumbar_extension'])
        calc_pos = json_data['data']['calcn_r_Y']  # Right heel vertical position, to determine flight phase
        to_idx, td_idx = get_airborn_idx(calc_pos) # Get the phase where the model is in flight    
        
        ## Get the scores
        scores = {
            'rotation': calculate_rotation_score(),
            'jump_height': calculate_height_score(),
            'knee_flexion': calculate_knee_flexion_score(),
            'hip_flexion': calculate_hip_flexion_score(),
        }
       
        weights = {
            'rotation': 1.0,
            'jump_height': 0.75,
            'knee_flexion': 0.5,
            'hip_flexion': 0.5,
        }

        self.fitness = sum((weights[k] * scores[k] for k in scores))

        if np.min(hip_flexion) < -30:
            self.fitness *= 0.5  # Penalize extreme hip hyperextension

        if np.max(lumbar_extension) > 40:
            self.fitness *= 0.5  # Penalize excessive lumbar extension

        if np.min(lumbar_extension) < -80:
            self.fitness *= 0.5  # Penalize excessive lumbar flexion

        return self.fitness if self.fitness > 0 else 0 # Ensure non-negative fitness
        