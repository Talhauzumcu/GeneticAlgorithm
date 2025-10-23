import opensim as osim
import os
from utils import *

class modelController:
    """
    Handles loading, modifying properties (posture, strength, activation),
    and managing specific components like external forces and visualization
    settings for the OpenSim model.
    """
    def __init__(self, model_filepath: str, visualize: bool = False):
        self.model_filepath = model_filepath
        self.visualize = visualize
        self.model = osim.Model(self.model_filepath)
        self.state = None
        self._original_viz_setting = False # Store the original visualizer state
        self.state_list = []
        # Store original strengths dynamically when scaling is applied
        self._original_strengths = {}
        self._added_force_component_names = [] # Keep track of added forces

    def initialize(self):
        """Initializes the model's system and gets the default state."""
        try:
            self.model.setUseVisualizer(self.visualize) #Set the visualizer before initialization to avoid repeat initialization
            self.model.finalizeConnections()
            self.state = self.model.initSystem()
            self.equlibriate_muscles()
            if self.visualize:
                self.model.updVisualizer().updSimbodyVisualizer().setShowSimTime(True)
            # print(f"Model {os.path.basename(self.model_filepath)} initialized.")
        except Exception as e:
            print(f"ERROR: Failed to initialize model system: {e}")
            self.state = None # Ensure state is None on failure
            raise # Re-raise the exception
    
    def set_use_visualizer(self, use_visualizer: bool):
        self.model.setUseVisualizer(use_visualizer)
        
    def reload_model(self):
        """Reloads the model from the original file."""
        try:
            self.model = osim.Model(self.model_filepath)
            self.initialize() # Re-initialize after reloading
            print(f"Model {os.path.basename(self.model_filepath)} reloaded.")
        except Exception as e:
            print(f"ERROR: Failed to reload model: {e}")
            raise

    def set_posture(self, posture_dict: dict):
        """Sets joint coordinates."""
        
        state = self.get_state()
        coord_set = self.model.updCoordinateSet()
        print(f"Setting posture: {posture_dict}")
        for coord_name, (value, locked) in posture_dict.items(): 
            try:
                coord = coord_set.get(coord_name)
                coord.setValue(state, value)
                coord.setLocked(state, locked)
            except Exception as e:
                print(f"Warning: Could not set coordinate '{coord_name}'. Error: {e}")
        try:
             self.model.equilibrateMuscles(state)
             print("Model muscles equilibrated after posture change.")
        except Exception as e:
             print(f"Warning: Could not equilibrate muscles. Error: {e}")
        print('Posture set.')

    def set_muscle_strength(self, scale_factor: float, muscle_group= None):
        """Scales max isometric force, storing original values if first time."""
        if muscle_group is None:
            muscle_group = [muscle for muscle in self.model.getMuscleList()] 
        print(f"Setting strength scale factor {scale_factor} for {len(muscle_group)} muscles.")
        for muscle in muscle_group:
            try:
                original_strength = muscle.getMaxIsometricForce()
                muscle.setMaxIsometricForce(original_strength * scale_factor)
            except Exception as e:
                print(f"Warning: Could not scale strength for '{muscle.getName()}'. Error: {e}")
            
                  
    def reset_muscle_strength(self):
        """Resets muscles to their original stored strengths."""
        if not self._original_strengths:
            print("No original strengths stored to reset.")
            return
        force_set = self.model.getForceSet()
        print("Resetting muscle strengths...")
        for muscle_name, original_force in self._original_strengths.items():
             try:
                 muscle = osim.Muscle.safeDownCast(force_set.get(muscle_name))
                 if muscle:
                     muscle.setMaxIsometricForce(original_force)
                 else:
                     print(f"Warning: Could not find/cast muscle '{muscle_name}' during reset.")
             except Exception as e:
                 print(f"Warning: Could not reset strength for '{muscle_name}'. Error: {e}")
        # Clear stored originals after resetting (optional, depends if you re-scale later)
        # self._original_strengths = {}


    def setup_muscle_controller(self,
                               muscle_excitations: dict):
        if hasattr(self, 'controller'):
            self.model.removeController(self.controller)
        self.controller = osim.PrescribedController()
        for muscle_name, excitation_info in muscle_excitations.items():
            muscle = self.model.getMuscles().get(muscle_name)
            self.controller.addActuator(muscle)
            excitation_function = osim.PiecewiseLinearFunction()
            for t, excitation in zip(excitation_info['time'], excitation_info['value']):
                excitation_function.addPoint(t, excitation)
            self.controller.prescribeControlForActuator(muscle_name, excitation_function)
        self.model.addController(self.controller)
        self.model.finalizeConnections()

    def set_muscle_controls(self, muscle_controls_filepath: str):
        """Sets muscle activations (requires PrescribedControllers)."""
        
        self.prescribed_controller = osim.PrescribedController()
        self.prescribed_controller.set_controls_file(muscle_controls_filepath)
        self.prescribed_controller.setName("muscle_controls")
        self.model.addComponent(self.prescribed_controller)
        # Ensure the controller is properly connected
        self.model.finalizeConnections()
        print(f"Muscle controls set from {muscle_controls_filepath}.")

    @staticmethod
    def create_external_loads_from_file(force_filepath: str, config: dict):
        """
        Creates an ExternalLoads component from a file and returns it.
        This is a class method to allow easy creation without needing an instance.
        """
        ext_loads = osim.ExternalLoads()
        ext_loads.setName(config['external_load_name'])
        ext_loads.setDataFileName(force_filepath)
        ext_force = osim.ExternalForce()
        # Use component_name or a derived name if needed, ensure uniqueness if multiple ExternalForces added
        ext_force.setName(config['external_force_name'])
        ext_force.set_applied_to_body(config['applied_to_body'])
        ext_force.set_force_identifier(config['force_identifier'])
        ext_force.set_point_identifier(config['point_identifier'])
        ext_force.set_torque_identifier(config['torque_identifier'])
        ext_force.set_force_expressed_in_body(config['force_expressed_in_body'])
        ext_force.set_point_expressed_in_body(config['point_expressed_in_body'])
        ext_loads.cloneAndAppend(ext_force)

        return ext_loads
    
    def add_external_loads(self, force_filepath: str, config: dict): 
        """Adds ExternalLoads component to the model."""
        self.impact_config = config
        self.impact_force_filepath = force_filepath # Store force_filepath
        ext_loads = self.create_external_loads_from_file(force_filepath, config)
        self.model.addComponent(ext_loads)
        print(f"ModelController: Added ExternalLoads component from file {force_filepath}.")

    def _remove_external_loads_component(self, name: str):
        """Removes ExternalLoads component by name."""
        component_path = f"/forceset/{name}" # Assuming it's added under forceset
        if self.model.hasComponent(component_path):
            try:
                comp_to_remove = self.model.updComponent(component_path)
                removed = self.model.removeComponent(comp_to_remove)
                if removed:
                    print(f"ModelController: Removed ExternalLoads component '{name}'.")
                    if name in self._added_force_component_names:
                        self._added_force_component_names.remove(name)
                     # Re-initialize after removing component
                    self.initialize_state()
                    return True
                else:
                     print(f"Warning: Failed to remove component '{name}'.")
            except Exception as e:
                print(f"Warning: Error removing component '{name}'. Error: {e}")
        else:
             print(f"Warning: Component '{name}' not found for removal.")
        return False

    def _remove_all_added_external_loads(self):
        """Removes all tracked external loads."""
        print("Removing all tracked external load components...")
        # Iterate over a copy as list is modified during iteration
        names_to_remove = list(self._added_force_component_names)
        any_removed = False
        for name in names_to_remove:
             if self._remove_external_loads_component(name):
                  any_removed = True
        # No need to re-initialize here, done within _remove_external_loads_component

    def setup_output_reporter(self, output_reporter_config: dict, report_time_interval=0.0001):

        """Sets up reporters for the model to track various outputs."""
        bodies = output_reporter_config.get('body', [])
        outputs = output_reporter_config.get('output', [])
        self.output_reporter = osim.TableReporterVec3()
        self.output_reporter.setName(output_reporter_config.get('name', 'output_reporter'))
        self.output_reporter.set_report_time_interval(report_time_interval)
        #Setup the output reporter for each body and output specified in the config
        for body_name in bodies:
            body = self.model.getBodySet().get(body_name)
            if body:
                for output in outputs:
                    
                    self.output_reporter.addToReport(body.getOutput(output), f"{body_name}_{output}")
            else:
                print(f"Warning: Body '{body_name}' not found in the model. Skipping reporter setup.")

        self.model.addComponent(self.output_reporter)

    def setup_state_reporter(self, report_time_interval=0.001):
        """Sets up a states trajectory reporter for the model."""
        self.states_reporter = osim.StatesTrajectoryReporter()
        self.states_reporter.setName("States_reporter")
        self.states_reporter.set_report_time_interval(report_time_interval)
        self.model.addComponent(self.states_reporter)

    def export_states_reporter(self, output_file_prefix: str, results_dir: str):
        """
        Exports the states reporter data to a .sto file.
        """
        output_dir = os.path.join(results_dir, 'states')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        states_table = self.states_reporter.getStates().exportToTable(self.model)
        filepath = os.path.join(output_dir, f"{output_file_prefix}_states.sto")
        osim.STOFileAdapter().write(states_table, filepath)

    def setup_reporters(self, output_reporter_config, report_time_interval=0.0001):
        """Sets up reporters for the model to track various outputs."""
        bodies = output_reporter_config.get('body', [])
        outputs = output_reporter_config.get('outputs', [])
        self.output_reporter = osim.TableReporterVec3()
        self.output_reporter.setName(output_reporter_config.get('name', 'output_reporter'))
        self.output_reporter.set_report_time_interval(report_time_interval)
        #Setup the output reporter for each body and output specified in the config
        for body_name in bodies:
            body = self.model.getBodySet().get(body_name)
            if body:
                for output in outputs:
                    self.output_reporter.addToReport(body.getOutput(output), f"{body_name}_{output}")
            else:
                print(f"Warning: Body '{body_name}' not found in the model. Skipping reporter setup.")

        self.states_reporter = osim.StatesTrajectoryReporter()
        self.states_reporter.setName("States_reporter")
        self.states_reporter.set_report_time_interval(report_time_interval)

        self.force_reporter = osim.ForceReporter()
        self.force_reporter.setName("Force_reporter")
        self.force_reporter.setModel(self.model)

        self.body_kinematics_reporter = osim.BodyKinematics()
        self.body_kinematics_reporter.setName("Body_kinematics_reporter")
        self.body_kinematics_reporter.setModel(self.model)

        self.model.addComponent(self.states_reporter)
        self.model.addComponent(self.output_reporter)
        self.model.addAnalysis(self.force_reporter)
        self.model.addAnalysis(self.body_kinematics_reporter)
        # self.model.addComponent(self.force_reporter)
    
    def add_kinematics_analysis(self):
        self.kinematics_analysis = osim.BodyKinematics()
        self.kinematics_analysis.setName('body_kinematics')
        self.kinematics_analysis.setModel(self.model)
        self.kinematics_analysis.setPrintResultFiles(True)
        self.model.addAnalysis(self.kinematics_analysis)

    def add_states_reporter(self):
        self.states_reporter = osim.StatesReporter()
        self.states_reporter.setName("States_reporter")
        self.model.addAnalysis(self.states_reporter)

    
    def convert_kinematics_to_json(self, positions_csv_path: str) -> dict:
        """Convert the kinematics positions CSV output into a JSON file."""
        if not os.path.exists(positions_csv_path):
            raise FileNotFoundError(f"Positions file not found at {positions_csv_path}")

        metadata_lines = []
        metadata_pairs = {}
        headers = None
        column_data = None
        with open(positions_csv_path, 'r', encoding='utf-8') as csv_file:
            for line in csv_file:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped == 'endheader':
                    header_line = csv_file.readline()
                    if not header_line:
                        raise ValueError("Positions file missing header row after 'endheader'.")
                    headers = header_line.strip().split()
                    column_data = {header: [] for header in headers}
                    break
                if '=' in stripped:
                    key, value = stripped.split('=', 1)
                    metadata_pairs[key.strip()] = value.strip()
                else:
                    metadata_lines.append(stripped)

            if headers is None:
                raise ValueError("Could not locate column headers in positions file.")

            for line in csv_file:
                stripped = line.strip()
                if not stripped:
                    continue
                values = stripped.split()
                if len(values) != len(headers):
                    # skip malformed lines but keep a trace for debugging
                    print(f"Warning: Skipping malformed data row in positions file: {line.rstrip()}")
                    continue
                for key, value in zip(headers, values):
                    try:
                        column_data[key].append(float(value))
                    except ValueError:
                        column_data[key].append(value)

        json_payload = {
            "metadata": {
                "lines": metadata_lines,
                "pairs": metadata_pairs
            },
            "columns": headers,
            "data": column_data
        }

        return json_payload
    
    def convert_muscle_activations_to_json(self, states_csv_path: str) -> dict:
        """Convert muscle activation data from states CSV to JSON format.
        
        The states CSV has columns like:
        - time
        - /forceset/{muscle_name}/activation
        - /forceset/{muscle_name}/fiber_length
        
        We extract only the activation columns and clean up the names.
        """
        if not os.path.exists(states_csv_path):
            raise FileNotFoundError(f"States file not found at {states_csv_path}")
        
        metadata_lines = []
        metadata_pairs = {}
        headers = None
        column_data = None
        
        with open(states_csv_path, 'r', encoding='utf-8') as csv_file:
            for line in csv_file:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped == 'endheader':
                    header_line = csv_file.readline()
                    if not header_line:
                        raise ValueError("States file missing header row after 'endheader'.")
                    headers = header_line.strip().split('\t')  # Tab-separated
                    column_data = {header: [] for header in headers}
                    break
                if '=' in stripped:
                    key, value = stripped.split('=', 1)
                    metadata_pairs[key.strip()] = value.strip()
                else:
                    metadata_lines.append(stripped)
            
            if headers is None:
                raise ValueError("Could not locate column headers in states file.")
            
            # Read data rows
            for line in csv_file:
                stripped = line.strip()
                if not stripped:
                    continue
                values = stripped.split('\t')  # Tab-separated
                if len(values) != len(headers):
                    print(f"Warning: Skipping malformed data row in states file")
                    continue
                for key, value in zip(headers, values):
                    try:
                        column_data[key].append(float(value))
                    except ValueError:
                        column_data[key].append(value)
        
        # Extract only activation columns and clean up names
        # Column format: /forceset/{muscle_name}/activation
        activation_data = {'time': column_data['time']}
        clean_headers = ['time']
        
        for header in headers:
            if '/activation' in header and '/forceset/' in header:
                # Extract muscle name from /forceset/{muscle_name}/activation
                muscle_name = header.split('/forceset/')[1].split('/activation')[0]
                activation_data[muscle_name] = column_data[header]
                clean_headers.append(muscle_name)
        
        json_payload = {
            "metadata": {
                "lines": metadata_lines,
                "pairs": metadata_pairs
            },
            "columns": clean_headers,
            "data": activation_data
        }
        
        return json_payload
    
    def convert_joint_angles_to_json(self, states_csv_path: str) -> dict:
        """Convert joint angle data from states CSV to JSON format.
        
        The states CSV has columns like:
        - time
        - /jointset/{joint_name}/{coordinate_name}/value
        - /jointset/{joint_name}/{coordinate_name}/speed
        
        We extract only the joint angle value columns and clean up the names.
        """
        if not os.path.exists(states_csv_path):
            raise FileNotFoundError(f"States file not found at {states_csv_path}")
        
        metadata_lines = []
        metadata_pairs = {}
        headers = None
        column_data = None
        
        with open(states_csv_path, 'r', encoding='utf-8') as csv_file:
            for line in csv_file:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped == 'endheader':
                    header_line = csv_file.readline()
                    if not header_line:
                        raise ValueError("States file missing header row after 'endheader'.")
                    headers = header_line.strip().split('\t')  # Tab-separated
                    column_data = {header: [] for header in headers}
                    break
                if '=' in stripped:
                    key, value = stripped.split('=', 1)
                    metadata_pairs[key.strip()] = value.strip()
                else:
                    metadata_lines.append(stripped)
            
            if headers is None:
                raise ValueError("Could not locate column headers in states file.")
            
            # Read data rows
            for line in csv_file:
                stripped = line.strip()
                if not stripped:
                    continue
                values = stripped.split('\t')  # Tab-separated
                if len(values) != len(headers):
                    print(f"Warning: Skipping malformed data row in states file")
                    continue
                for key, value in zip(headers, values):
                    try:
                        column_data[key].append(float(value))
                    except ValueError:
                        column_data[key].append(value)
        
        # Extract only joint angle value columns and clean up names
        # Column format: /jointset/{joint_name}/{coordinate_name}/value
        joint_angles_data = {'time': column_data['time']}
        clean_headers = ['time']
        
        for header in headers:
            if '/jointset/' in header and '/value' in header and '/speed' not in header:
                # Extract coordinate name from /jointset/{joint_name}/{coordinate_name}/value
                parts = header.split('/jointset/')[1].split('/value')[0].split('/')
                if len(parts) >= 2:
                    coordinate_name = parts[-1]  # Get the coordinate name (e.g., 'knee_angle_r')
                    joint_angles_data[coordinate_name] = column_data[header]
                    clean_headers.append(coordinate_name)
        
        json_payload = {
            "metadata": {
                "lines": metadata_lines,
                "pairs": metadata_pairs
            },
            "columns": clean_headers,
            "data": joint_angles_data
        }
        
        return json_payload
    
    
    def prepare_json(self, results_path, positions_filename: str, states_filename: str) -> dict:
        """Prepares a combined JSON structure with kinematics and muscle activations."""
        json_data = self.convert_kinematics_to_json(os.path.join(results_path, positions_filename))
        activation_data = self.convert_muscle_activations_to_json(os.path.join(results_path, states_filename))
        joint_angles_data = self.convert_joint_angles_to_json(os.path.join(results_path, states_filename))
        json_data['muscle_activations'] = activation_data
        json_data['joint_angles'] = joint_angles_data
        muscles = self.model.getMuscles()
        muscle_names = [muscles.get(i).getName() for i in range(muscles.getSize())]
        json_data['metadata']['muscles'] = muscle_names

        return json_data
    
    def save_results(self, output_file_prefix: str, results_dir: str):
        if not results_dir:
            raise ValueError("results_dir must be provided when saving results.")

        results_path = os.path.abspath(results_dir)
        os.makedirs(results_path, exist_ok=True)
        self.kinematics_analysis.printResults(
            f'{output_file_prefix}_kinematicResults',
            results_path, 1/240, '.sto'
        )
        self.states_reporter.printResults(
            f'{output_file_prefix}_actuationResults',
            results_path, 1/240, '.sto'
        )
        positions_file_name = f"{output_file_prefix}_kinematicResults_body_kinematics_pos_global.sto"
        positions_csv_path = os.path.join(results_path, positions_file_name)

        states_file_name = f"{output_file_prefix}_actuationResults_States_reporter_states.sto"
        states_csv_path = os.path.join(results_path, states_file_name)
        
        # print(f"Body kinematics data saved to: {positions_csv_path}")
        # print(f"Muscle activation data saved to: {states_csv_path}")
        
        return results_path, positions_file_name, states_file_name


    def export_output_reporter(self, output_file_prefix: str, results_dir: str):
        """
        Exports the output reporter data to a .sto file.
        """
        output_dir = os.path.join(results_dir, 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filepath = os.path.join(output_dir, output_file_prefix)
        output_reporter_table = self.output_reporter.getTable()
        osim.STOFileAdapterVec3().write(output_reporter_table, filepath)
        print(f"Reporter data saved to: {filepath}")

    def get_model(self) -> osim.Model:
        return self.model

    def get_state(self) -> osim.State:
        """Gets the current state, initializing if necessary."""
        # Don't automatically initialize here, let caller handle timing
        if not self.state:
             print("Warning: Accessing state before initialization.")
        return self.state
    
    def get_output_reporter(self) -> osim.TableReporterVec3:
        """Returns the Vec3 reporter."""
        if not self.output_reporter:
            print("Warning: Vec3 reporter not initialized.")
            return None
        return self.output_reporter
    
    def get_states_reporter(self) -> osim.StatesTrajectoryReporter:
        """Returns the States reporter."""
        if not self.states_reporter:
            print("Warning: States reporter not initialized.")
            return None
        return self.states_reporter
    
    def equlibriate_muscles(self):
        """Equilibrates the model's muscles."""
        if not self.state:
            print("Warning: State not initialized for muscle equilibration.")
            return False
        try:
            self.model.equilibrateMuscles(self.state)
            # print("Model muscles equilibrated.")
            return True
        except Exception as e:
            print(f"Warning: Could not equilibrate muscles. Error: {e}")
            return False

    def set_manager(self, manager: osim.Manager):
        """Sets the manager for the model."""
        self.manager = manager
        # print("ModelController: Manager set.")
    
    def get_manager(self) -> osim.Manager:
        """Returns the manager."""
        if not hasattr(self, 'manager'):
            print("Warning: Manager not set.")
            return None
        return self.manager
    
    def append_state(self, state: osim.State):
        """Appends a state to the list."""
        self.state_list.append(state)
    
    def get_states(self):
        """Returns the list of states."""
        return self.state_list
    
   