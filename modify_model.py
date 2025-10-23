import opensim as osim

def increase_muscle_strength(model_path, output_path, strength_multiplier=4.0):
    """
    Load an OpenSim model, increase all muscle strengths by a multiplier, and save it.
    
    Args:
        model_path: Path to the input model file
        output_path: Path to save the modified model
        strength_multiplier: Factor to multiply all muscle max isometric forces by
    """
    # Load the model
    print(f"Loading model from: {model_path}")
    model = osim.Model(model_path)
    
    # Get all muscles in the model
    muscle_set = model.getMuscles()
    num_muscles = muscle_set.getSize()
    
    print(f"Found {num_muscles} muscles in the model")
    print(f"Increasing strength by {strength_multiplier}x...")
    
    # Iterate through all muscles and increase their max isometric force
    for i in range(num_muscles):
        muscle = muscle_set.get(i)
        current_force = muscle.getMaxIsometricForce()
        new_force = current_force * strength_multiplier
        muscle.setMaxIsometricForce(new_force)
        print(f"  {muscle.getName()}: {current_force:.2f} N -> {new_force:.2f} N")
    
    # Finalize connections and save
    model.finalizeConnections()
    print(f"\nSaving modified model to: {output_path}")
    model.printToXML(output_path)
    print("Done!")

if __name__ == "__main__":
    # Model path from main.py
    input_model = './models/H0918v3_web_backflip_5x.osim'
    output_model = './models/H0918v3_web_backflip_20x.osim'

    increase_muscle_strength(input_model, output_model, strength_multiplier=4.0)