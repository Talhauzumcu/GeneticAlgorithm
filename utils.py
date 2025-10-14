def parse_muscle_excitations(muscle_excitations, muscle_groups):
    parsed_excitations = {}
    for muscle in muscle_excitations.keys():
        excitation = muscle_excitations[muscle]['value']
        time = muscle_excitations[muscle]['time']
        muscles = muscle_groups[muscle]
        for m in muscles:
            parsed_excitations[m] = {'value': excitation, 'time': time}
    return parsed_excitations
