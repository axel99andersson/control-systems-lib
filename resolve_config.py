import json

def resolve_config(system: str, filepath: str):
    with open(filepath, 'r') as file:
        cfg = json.loads(file)

    if system == "car":
        pass

    elif system == "copter":
        pass
    

