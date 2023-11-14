import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time


class Atom:
    def __init__(self, id, atom_type, residue_type, chain_id, residue_id, x, y, z, occupancy, temp_factor, element):
        self.id = id
        self.atom_type = atom_type
        self.residue_type = residue_type
        self.chain_id = chain_id
        self.residue_id = residue_id
        self.x = x
        self.y = y
        self.z = z
        self.occupancy = occupancy
        self.temp_factor = temp_factor
        self.element = element


class Model:
    def __init__(self, atoms):
        self.atoms = atoms


def getModels(fileName: str):
    # Open the file and read the lines
    with open(fileName, 'r') as f:
        lines = f.readlines()

    # Split the data into models
    models_str = ''.join(lines).split('ENDMDL\n')

    # Initialize max and min coordinates
    max_coords = {"x": float('-inf'), "y": float('-inf'), "z": float('-inf')}
    min_coords = {"x": float('inf'), "y": float('inf'), "z": float('inf')}

    # Extract the atoms for each model
    models = []
    for model_str in models_str:
        atoms = []
        atoms_str = model_str.split('\n')[1:-1]  # Ignore the 'MODEL' and 'ENDMDL' lines
        for atom_str in atoms_str:
            atom_id = int(atom_str[6:11])
            atom_type = atom_str[12:16].strip()
            residue_type = atom_str[17:20].strip()
            chain_id = atom_str[21]
            residue_id = int(atom_str[22:26])
            x, y, z = map(float, [atom_str[30:38], atom_str[38:46], atom_str[46:54]])
            occupancy, temp_factor = map(float, [atom_str[54:60], atom_str[60:66]])
            element = atom_str[76:78].strip()
            atom = Atom(atom_id, atom_type, residue_type, chain_id, residue_id, x, y, z, occupancy, temp_factor,
                        element)
            atoms.append(atom)

            # Update max and min coordinates
            max_coords["x"] = max(max_coords["x"], x)
            max_coords["y"] = max(max_coords["y"], y)
            max_coords["z"] = max(max_coords["z"], z)
            min_coords["x"] = min(min_coords["x"], x)
            min_coords["y"] = min(min_coords["y"], y)
            min_coords["z"] = min(min_coords["z"], z)

        model = Model(atoms)
        models.append(model)

    return models, max_coords, min_coords

if __name__ == "__main__":
    models, max_coords, min_coords = getModels('tra.pdb')
    print(f"Max coordinates: {max_coords}")
    print(f"Min coordinates: {min_coords}")