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
        model = Model(atoms)
        models.append(model)

    return models


def playMovie(models: list):
    # Create the scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Start with the first frame
    x = [atom.x for atom in models[0].atoms]
    y = [atom.y for atom in models[0].atoms]
    z = [atom.z for atom in models[0].atoms]
    scatter = ax.scatter(x, y, z)

    # Set the axes limits
    ax.set_xlim([-500, 500])
    ax.set_ylim([-500, 500])
    ax.set_zlim([-500, 500])

    plt.ion()
    plt.show()

    # Update the scatter plot for each frame
    for model in models[1:]:
        x = [atom.x for atom in model.atoms]
        y = [atom.y for atom in model.atoms]
        z = [atom.z for atom in model.atoms]
        scatter._offsets3d = (x, y, z)
        plt.draw()
        plt.pause(0.1)  # Adjust this to control the speed of the animation


if __name__ == "__main__":
    models = getModels('tra.pdb')
    playMovie(models)