import os

def check_atom_movement(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                try:
                    x, y, z = map(float, [line[30:38], line[38:46], line[46:54]])
                    if abs(x) > 1000 or abs(y) > 1000 or abs(z) > 1000:
                        return True
                except ValueError:
                    return False
    return False

def process_trajectory_files():
    directories = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('run')]
    print("Directories to search:")
    print(directories)
    exceeded_box_dirs = []
    within_box_dirs = []
    for directory in directories:
        file_path = os.path.join(directory, 'tra.pdb')
        if os.path.isfile(file_path):
            try:
                if check_atom_movement(file_path):
                    exceeded_box_dirs.append(directory)
                else:
                    within_box_dirs.append(directory)
            except ValueError:
                print("Error processing directory: {}".format(directory))
        else:
            print("File not found: {}".format(file_path))
    print("Directories where atoms exceeded the simulation box:")
    print(exceeded_box_dirs)
    print("\nDirectories where atoms remained within the simulation box:")
    print(within_box_dirs)

process_trajectory_files()
