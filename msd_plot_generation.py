import os
import matplotlib.pyplot as plt
from typing import List

class Point3d:
    def __init__(self, xx: float, yy: float, zz: float):
        self.x = xx
        self.y = yy
        self.z = zz

class BoxBounds:
    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper

def check_bounds(positions: List[Point3d], bounds: BoxBounds) -> bool:
    for pos in positions:
        if (
            pos.x < bounds.lower or pos.x > bounds.upper or
            pos.y < bounds.lower or pos.y > bounds.upper or
            pos.z < bounds.lower or pos.z > bounds.upper
        ):
            return False
    return True

def parse_trajectory(file_path: str) -> List[Point3d]:
    atoms = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('ATOM'):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atoms.append(Point3d(x, y, z))
    return atoms

def calculate_msd(atoms: List[Point3d]) -> List[float]:
    msd = [0]
    for i in range(1, len(atoms)):
        dx = atoms[i].x - atoms[0].x
        dy = atoms[i].y - atoms[0].y
        dz = atoms[i].z - atoms[0].z
        msd.append(dx**2 + dy**2 + dz**2)
    return msd

def plot_msd(steps: List[int], msd: List[float], output_dir: str) -> None:
    plt.figure()
    plt.plot(steps, msd)
    plt.xlabel('Monte Carlo Steps')
    plt.ylabel('Mean Square Displacement')
    plt.title('Mean Square Displacement')

    # Extract directory name and use it as the filename
    filename = os.path.basename(os.path.normpath(output_dir)) + '.png'
    
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def process_directory(directory: str, trajectory_file_name: str, cm_file_name: str) -> None:
    trajectory_file = os.path.join(directory, trajectory_file_name)
    cm_file = os.path.join(directory, cm_file_name)

    atoms = parse_trajectory(trajectory_file)
    bounds = BoxBounds(-1000, 1000)  # Update bounds

    if check_bounds(atoms, bounds):
        cm_data = []
        with open(cm_file, 'r') as file:
            for line in file:
                if not line.startswith('#'):
                    cm_data.append(list(map(float, line.split())))

        steps = list(range(len(cm_data)))
        msd = calculate_msd([Point3d(*point) for point in cm_data])
        output_dir = os.path.join(directory, 'mean_square_displacement')
        os.makedirs(output_dir, exist_ok=True)
        plot_msd(steps, msd, output_dir)

def process_directories(root_directory: str, trajectory_file_name: str, cm_file_name: str) -> None:
    directories = [name for name in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, name)) and name.startswith('run')]
    for directory in directories:
        process_directory(os.path.join(root_directory, directory), trajectory_file_name, cm_file_name)

def main():
    root_directory = '.'  # Path to the parent directory of the "run" directories
    trajectory_file_name = 'tra.pdb'  # Define the file name
    cm_file_name = 'cm.dat'  # Define the file name
    process_directories(root_directory, trajectory_file_name, cm_file_name)

if __name__ == "__main__":
    main()
    
