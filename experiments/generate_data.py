import os
import subprocess
import numpy as np

def run_surpass(n_residues_min=50, n_residues_max=2001, inner_cycles=10, outer_cycles=1000, cycle_factor=10):
    # Define the path to the Surpass executable
    surpass_path = r'/home/mohammad/bioshell_v4/BioShell/target/release/surpass'
    run_count = 0
    # Generate a list of 20 residue-counts spaced equally on a log scale
    n_residues_values = np.logspace(np.log10(n_residues_min), np.log10(n_residues_max), num=10)
    # Loop over the generated residue-counts
    for n_residues in n_residues_values:
        # Round n_residues to the nearest integer, as it must be an integer
        n_residues = round(n_residues)
        # Create the output mc_directory_path name
        dir_name = f'run{run_count}_inner{inner_cycles}_outer{outer_cycles}_factor{cycle_factor}_residue{n_residues}'
        # Create the mc_directory_path if it doesn't exist
        os.makedirs(dir_name, exist_ok=True)
        # Change the working mc_directory_path
        os.chdir(dir_name)
        # Form the Surpass command
        cmd = [
            surpass_path,
            '-r', str(n_residues),
            '-i', str(inner_cycles),
            '-o', str(outer_cycles),
            '-s', str(cycle_factor),
        ]
        # Run the Surpass command
        subprocess.run(cmd)
        run_count = run_count + 1
        # Change back to the parent mc_directory_path
        os.chdir('../..')


if __name__ == "__main__":
    # Call the function
    run_surpass()