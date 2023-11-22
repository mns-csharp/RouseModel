import os
import subprocess

def run_surpass(n_residues_range, inner_cycles_range, outer_cycles_range, cycle_factor_range):
    # Define the path to the Surpass executable
    surpass_path = r'/home/mohammad/bioshell_v4/BioShell/target/release/surpass'
    run_count = 0
    # Loop over the desired range of parameters
    for n_residues in n_residues_range:
        for inner_cycles in inner_cycles_range:
            for outer_cycles in outer_cycles_range:
                for cycle_factor in cycle_factor_range:
                    # Create the output directory name
                    dir_name = f'run{run_count}_inner{inner_cycles}_outer{outer_cycles}_factor{cycle_factor}_residue{n_residues}'
                    # Create the directory if it doesn't exist
                    os.makedirs(dir_name, exist_ok=True)
                    # Change the working directory
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

                    # Change back to the parent directory
                    os.chdir('..')

# Call the function
n_residues_range = range(100, 1001, 100)
inner_cycles_range = range(1, 1001, 100)
outer_cycles_range = range(1000, 10001, 1000)
cycle_factor_range = range(1, 101, 10)

run_surpass(n_residues_range, inner_cycles_range, outer_cycles_range, cycle_factor_range)

