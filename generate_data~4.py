import os
import subprocess

def run_surpass(n_residues=75, inner_cycles=10, outer_cycles_start=1000, outer_cycles_end=100001, cycle_factor=10, increment=3333):
    # Define the path to the Surpass executable
    surpass_path = r'/home/mohammad/bioshell_v4/BioShell/target/release/surpass'
    run_count = 0
    # Loop over the desired range of outer cycles
    for outer_cycles in range(outer_cycles_start, outer_cycles_end, increment):
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
run_surpass(inner_cycles=100, outer_cycles_start=10000, outer_cycles_end=10001, cycle_factor=100, increment=1)
