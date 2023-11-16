import subprocess
import os
import numpy as np

# Define the start and end of the logarithmic values
start = 1.7
end = 3.0
num_points = 200

# Generate exponents logarithmically spaced from 1.7 to 3.0
exponents = np.linspace(start, end, num=num_points)

# Define fixed values for inner_cycle, outer_cycle, and cycle_factor
inner_cycle = 100
outer_cycle = 1000
cycle_factor = 100

# Run the following loop for each exponent
for i, exponent in enumerate(exponents):
    residue_count = 10 ** exponent
    directory_name = f"run{i:03d}_inner{inner_cycle}_outer{outer_cycle}_factor{cycle_factor}_res{int(residue_count):03d}"
    os.makedirs(directory_name, exist_ok=True)

    command = [
        "../surpass",
        "--n-chains", "1",
        "--n-res-in-chain", str(int(residue_count)),
        "--inner-cycles", str(inner_cycle),
        "--outer-cycles", str(outer_cycle),
        "--cycle-factor", str(cycle_factor),
        "--box-size", "2000"
    ]

    subprocess.run(command, cwd=directory_name)