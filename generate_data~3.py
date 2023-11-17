import subprocess
import os
import numpy as np

# Define the start and end of the logarithmic values
start = 1.7
end = 3.0
num_points = 3

# Generate exponents logarithmically spaced from 1.7 to 3.0
exponents = np.linspace(start, end, num=num_points)

# Define the ranges for the loop variables
inner_cycle_range = range(10, 101, 30)
outer_cycle_range = range(10, 1001, 300)
cycle_factor_range = range(10, 101, 30)
num_points_range = range(1, 3)

# Run the loops for each combination of loop variables
for inner_cycle in inner_cycle_range:
    for outer_cycle in outer_cycle_range:
        for cycle_factor in cycle_factor_range:
            for num_points in num_points_range:
                for i, exponent in enumerate(exponents):
                    residue_count = 10 ** exponent
                    directory_name = f"run{i:03d}_inner{inner_cycle}_outer{outer_cycle}_factor{cycle_factor}_res{int(residue_count):$
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
