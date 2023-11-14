import random
import subprocess
import os

for i in range(1000):
    inner_cycle = random.randint(1, 100)
    outer_cycle = random.randint(1, 1000)
    cycle_factor = random.randint(1, 100)

    directory_name = "run{:03d}".format(i)
    os.makedirs(directory_name, exist_ok=True)

    command = f"surpass --n-chains 1 --n-res-in-chain 100 --inner-cycles {inner_cycle} --outer-cycles {outer_cycle} --cycle-factor {cycle_factor} --box-size 2000"

    subprocess.run(command, shell=True, cwd=directory_name)