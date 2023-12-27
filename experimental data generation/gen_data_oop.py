import os
import math
import subprocess
from datetime import datetime
import numpy as np

class DataGenerator:
    def __init__(self, surpass_path, residue_length_min, residue_length_max, inner_cycle, outer_cycle, cycle_factor, num_of_uniq_residue_lengths, is_log):
        self.surpass_path = surpass_path
        self.residue_length_min = residue_length_min
        self.residue_length_max = residue_length_max
        self.inner_cycle = inner_cycle
        self.outer_cycle = outer_cycle
        self.cycle_factor = cycle_factor
        self.num_of_uniq_residue_lengths = num_of_uniq_residue_lengths
        self.is_log = is_log

    def run(self):
        root_directory = os.path.dirname(self.surpass_path)
        mc_directory = self.get_current_datetime_as_directory_name()
        run_count = 0
        # Generate a list of n residue-counts spaced equally
        n_residues_values = []
        if self.is_log:
            n_residues_values = self.generate_log_space(self.residue_length_min, self.residue_length_max, self.num_of_uniq_residue_lengths)
        else:
            n_residues_values = np.linspace(self.residue_length_min, self.residue_length_max, self.num_of_uniq_residue_lengths)
        # END of if-else
        #
        # Loop over the generated residue-counts
        for n_residues in n_residues_values:
            # Round n_residues to the nearest integer, as it must be an integer
            n_residues_rounded = int(round(n_residues))
            # Create the output directory name
            dir_name = f"run{run_count}_inner{self.inner_cycle}_outer{self.outer_cycle}_factor{self.cycle_factor}_residue{n_residues_rounded}"
            # Create the directory if it doesn't exist
            dir_path = os.path.join(root_directory, mc_directory, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            # Form the Surpass command
            cmd = [
                self.surpass_path,
                "-r", str(n_residues_rounded),
                "-i", str(self.inner_cycle),
                "-o", str(self.outer_cycle),
                "-s", str(self.cycle_factor)
            ]
            # Run the Surpass command
            subprocess.run(cmd, cwd=dir_path, check=True)
            run_count += 1

    def generate_log_space(self, min_value, max_value, count):
        log_min = math.log10(min_value)
        log_max = math.log10(max_value)
        return np.logspace(log_min, log_max, num=count)

    def get_current_datetime_as_directory_name(self):
        # Get the current local date and time
        now = datetime.now()
        # Create a directory name using a safe format (YearMonthDay_HourMinuteSecond)
        # Example: "20231227_153045" for December 27, 2023 at 3:30:45 PM
        directory_name = now.strftime("%Y%m%d_%H%M%S")
        return directory_name

if __name__ == '__main__':
    gen = DataGenerator(
        surpass_path="/home/mohammad/bioshell_v4/BioShell/target/release/surpass",
        residue_length_min=10,
        residue_length_max=2001,
        inner_cycle=10,
        outer_cycle=1000,
        cycle_factor=10,
        num_of_uniq_residue_lengths=10,
        is_log=True
    )
    for _ in range(5):
        gen.run()