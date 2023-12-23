import os
import re
from R2AutocorrelationProcessor import R2AutocorrelationProcessor

class SingleSimulationProcessor:
    dir_pattern = re.compile(r'run\d+_inner\d+_outer\d+_factor\d+_residue\d+', re.IGNORECASE)

    def __init__(self, root_directory):
        self.root_directory = root_directory
        self.residue_lengths = []
        self.rouse_relaxation = []
        self.sub_directories = self.get_r2_directories()
        self.process_simulations()

    def get_r2_directories(self):
        return [dir_path for dir_path in os.listdir(self.root_directory)
                if os.path.isdir(os.path.join(self.root_directory, dir_path))
                and self.dir_pattern.match(dir_path)]

    def process_simulations(self):
        for dir_path in self.sub_directories:
            full_dir_path = os.path.join(self.root_directory, dir_path)
            print(f"Now processing residue: {full_dir_path}")
            try:
                r2_processor = R2AutocorrelationProcessor(full_dir_path)

                if r2_processor.intersection is not None:
                    self.residue_lengths.append(r2_processor.residue_length)
                    self.rouse_relaxation.append(r2_processor.intersection)
            except FileNotFoundError:
                print(f"r2.dat file not found in directory: {full_dir_path}")
            except (ValueError, OverflowError) as ex:
                print(str(ex))

# Example usage:
# processor = SingleSimulationProcessor('/path/to/simulations')
# print(processor.residue_lengths)
# print(processor.rouse_relaxation)