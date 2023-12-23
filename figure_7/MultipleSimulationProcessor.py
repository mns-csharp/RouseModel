import os
import re
from SingleSimulationProcessor import SingleSimulationProcessor
from LineChartAggregator import LineChartAggregator

class MultipleSimulationProcessor:
    def __init__(self, root_directory):
        self._root_directory = root_directory
        self.x_list = None
        self.y_lists = None
        self.mean_list = None
        self.stddev_list = None

    def process_all_simulations(self):
        dir_pattern = re.compile(r'^mc\d{3}$')

        agg = LineChartAggregator()

        directories = [os.path.join(self._root_directory, d) for d in os.listdir(self._root_directory)
                       if os.path.isdir(os.path.join(self._root_directory, d))]

        for item in directories:
            if dir_pattern.match(os.path.basename(item)):
                print(f"Now processing simulation {item}")
                sim_processor = SingleSimulationProcessor(item)

                x_list = sim_processor.residue_lengths
                y_list = sim_processor.rouse_relaxation

                if x_list is not None and y_list is not None:
                    if len(x_list) != 0 and len(y_list) != 0:
                        if len(x_list) == len(y_list):
                            agg.add_data(x_list, y_list)

        agg.process_data()

        self.x_list = agg.get_x_common()
        self.y_lists = agg.get_y_interpolations()
        self.mean_list = agg.calculate_mean()
        self.stddev_list = agg.calculate_std_dev()

# Example usage:
# multiple_processor = MultipleSimulationProcessor('/path/to/multiple/simulations')
# multiple_processor.process_all_simulations()
# print(multiple_processor.x_list)
# print(multiple_processor.y_lists)
# print(multiple_processor.mean_list)
# print(multiple_processor.stddev_list)