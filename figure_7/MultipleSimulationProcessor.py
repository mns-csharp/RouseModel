import os
import re

from LineChartAggregator import LineChartAggregator
from SimulationProcessor import SimulationProcessor

class MultipleSimulationProcessor:
    def __init__(self, root_directory):
        self.root_directory = root_directory
        self.all_intersection_data = []

    def process_all_simulations(self):
        # Compile the regular expression pattern for matching directory names
        dir_pattern = re.compile(r'^mc\d{3}$')

        # Iterate over all items in the root directory
        for item in os.listdir(self.root_directory):
            # Full path of the item
            item_path = os.path.join(self.root_directory, item)
            # Check if the item is a directory and matches the pattern
            if os.path.isdir(item_path) and dir_pattern.match(item):
                # Initialize a SimulationProcessor for the current directory
                print(r'Now processing simulation {item_path}')
                sim_processor = SimulationProcessor(item_path)
                # Process simulations in the current directory
                sim_processor.process_simulations()
                # Collect all intersection data
                self.all_intersection_data.extend(sim_processor.get_intersection_data())

    def get_all_intersection_data(self):
        # Return the collected all intersection data
        return self.all_intersection_data

if __name__ == '__main__':
    # root_dir = r'C:\git\rouse_data'
    root_dir = r'/home/mohammad/rouse_data'
    multi_sim_processor = MultipleSimulationProcessor(root_dir)
    multi_sim_processor.process_all_simulations()
    all_intersection_data = multi_sim_processor.get_all_intersection_data()

    # Initialize the LineChartAggregator
    chart_aggregator = LineChartAggregator()

    # Add each simulation's intersection data to the aggregator
    for x_inter, y_inter in all_intersection_data:
        # Assuming x_inter and y_inter are numpy arrays of the same length
        chart_aggregator.add_data(x_inter, y_inter)

    # Draw the aggregate plot
    chart_aggregator.plot('multiple_simulation_plot')