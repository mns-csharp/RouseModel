import numpy as np

class LineChartAggregator:
    def __init__(self):
        self.x_list_common = []
        self.original_lists = []
        self.y_interpolated = []

    def add_data(self, x_list, y_list):
        if x_list and y_list and len(x_list) == len(y_list):
            self.original_lists.append((x_list, y_list))

    def process_data(self):
        # Flatten the list of x values and get the unique values in sorted order
        all_x_values = sorted(set([x for x_list, _ in self.original_lists for x in x_list]))
        self.x_list_common = all_x_values

        for x_list, y_list in self.original_lists:
            y_interpolated_list = np.interp(self.x_list_common, x_list, y_list)
            self.y_interpolated.append(y_interpolated_list)

    def calculate_mean(self):
        # Calculate the mean across all y_interpolated lists for each x value
        mean_values = np.mean(self.y_interpolated, axis=0)
        return mean_values.tolist()

    def calculate_std_dev(self):
        # Calculate the standard deviation across all y_interpolated lists for each x value
        std_dev_values = np.std(self.y_interpolated, axis=0, ddof=0)
        return std_dev_values.tolist()

    def get_y_interpolations(self):
        return self.y_interpolated

    def get_x_common(self):
        return self.x_list_common