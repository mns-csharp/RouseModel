import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class LineChartAggregator:
    def __init__(self):
        self.x_lists = []
        self.y_lists = []

    def add_data(self, x_list, y_list):
        if len(x_list) != len(y_list):
            raise ValueError("x_list and y_list must be of the same length.")
        self.x_lists.append(x_list)
        self.y_lists.append(y_list)

    def get_mean_and_stddev_points(self):
        # Determine the common x range for interpolation
        x_min = max(min(x) for x in self.x_lists)
        x_max = min(max(x) for x in self.x_lists)
        common_x = np.linspace(x_min, x_max, num=100)  # You can adjust the number of points

        # Interpolate or extrapolate y values for all lists
        y_interp_lists = []
        for x_list, y_list in zip(self.x_lists, self.y_lists):
            f = interp1d(x_list, y_list, kind='linear', fill_value='extrapolate')
            y_interp_lists.append(f(common_x))

        # Calculate mean and standard deviation
        y_stacked = np.vstack(y_interp_lists)
        y_mean = np.mean(y_stacked, axis=0)
        y_std = np.std(y_stacked, axis=0)

        return common_x, y_mean, y_std

    def plot(self):
        common_x, y_mean, y_std = self.get_mean_and_stddev_points()

        # Plot the mean line chart
        plt.plot(common_x, y_mean, label='Mean')

        # Plot the standard deviation area
        plt.fill_between(common_x, y_mean - y_std, y_mean + y_std, alpha=0.2, label='Std Dev')

        # Enhance the plot with labels, title, and legend
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Mean and Standard Deviation of Line Charts')
        plt.legend()

        # Show the plot
        plt.show()


if __name__ == '__main__':
    # Example usage:
    chart_aggregator = LineChartAggregator()

    # Add data (replace these with your actual data)
    chart_aggregator.add_data(np.arange(11), np.random.normal(loc=0, scale=1, size=11))
    chart_aggregator.add_data(np.arange(17), np.random.normal(loc=0, scale=1, size=17))
    chart_aggregator.add_data(np.arange(23), np.random.normal(loc=0, scale=1, size=23))

    # Draw the plot
    chart_aggregator.plot()