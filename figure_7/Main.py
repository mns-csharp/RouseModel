import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt

from MultipleSimulationProcessor import MultipleSimulationProcessor

import numpy as np  # Import numpy for regression line calculation

# Assume the rest of your required modules and MultipleSimulationProcessor class are defined elsewhere

class Main:
    def __init__(self, processor):
        self.processor = processor

    def draw_aggregate_chart(self):
        try:
            self.processor.process_all_simulations()

            x_values = self.processor.x_list
            mean_y_list = self.processor.mean_list
            std_dev_y_list = self.processor.stddev_list

            # Set the titles and axis labels
            plt.title("Mean and StdDev Plot")
            plt.xlabel("X Axis = N")
            plt.ylabel("Y Axis = τ")

            # Plotting the mean and standard deviation lines
            plt.loglog(x_values, mean_y_list, label='Mean', color='r', linestyle='--', linewidth=2, marker='o', markersize=7)
            plt.loglog(x_values, std_dev_y_list, label='StdDev', color='m', linestyle='-', linewidth=2, marker='o', markersize=7)

            # Calculate the regression line for the mean values
            coeffs = np.polyfit(np.log(x_values), np.log(mean_y_list), 1)  # Linear fit on the log-log scale
            regression_line = np.poly1d(coeffs)
            reg_y_values = np.exp(regression_line(np.log(x_values)))  # Convert back to linear space

            # Plot the regression line
            plt.loglog(x_values, reg_y_values, label='Mean Regression', color='blue', linestyle='-', linewidth=1)

            # Write the slope of the regression line on the plot
            slope_text = f"Slope of mean regression: {coeffs[0]:.2f}"
            plt.text(0.05, 0.95, slope_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

            plt.legend()  # Display a legend
            plt.savefig('figure_7_aggregare_chart.png')
            plt.show()

        except Exception as ex:
            print(str(ex))


def main():
    processor = MultipleSimulationProcessor(r'C:\git\rouse_data')
    # processor = MultipleSimulationProcessor(r'/home/mohammad/rouse_data')
    form = Main(processor)
    form.draw_aggregate_chart()


if __name__ == "__main__":
    main()