import matplotlib

matplotlib.use("QT5Agg")
import matplotlib.pyplot as plt

from MultipleSimulationProcessor import MultipleSimulationProcessor

import numpy as np  # Import numpy for regression line calculation

# Assume the rest of your required modules and MultipleSimulationProcessor class are defined elsewhere

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np


class Main:
    def __init__(self, processor):
        self.processor = processor

    def draw_aggregate_chart(self):
        try:
            self.processor.process_all_simulations()

            # Ensure x_values is a NumPy array
            x_values = np.array(self.processor.x_list)
            y_values = self.processor.y_lists
            mean_y_list = self.processor.mean_list
            std_dev_y_list = self.processor.stddev_list

            # Set the titles and axis labels
            plt.title("Mean, StdDev, and Individual Simulations Plot")
            plt.xlabel("X Axis = N")
            plt.ylabel("Y Axis = τ")

            # Plot individual simulation lines
            for y_list in y_values:
                plt.loglog(
                    x_values,
                    y_list,
                    color="grey",
                    alpha=0.5,
                    linestyle="-",
                    linewidth=1,
                    marker="s",
                )  # Individual lines

            # Plotting the mean and standard deviation lines
            plt.loglog(
                x_values,
                mean_y_list,
                label="Mean",
                color="red",
                linestyle="--",
                linewidth=2,
                marker="o",
                markersize=7,
            )
            plt.loglog(
                x_values,
                std_dev_y_list,
                label="StdDev",
                color="magenta",
                linestyle="-",
                linewidth=2,
                marker="o",
                markersize=7,
            )

            # Calculate the regression line for the mean values
            coeffs = np.polyfit(
                np.log(x_values), np.log(mean_y_list), 1
            )  # Linear fit on the log-log scale
            regression_line = np.poly1d(coeffs)
            reg_y_values = np.exp(
                regression_line(np.log(x_values))
            )  # Convert back to linear space

            # Plot the regression line
            plt.loglog(
                x_values,
                reg_y_values,
                label="Mean Regression",
                color="blue",
                linestyle="-",
                linewidth=1,
            )

            # Write the slope of the regression line on the plot
            slope_text = f"Slope of mean regression: {coeffs[0]:.2f}"
            plt.text(
                0.05,
                0.95,
                slope_text,
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment="top",
            )

            # Plot the reference line with slope 2.0 that intersects the y-values
            # Using the first point (x_values[0], mean_y_list[0]) as the reference point
            x0, y0 = x_values[0], mean_y_list[0]
            # Calculate the y-values for the reference line
            ref_y_values = y0 * (x_values / x0) ** 2.0

            # Plot the reference line
            plt.loglog(
                x_values,
                ref_y_values,
                label="Slope=2 Ref Line",
                color="black",
                linestyle=":",
                linewidth=2,
            )

            plt.legend()  # Display a legend
            plt.savefig("aggregate_chart.png")
            plt.show()

        except Exception as ex:
            print(str(ex))


# Assuming processor is an instantiated object of a class that has the required attributes
# processor = SomeProcessorClass()
# main_instance = Main(processor)
# main_instance.draw_aggregate_chart()


def main():
    processor = MultipleSimulationProcessor(r"C:\git\rouse_data~~")
    # processor = MultipleSimulationProcessor(r'/home/mohammad/rouse_data')
    form = Main(processor)
    form.draw_aggregate_chart()


if __name__ == "__main__":
    main()
