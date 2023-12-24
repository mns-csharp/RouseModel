import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt

from MultipleSimulationProcessor import MultipleSimulationProcessor


class Main:
    def __init__(self, processor):
        self.processor = processor

    def draw_aggregate_chart(self):
        try:
            self.processor.process_all_simulations()

            y_list_of_lists = []

            x_values = self.processor.x_list
            # y_list_of_lists = self.processor.y_lists
            mean_y_list = self.processor.mean_list
            std_dev_y_list = self.processor.stddev_list

            y_list_of_lists.append(mean_y_list)
            y_list_of_lists.append(std_dev_y_list)

            # Set the titles and axis labels
            plt.title("Multiple Line Graphs Example")
            plt.xlabel("X Axis")
            plt.ylabel("Y Axis")

            # Define a list of colors
            colors = ['b', 'g', 'r', 'm', 'orange', 'brown']

            # Define a list of line styles
            line_styles = ['-', '--', ':', '-.', (0, (3, 5, 1, 5))]

            for i, y_values in enumerate(y_list_of_lists):
                # Use modular arithmetic to loop through colors and line_styles
                color = colors[i % len(colors)]
                line_style = line_styles[i % len(line_styles)]

                # Generate a line with a unique color and line style
                plt.plot(x_values, y_values, color=color, linestyle=line_style, linewidth=2, marker='o', markersize=7)

            plt.savefig('figure_7_aggregare_chart.png');
            plt.show()

        except Exception as ex:
            print(str(ex))


def main():
    processor = MultipleSimulationProcessor(r'C:\git\rouse_data')
    form = Main(processor)
    form.draw_aggregate_chart()


if __name__ == "__main__":
    main()