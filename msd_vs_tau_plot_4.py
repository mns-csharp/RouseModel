# File: msd_vs_tau_plot.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def read_file_to_vec3(filename):
    vecs = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) != 3:
                # Handle error or ignore the line
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                vecs.append(Vec3(x, y, z))
            except ValueError:
                # Handle error or ignore the line
                continue
    return vecs

def calculate_msd(positions, initial_position_index):
    msd_data = []
    if not positions or initial_position_index >= len(positions):
        return msd_data

    initial_position = positions[initial_position_index]

    for pos in positions:
        squared_displacement = ((pos.x - initial_position.x) ** 2 +
                                (pos.y - initial_position.y) ** 2 +
                                (pos.z - initial_position.z) ** 2)
        msd_data.append(squared_displacement)

    return msd_data

def write_data_to_file(filename, tau_data, msd_data):
    if len(tau_data) != len(msd_data):
        print("Error: The size of tauData and msdData must match.")
        return

    with open(filename, 'w') as out_file:
        for i in range(len(tau_data)):
            out_file.write(f"{tau_data[i]}\t{msd_data[i]}\n")

def main():
    filename = r"C:\git\rouse_data\mc010\run0_inner10_outer1000_factor10_residue50\cm.dat"

    vectors = read_file_to_vec3(filename)

    initial_position_index = 0

    msd_data = calculate_msd(vectors, initial_position_index)

    tau_data = [float(i) for i in range(len(msd_data))]

    tau_data_excluding_first = tau_data[1:]
    msd_data_excluding_first = msd_data[1:]

    write_data_to_file("tau_vs_msd.txt", tau_data_excluding_first, msd_data_excluding_first)

    print("done writing!")

    try:
        plt.loglog(tau_data_excluding_first, msd_data_excluding_first)
        plt.xlabel("x-axis-label")
        plt.ylabel("y-axis-label")
        plt.title("Plot Graph")

        # Define the filename for the image
        image_filename = "msd_vs_tau_plot.png"

        # Save the figure as a PNG file
        plt.savefig(image_filename)
        plt.close()

        print(f"Plot saved as {image_filename}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()