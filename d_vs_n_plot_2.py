import os
import re
import math
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt


class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def length_squared(self):
        return self.x**2 + self.y**2 + self.z**2

class DiffusionCalculator:
    @staticmethod
    def get_cm(file_path):
        # This function reads the file, skips the header, and converts the content to a list of Vec3 objects.
        # Assuming the file contains lines with x, y, z coordinates separated by whitespace
        # and that the first line is a header.
        cm_list = []
        with open(file_path, 'r') as file:
            next(file)  # Skip the header or the first line
            for line in file:
                parts = line.split()
                if len(parts) >= 3:
                    cm_list.append(Vec3(float(parts[0]), float(parts[1]), float(parts[2])))
        return cm_list

    @staticmethod
    def calculate_diffusion_coefficient(cm_positions):
        if cm_positions is None or len(cm_positions) < 2:
            raise ValueError("There must be at least two positions to calculate the diffusion coefficient.")

        sum_of_squared_displacements = 0

        for i in range(1, len(cm_positions)):
            displacement = cm_positions[i] - cm_positions[i - 1]
            sum_of_squared_displacements += displacement.length_squared()

        diffusion_coefficient = sum_of_squared_displacements / (6 * (len(cm_positions) - 1))

        return diffusion_coefficient

    @staticmethod
    def calculate_diffusion_coefficients(data_path, time_step):
        res_list = []
        diff_coeff_list = []

        directories = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

        for dir in directories:
            match = re.search(r'residue(\d+)', dir)
            if match:
                residue_length = int(match.group(1))
                res_list.append(residue_length)

                file_path = os.path.join(data_path, dir, 'cm.dat')
                cm_list = DiffusionCalculator.get_cm(file_path)

                diffusion_coefficient = DiffusionCalculator.calculate_diffusion_coefficient(cm_list)

                diff_coeff_list.append(diffusion_coefficient)

        return (res_list, diff_coeff_list)



def plot_diffusion_coefficients(data_path, time_step):
    try:
        res_len_list, diff_coeff_list = DiffusionCalculator.calculate_diffusion_coefficients(data_path, time_step)

        plt.scatter(res_len_list, diff_coeff_list)
        plt.xscale('log')
        plt.yscale('log')

        # Set the minimum range for x and y if needed
        plt.xlim(min(res_len_list), max(res_len_list))
        plt.ylim(min([dc for dc in diff_coeff_list if dc > 0]), max(diff_coeff_list))  # Avoid zero or negative values

        plt.xlabel('Residue Length')
        plt.ylabel('Diffusion Coefficient')
        plt.title('Diffusion Coefficient by Residue Length')
        plt.show()
    except Exception as ex:
        print(str(ex))

# You would call the function with the path to your data directory
plot_diffusion_coefficients("C:/git/rouse_data/mc010", time_step=1)