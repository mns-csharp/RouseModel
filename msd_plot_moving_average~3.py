import os
import matplotlib
from scipy.signal import savgol_filter

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def wrap_around(value):
    if value > 1000:
        return value - 2000
    elif value < -1000:
        return value + 2000
    else:
        return value

def moving_average(data, window_size):
    return [sum(data[i:i+window_size])/window_size for i in range(len(data)-window_size+1)]

def calculate_msd(data, wrap=False):
    msd_values = []
    initial_point = data[0]
    for point in data:
        delta = [(wrap_around(a - b) if wrap else a - b)**2 for a, b in zip(point, initial_point)]
        msd = sum(delta)
        msd_values.append(msd)
    return msd_values

def denoise_data(data, window_length, polyorder):
    denoised_data = []
    for dimension in zip(*data):
        dimension = savgol_filter(dimension, window_length, polyorder)
        denoised_data.append(dimension)
    return list(zip(*denoised_data))

def generate_msd_plots(source_dir, destination_dir,
                       wrap=False, smooth=False, denoise=False, window_size=1000, window_length=11,
                       polyorder=2):
    index = 0
    for subdir, dirs, files in os.walk(source_dir):
        for file in files:
            if file == "cm.dat":
                # print(f"Processing file: {os.path.join(subdir, file)}")  # print the file name
                data = []
                with open(os.path.join(subdir, file), 'r') as f:
                    next(f)  # skip header row
                    for line in f:
                        x, y, z = map(float, line.strip().split())
                        data.append([x, y, z])
                if denoise:
                    data = denoise_data(data, window_length, polyorder)
                msd_values = calculate_msd(data, wrap)
                if smooth:
                    msd_values = moving_average(msd_values, window_size)
                plt.figure()
                plt.loglog(range(len(msd_values)), msd_values)
                plt.xlabel("MC-steps")
                plt.ylabel(f"{'Smoothed ' if smooth else ''}{'Denoised ' if denoise else ''}MSD")
                plt.title(f"{'Smoothed ' if smooth else ''}{'Denoised ' if denoise else ''}MSD vs MC-steps (Log-Log Scale){', No Wrap' if not wrap else ''}")
                subdir_name = os.path.basename(subdir)
                destination_path = os.path.join(destination_dir, subdir_name)
                if not os.path.exists(destination_path):
                    os.makedirs(destination_path)
                output_file_name = f"{'smooth' if smooth else 'no_smooth'}{'_denoised' if denoise else '_noisy'}{'_wrap' if wrap else '_no_wrap'}_ws{window_size}_wl{window_length}_po{polyorder}.png"
                output_file = f"{destination_path}/{output_file_name}.png"
                plt.savefig(output_file)
                print(f"Output file: {index}: {output_file_name}")  # print the name of the output PNG file
                index = index + 1
                plt.close()

def main():
    source_dir = r"C:\Users\pc\Documents\__protein design\SURPASS\Rouse Model\mc004"  # Add the path to your source directory
    destination_dir = r"C:\Users\pc\Documents\__protein design\SURPASS\Rouse Model\mc004\msd"  # Add the path to your destination directory

    conditions = {
        'no_wrap': False,
        'wrap': True,
    }

    denoising = {
        'no_denoise': False,
        'denoise': True,
    }

    smoothing = {
        ### 'no_smooth': False,
        'smooth': True,
    }

    # Define the range of values for each parameter
    window_sizes = [5, 750, 1500]
    window_lengths = [11, 21, 31]
    polyorders = [1, 2, 3]

    # Nested loops to iterate over all combinations of parameters
    for wrap in conditions.values():
        for denoise in denoising.values():
            for smooth in smoothing.values():
                for window_size in window_sizes:
                    for window_length in window_lengths:
                        for polyorder in polyorders:
                            generate_msd_plots(source_dir, destination_dir,
                                               wrap=wrap, smooth=smooth, denoise=denoise,
                                               window_size=window_size, window_length=window_length,
                                               polyorder=polyorder)

if __name__ == "__main__":
    main()