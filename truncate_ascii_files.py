import os

def is_ascii(file_path):
    try:
        with open(file_path, 'r', encoding='ascii') as file:
            # Try reading a small portion of the file
            file.read(1024)
        return True
    except UnicodeDecodeError:
        # If a UnicodeDecodeError occurs, it's not an ASCII text file
        return False
    except Exception:
        # For other exceptions, we'll assume it's not an ASCII text file
        return False

def truncate_file_to_21_lines(file_path):
    with open(file_path, 'r', encoding='ascii', errors='ignore') as file:
        lines = file.readlines()

    with open(file_path, 'w', encoding='ascii') as file:
        file.writelines(lines[:21])

def scan_and_truncate_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if is_ascii(file_path):
                try:
                    truncate_file_to_21_lines(file_path)
                    print(f"Processed file: {file_path}")
                except Exception as e:
                    print(f"Failed to process file: {file_path}. Error: {e}")

# Replace 'your_directory_path' with the path to the directory you want to scan
your_directory_path = r'C:\git\rouse_data'
scan_and_truncate_files(your_directory_path)

