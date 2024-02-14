from framework.ComputeCForRange import ComputeCForRange
from framework.read_vec3 import read_vec3

if __name__ == "__main__":
    r_end_vec = r"framework\test_data_2.txt"
    dir1 = "."
    chain1Vec3List = read_vec3(dir_path=dir1, file_name=r_end_vec)
    print(chain1Vec3List)
    lags, autos = ComputeCForRange(chain1Vec3List, 3)
    print(lags)
    print(autos)

