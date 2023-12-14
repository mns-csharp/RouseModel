import math
import os
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt

import os

class Vec3:
    def __init__(self, x, y, z):
        self.X = x
        self.Y = y
        self.Z = z

def MSD(centerOfMassPositions, steps):
    sumSquaredDistance = 0.0

    for j in range(len(centerOfMassPositions) - steps):
        dx = centerOfMassPositions[j + steps].X - centerOfMassPositions[j].X
        dy = centerOfMassPositions[j + steps].Y - centerOfMassPositions[j].Y
        dz = centerOfMassPositions[j + steps].Z - centerOfMassPositions[j].Z
        sumSquaredDistance += dx * dx + dy * dy + dz * dz

    msd = sumSquaredDistance / (len(centerOfMassPositions) - steps)

    return msd

def CalculateMSD(centerOfMassPositions):
    msdValues = []
    for i in range(len(centerOfMassPositions)):
        msd = MSD(centerOfMassPositions, i)
        msdValues.append(msd)
    return msdValues

def GetCM(filePath):
    cmList = []
    with open(filePath, 'r') as file:
        for line in file:
            parts = line.split() # Assuming the data is whitespace-separated
            if len(parts) == 3:
                x, y, z = map(float, parts)
                cmList.append(Vec3(x, y, z))
    return cmList

def main():
    filePath = "C:/git/RouseModelCSharp/sim_data/cm.dat"
    cmlist = GetCM(filePath)
    msdList = CalculateMSD(cmlist)
    indexList = [i for i in range(len(msdList))]

    plt.figure(figsize=(19.20, 10.80)) # Size in inches

    plt.plot(indexList, msdList)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(left=1)
    plt.ylim(bottom=1)

    directoryPath = os.path.dirname(filePath)
    chartFileName = "chart.png"
    chartFilePath = os.path.join(directoryPath, chartFileName)

    plt.show()

if __name__ == "__main__":
    main()