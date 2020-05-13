import json
import matplotlib.pyplot as plt
from matplotlib.scale import LogScale
import sys

args = sys.argv[1:]
data = []
with open(args[0], "r") as f:
    data = json.loads(f.read())

grain = []
deer = []
months = []
precip = []
temp = []
pandemic = []
for case in data:
    grain.append(case["grain_height"])
    deer.append(case["deer_count"])
    months.append(((case["year"] - 2020) * 12) + case["month"])
    precip.append(case["precipitation"])
    temp.append(case["temperature"])
    pandemic.append(case["is_pandemic"])


def gen_pic():
    plt.title("Visualization Data")
    temp = list(map(lambda x: (5.0/9.0)*(x-32), temp))

    # print(months)
    # print(grain, deer, months, precip, temp)

    plt.title("Simulation Data")
    plt.plot(months, grain, label="Grain Height Inches")
    plt.plot(months, deer, label="Deer Population")
    plt.plot(months, precip, label="Precipitation Inches")
    plt.plot(months, temp, label="Temperature °C")
    plt.plot(months, pandemic, label="Pandemic Status")
    plt.legend(loc="upper right")
    plt.savefig(f"visualized.png")
    # plt.show()


def write_table():
    sep = '\t'
    print(
        f"Month{sep}Grain Height{sep}Deer Count{sep}Precipitation{sep}Temperature{sep}Pandemic Status")
    for month, grn, dr, prec, temperature, pand in zip(months, grain, deer, precip, temp, pandemic):
        print(f"{month}{sep}{grn}{sep}{dr}{sep}{prec}{sep}{temperature}{sep}{pand}")


if __name__ == "__main__":
    if args[1] == None:
        gen_pic()
    else:
        write_table()
