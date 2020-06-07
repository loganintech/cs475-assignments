import matplotlib.pyplot as plt

x = []
y = []
with open("./data.txt") as f:
    for line in f.readlines():
        split = line.split(":")
        x.append(float(split[0]))
        y.append(float(split[1]))

plt.gcf().subplots_adjust(left=0.15)

plt.xlabel("Shift")
plt.ylabel("Sums")
plt.title("Sums[*] vs Shift")
plt.scatter(x, y)
plt.savefig("sums_vs_shift.png")
