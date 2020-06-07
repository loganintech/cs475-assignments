import matplotlib.pyplot as plt

x = []
y = []
with open("./performance.txt") as f:
    for line in f.readlines():
        split = line.split(":")
        x.append(split[0].replace(" ", "\n"))
        y.append(float(split[1]))

plt.gcf().subplots_adjust(bottom=0.15)
plt.xlabel("Method")
plt.ylabel("MegaMults")
plt.title("MegaMults vs Multiplication Method")
plt.bar(x, y)
plt.savefig("speedup_vs_method.png")
