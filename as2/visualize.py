import json
import matplotlib.pyplot as plt
from matplotlib.scale import LogScale
import sys

args = sys.argv[1:]
data = []
with open(args[0], "r") as f:
    data = json.loads(f.read())

yprop = args[1]
xprop = args[2]
ylabel = args[3]
xlabel = args[4]

plt.ylabel(ylabel)
plt.xlabel(xlabel)
plt.title(" ".join(args[5:]))

yval = []
xval = []
for case in data:
    xval.append(case[xprop])
    yval.append(case[yprop])

plt.plot(xval, yval, 'r')
# plt.xscale("log")

print(xval)
print(yval)

for (x, y) in zip(xval, yval):
    plt.annotate("{:0.2f}".format(y), xy=(x+0.3, y))


plt.savefig(f"{args[0]}.png")
