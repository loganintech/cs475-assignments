import json
import matplotlib.pyplot as plt
from matplotlib.scale import LogScale
import sys

args = sys.argv[1:]
data = []
with open(args[0], "r") as f:
    data = json.loads(f.read())


plt.ylabel("Speedup")
plt.xlabel("Array Size - Bytes")
plt.title(" ".join(args[5:]))

array_size = []
control_times = []
simd_times = []
for case in data:
    control_times.append(case["control_time"])
    simd_times.append(case["simd_time"])
    array_size.append(case["array_size"])

print(array_size)
print(control_times)
print(simd_times)

speedup = list(map(lambda a: a[0] / a[1], zip(control_times, simd_times)))

plt.plot(array_size, speedup, 'b')
# plt.plot(array_size, simd_times, 'b')
plt.xscale("log")
# plt.yscale("log")

# print(control_times)
# print(array_size)

for (x, y) in zip(array_size, speedup):
    plt.annotate("{:0.2f}".format(y), xy=(x+0.3, y-0.01))

# plt.show()
plt.savefig("simd_speedup.png")
