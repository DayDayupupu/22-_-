import matplotlib.pyplot as plt

# Data from the user
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27, 28, 29, 30]
loss_avg = [
    0.631537, 0.522883, 0.444081, 0.380033, 0.326202, 0.278359, 0.240543,
    0.210978, 0.186666, 0.163607, 0.147296, 0.133167, 0.116318, 0.106477,
    0.098742, 0.089524, 0.082378, 0.075404, 0.070171, 0.069482, 0.063825,
    0.055996, 0.053349, 0.051258, 0.046397, 0.045281, 0.044254, 0.043726
]
time_out = [
    179.359134, 843.359134, 1789.300817, 2627.992800, 3467.058779, 4299.375260,
    5062.282774, 5826.289459, 6952.924934, 8087.558502, 8859.931148, 9848.352268,
    10590.005724, 11495.419901, 12852.639229, 13666.798587, 14486.851452,
    15180.962241, 15890.557508, 16598.448811, 17304.497486, 18717.800266,
    19423.711903, 20129.920232, 21543.588224, 22249.990606, 22955.687811,
    24087.558502
]

# Plot Loss vs Epochs
plt.figure(figsize=(14, 6))

# Loss vs Epoch
plt.subplot(1, 2, 1)
plt.plot(epochs, loss_avg, marker='o', label="Loss")
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

# Loss vs Time
plt.subplot(1, 2, 2)
plt.plot(time_out, loss_avg, marker='o', label="Loss")
plt.title("Loss vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
