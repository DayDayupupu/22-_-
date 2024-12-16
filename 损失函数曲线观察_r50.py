import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'data_MoV2.csv'
data = pd.read_csv(file_path, header=None)

# Extract tensor values from the CSV and convert to floats
# Assuming tensors are stored as strings like "tensor(0.6367, device='cuda:0')"
def extract_value(tensor_str):
    return float(tensor_str.split('(')[1].split(',')[0])

# Apply the extraction function to the dataframe
values = data.iloc[0].apply(extract_value).tolist()

# Plot the curve
iterations = range(0, len(values) )
plt.figure(figsize=(8, 6))
plt.plot(iterations, values, marker='o', linestyle='-', color='b', markersize=5)  # smaller points

# Set title and labels
plt.title("Loss vs MoV2")
plt.xlabel("Epochs")
plt.ylabel("Loss_MoV2")

# Set x-axis ticks at intervals of 10
tick_interval = 10
plt.xticks(range(0, len(values) , tick_interval))

# Enable grid
plt.grid(True)

# Display the plot
plt.show()
