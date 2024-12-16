import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
loss_file_path = 'data_MoV2.csv'
time_file_path = 'time_MoV2.csv'

# Load data from the files
data = pd.read_csv(loss_file_path, header=None)
time_data = pd.read_csv(time_file_path, header=None)

# Extract tensor values (loss values) from the CSV and convert to floats
def extract_value(tensor_str):
    return float(tensor_str.split('(')[1].split(',')[0])

# Apply the extraction function to the dataframe for loss values
loss_values = data.iloc[0].apply(extract_value).tolist()

# Extract time values (assuming it's a single row in time_MoV2.csv)
time_values = time_data.iloc[0].tolist()

# Ensure the length of time and loss values match
if len(loss_values) != len(time_values):
    print(f"Warning: Length mismatch between loss and time data. Losses: {len(loss_values)}, Time: {len(time_values)}")
else:
    # Plot the curve with time on the x-axis and loss on the y-axis
    plt.figure(figsize=(8, 6))
    plt.plot(time_values, loss_values, marker='o', linestyle='-', color='b', markersize=5)

    # Set title and labels
    plt.title("Loss vs Time")
    plt.xlabel("Time")
    plt.ylabel("Loss_R50")

    # Enable grid
    plt.grid(True)

    # Display the plot
    plt.show()
