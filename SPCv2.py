import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import date

def calculate_control_limits(data, confidence_level=0.99):
    """
    Calculate the control limits for a given dataset using the confidence level.
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    z_score = norm.ppf((1 + confidence_level) / 2)
    upper_limit = mean + (z_score * std_dev)
    lower_limit = mean - (z_score * std_dev)
    return upper_limit, lower_limit

def check_out_of_control(data_point, upper_limit, lower_limit):
    """
    Check if a data point is out of control based on the control limits.
    """
    return data_point > upper_limit or data_point < lower_limit

def calculate_control_limits(data, num_samples, confidence_level=0.99):
    """
    Calculate the control limits for a given dataset using the confidence level.
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    z_score = norm.ppf((1 + confidence_level) / 2)
    upper_limit = mean + (z_score * (std_dev / np.sqrt(num_samples)))
    lower_limit = mean - (z_score * (std_dev / np.sqrt(num_samples)))
    return upper_limit, lower_limit

def plot_control_chart(df, upper_limit, lower_limit):
    """
    Plot the control chart with the data points and control limits.
    """
#    num_samples = len(data)
#   sample_indices = np.arange(1, num_samples+1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df["Timestamp"], df["Data"], 'bo-', label='Data') 
    plt.axhline(upper_limit, color='r', linestyle='--', label='Upper Control Limit')
    plt.axhline(lower_limit, color='r', linestyle='--', label='Lower Control Limit')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xlabel('Timestamp')
    plt.ylabel('Max number of simultaneous GCUs')
    plt.title('Cloud resource usage')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example data points (e.g., production measurements)
data = [10, 12, 11, 9, 10, 11, 10, 12, 51, 8, 9, 11, 10, 11]
timestamps = pd.date_range(start='00:00:00', periods=len(data), freq='5min')
df = pd.DataFrame({'Timestamp': timestamps, 'Data': data})

# Control chart parameters
confidence_level = 0.99
sample_size = 1  # Number of data points in each sample

# Initialize lists to store control limits and out-of-control points
upper_limits = []
lower_limits = []
out_of_control_points = []

# Iterate over samples of the data
for i in range(sample_size, len(data) + 1):
    sample_data = data[i-sample_size:i]  # Extract the current sample
    
    # Calculate control limits for the current sample
    upper_limit, lower_limit = calculate_control_limits(sample_data, sample_size, confidence_level)
    
    # Check if each data point in the sample is out of control
    out_of_control = [data_point for data_point in sample_data if check_out_of_control(data_point, upper_limit, lower_limit).any()]
    
    # Append control limits and out-of-control points to the respective lists
    upper_limits.append(upper_limit)
    lower_limits.append(lower_limit)
    out_of_control_points.extend(out_of_control)

# Plot the control chart
plot_control_chart(df, upper_limit, lower_limit)

# Print the out-of-control data points
print("Control Limits: ", lower_limit, upper_limit)
print("Out-of-Control Data Points: ", out_of_control_points)