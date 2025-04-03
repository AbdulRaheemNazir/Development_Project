# Import the numpy library for numerical operations, especially useful for generating random numbers.
import numpy as np

# Import the datetime and timedelta classes from the datetime module.
# - datetime: for working with dates and times.
# - timedelta: for representing a duration, e.g., adding minutes to a time.
from datetime import datetime, timedelta

# Define a function named 'generate_synthetic_glucose_data' that creates fake glucose data.
# This function simulates a list of glucose measurements over time.
def generate_synthetic_glucose_data(length=100, seed=42):
    # Set the seed for numpy's random number generator to ensure the results are the same every time.
    np.random.seed(seed)
    
    # Define a base glucose value from which we start. Here, it's set to 110.
    base_glucose = 110
    
    # Create an empty list called 'data' where each generated data point will be stored.
    data = []
    
    # Get the current date and time, which will be the starting point for our data.
    current_time = datetime.now()

    # Loop 'length' times to generate the specified number of data points.
    for i in range(length):
        # Generate random noise from a normal distribution (mean 0, standard deviation 8).
        # This simulates small, natural fluctuations in glucose levels.
        noise = np.random.normal(0, 8)
        
        # Add the noise to the base glucose value to get the current glucose reading.
        glucose = base_glucose + noise

        # Every 20th data point, simulate a larger change (like a meal or exercise)
        # by randomly adding or subtracting 30 from the glucose value.
        if i % 20 == 0:
            glucose += np.random.choice([-30, 30])

        # Calculate the time for the current data point by adding i*5 minutes to the starting time.
        # The result is formatted as a string in "Year-Month-Day Hour:Minute:Second" format.
        time_str = (current_time + timedelta(minutes=5 * i)).strftime("%Y-%m-%d %H:%M:%S")
        
        # Append a tuple (pair) of the time string and the rounded glucose value to the data list.
        # The glucose value is rounded to two decimal places.
        data.append((time_str, round(glucose, 2)))

    # Return the complete list of generated data points.
    return data
