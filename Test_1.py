# Import libraries needed for generating data and making predictions
import numpy as np                      # Used for math operations and generating random numbers
import pandas as pd                    # Used for working with tables of data
from datetime import datetime, timedelta  # Used to work with dates and times
from statsmodels.tsa.arima.model import ARIMA  # This library provides ARIMA forecasting

# -----------------------------
# FUNCTION 1: Generate Synthetic Glucose Data
# -----------------------------

# This function simulates fake glucose readings over time
def generate_synthetic_glucose_data(length=100, seed=42):
    np.random.seed(seed)  # Set the seed so the random numbers are the same every time (for repeatability)
    base_glucose = 105    # Start with a base glucose value of 105 mg/dL
    data = []             # Create an empty list to store the glucose readings
    current_time = datetime.now()  # Get the current time to use as the starting timestamp

    # Loop as many times as needed to create the data
    for i in range(length):
        noise = np.random.normal(0, 8)  # Generate small random noise to simulate natural variation
        glucose = base_glucose + noise  # Add the noise to the base value to get the simulated glucose reading

        # Every 20 steps, simulate a sudden rise or fall, like after eating or exercising
        if i % 20 == 0:
            glucose += np.random.choice([-30, 30])  # Randomly add or subtract 30

        # Create a timestamp that increases by 5 minutes for each new reading
        time_str = (current_time + timedelta(minutes=5 * i)).strftime("%Y-%m-%d %H:%M:%S")

        # Save the timestamp and glucose value as a pair in the list
        data.append((time_str, round(glucose, 2)))  # Round glucose to 2 decimal places

    return data  # Return the full list of time + glucose readings

# -----------------------------
# FUNCTION 2: Run Rolling ARIMA Forecast
# -----------------------------

# This function uses past glucose data to predict future values using ARIMA
def run_rolling_arima(glucose_data, num_predictions, min_past_values=5):
    # If there’s not enough data to make predictions, return nothing
    if len(glucose_data) < num_predictions:
        print(f"Not enough glucose data for {num_predictions}-step rolling prediction.")
        return [], None

    predictions = []  # List to store future predicted values
    past_values = []  # List to hold past glucose values used to train the model
    best_order = (1, 1, 0)  # ARIMA model configuration: p=1, d=1, q=0

    # Extract only the glucose values (not timestamps) from the input data
    for i in range(len(glucose_data)):
        _, original_glucose = glucose_data[i]  # Ignore time, keep glucose value
        past_values.append(original_glucose)   # Add it to our list of values

    # Loop to create each future prediction, one at a time
    for _ in range(num_predictions):
        if len(past_values) >= min_past_values:
            try:
                model = ARIMA(past_values, order=best_order)  # Create ARIMA model
                model_fit = model.fit()                       # Train the model
                forecast = model_fit.get_forecast(steps=1)    # Ask for 1-step forecast
                predicted_glucose = forecast.predicted_mean[0]  # Get the number prediction
            except Exception as e:
                print(f"Rolling ARIMA prediction failed: {e}")
                predicted_glucose = None
        else:
            predicted_glucose = None

        # Add the prediction to our list
        if predicted_glucose is not None:
            past_values.append(predicted_glucose)
            predictions.append(predicted_glucose)

    last_time = glucose_data[-1][0]  # Get the last timestamp from the original data
    return predictions, last_time    # Return both predictions and that last timestamp

# -----------------------------
# MAIN TEST: Synthetic Data Generation
# -----------------------------

# This function creates and prints 100 synthetic glucose values
def test_synthetic_glucose_data():
    print("🔬 Testing synthetic glucose data generation...\n")

    data = generate_synthetic_glucose_data(length=100)  # Call the function to generate data

    print("📈 Generated Glucose Data (Time, Glucose):")
    for time_str, glucose in data:
        print(f" - {time_str} | {glucose} mg/dL")  # Print each data point

# -----------------------------
# MAIN TEST: Rolling ARIMA Forecast
# -----------------------------

# This function uses ARIMA to predict 5 future values and prints them
def test_rolling_arima_prediction():
    print("\n🔬 Testing rolling ARIMA prediction...\n")

    data = generate_synthetic_glucose_data(length=50)  # Use 50 historical points
    num_preds = 5  # Number of predictions to make

    predictions, last_time = run_rolling_arima(data, num_predictions=num_preds)

    print(f"🧠 Rolling ARIMA Predictions (starting from last time: {last_time}):")
    for i, val in enumerate(predictions):
        print(f" - Step {i+1}: {val:.2f} mg/dL")  # Print each predicted value

# -----------------------------
# RUN BOTH TESTS TOGETHER
# -----------------------------

# Run the two test functions when the script is executed
if __name__ == "__main__":
    test_synthetic_glucose_data()
    test_rolling_arima_prediction()
