'''
Blue: if below 80 (suggesting low glucose).

Red: if above 130 (indicating high glucose).

Green: for values in the normal range. 70-180
'''


# Import the pandas library, which helps in working with tables (dataframes)
import pandas as pd

# Import the ARIMA model from the statsmodels package for time series forecasting
from statsmodels.tsa.arima.model import ARIMA

# Import a custom function that generates synthetic (fake) glucose data
from data_loader import generate_synthetic_glucose_data

# Import a custom function that creates an interactive tree visualization of glucose predictions
from interactive_tree import generate_interactive_glucose_tree

# This line checks if this script is being run directly (not imported in another file)
if __name__ == "__main__":

    # Generate synthetic glucose data with 100 data points using the custom function
    glucose_data = generate_synthetic_glucose_data()
    
    # Check if the glucose data was successfully generated; if not, print an error and stop the program
    if not glucose_data:
        print("❌ Failed to load glucose data.")
        exit()  # Stop the program if there is no data

    # Ask the user to input the number of future glucose values they want to predict.
    # The input() function takes the user input as a string, so we convert it to an integer with int()
    num_predictions = int(input("Enter number of glucose values to predict: "))

    # Define a function that performs a rolling ARIMA forecast.
    # This function will use past glucose data to predict future glucose values one by one.
    def run_rolling_arima(glucose_data, num_predictions, min_past_values=5):
        # Check if there are enough glucose data points to perform the predictions
        if len(glucose_data) < num_predictions:
            print(f"Not enough glucose data for {num_predictions}-step rolling prediction.")
            return [], None  # Return an empty list and None if there isn't enough data

        # Create an empty list to store the predicted glucose values
        predictions = []
        # Create a list to hold past glucose values extracted from the data
        past_values = []
        # Set the ARIMA model parameters (p, d, q) to use for the prediction
        best_order = (1, 1, 0)

        # Loop through each data point in the glucose data to extract the actual glucose values
        for i in range(len(glucose_data)):
            # Each element in glucose_data is a tuple; we ignore the time (first element) and use the glucose value (second element)
            _, original_glucose = glucose_data[i]
            past_values.append(original_glucose)  # Add the glucose value to our list of past values

        # Now, for each future prediction we want to generate...
        for _ in range(num_predictions):
            # Check if we have at least the minimum required past values to build a model
            if len(past_values) >= min_past_values:
                try:
                    # Create an ARIMA model using the past glucose values and the defined order
                    model = ARIMA(past_values, order=best_order)
                    # Fit the model to our past data (this trains the model)
                    model_fit = model.fit()
                    # Forecast the next single glucose value
                    forecast = model_fit.get_forecast(steps=1)
                    # Get the predicted value from the forecast result
                    predicted_glucose = forecast.predicted_mean[0]
                except Exception as e:
                    # If any error occurs during the prediction, print an error message and set the predicted value to None
                    print(f"Rolling ARIMA prediction failed: {e}")
                    predicted_glucose = None
            else:
                # If we don't have enough past values, we cannot predict, so set predicted value to None
                predicted_glucose = None

            # If a valid prediction was made, update the past values and add the prediction to our list
            if predicted_glucose is not None:
                past_values.append(predicted_glucose)
                predictions.append(predicted_glucose)

        # Get the time from the last data point in the original glucose data. This is used to set the starting point for future times.
        last_time = glucose_data[-1][0]
        # Return the list of predictions and the time of the last original data point
        return predictions, last_time

    # Call the run_rolling_arima function to predict future glucose values.
    # It returns the list of predictions and the last time stamp from the original data.
    predictions, last_time = run_rolling_arima(glucose_data, num_predictions)

    # Build a range of future times (timestamps) for the predicted values.
    # We use pd.date_range to generate timestamps starting from the last original time, with a frequency of 5 minutes.
    # The [1:] part skips the first generated timestamp because it's the same as the last original time.
    future_times = pd.date_range(start=pd.to_datetime(glucose_data[-1][0]), periods=num_predictions + 1, freq='5min')[1:]

    # Combine the times from the original data and the future predictions.
    # This creates a complete list of time points.
    all_times = [t for t, _ in glucose_data] + list(future_times.astype(str))
    
    # Combine the glucose values from the original data and the predictions.
    all_glucose = [g for _, g in glucose_data] + predictions

    # Create a pandas DataFrame (table) that contains the combined times and glucose values.
    df = pd.DataFrame({"time": all_times, "glucose": all_glucose})
    
    # Convert the 'time' column in the DataFrame to a datetime format.
    # This ensures that the time data is recognized as dates/times.
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # Generate the interactive glucose prediction tree visualization.
    # This function creates an HTML file that displays the interactive tree.
    # The 'depth' parameter is set to the number of predictions, controlling how many levels the tree will have.
    generate_interactive_glucose_tree(df, "glucose_tree_interactive.html", depth=num_predictions)
