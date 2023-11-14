import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from flask import Flask, render_template, jsonify, request, url_for
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels

app = Flask(__name__)

NUM_FEATURES = 10  # Number of features to show in feature importance plot

# Determine the number of steps to forecast
num_steps = 30  # 30 days

# Load models and their corresponding datasets into memory on app startup
model_data = {
    'random_forest': {
        'model': joblib.load('models/random_forest/random_forest_model.pkl'),
        'data': 'data/random_forest/RF_Combined_Log_Clean_NoNeg.csv',
        'is_time_series': False
    },
    'arima': {
        'model': joblib.load('models/arima/arima_model.pkl'),
        'data': 'data/arima/diff_oil_prices_arima.csv',
        'is_time_series': True
    },
    'prophet': {
        'model': joblib.load('models/prophet/prophet_model.pkl'),
        'data': 'data/prophet/prophet_df.csv',
        'is_time_series': True
    }
}


def load_model(model_name):
    return model_data[model_name]['model']


def get_model_data(model_name):
    return model_data[model_name]['data']


def load_data_for_model(model_type):
    if model_type not in model_data:
        raise ValueError(f"No model found for type: {model_type}")
    data_path = get_model_data(model_type)
    return pd.read_csv(data_path)


# Load Data from CSV when app is launched
# df = pd.read_csv('data/Combined_Log_Excl_Roil_Clean.csv',skiprows=0, usecols=lambda x: x != 'date')

@app.route('/')
def index():
    return render_template('index.html')


# Feature Names Endpoint to get feature names for chosen model
@app.route('/features', methods=['GET'])
def get_features():
    model_type = request.args.get('model', 'random_forest')

    if model_type not in model_data:
        return jsonify({'error': f"No model found for type: {model_type}"}), 400

    data = load_data_for_model(model_type)  # Load the data into a DataFrame

    # Fetch feature names from the model-specific data
    feature_names = data.columns.tolist()[0:-1]  # remove the last column (target)
    return jsonify(feature_names)


# Data Endpoint to filter DataFrame based on selected features
@app.route('/data', methods=['POST'])
def get_data():
    selected_features = request.json['selected_features']
    # Correct way to get 'model_type' with a default value
    model_type = request.json.get('model_type', 'random_forest')

    data_path = get_model_data(model_type)
    df = pd.read_csv(data_path)  # Load the data into a DataFrame

    filtered_df = df[selected_features]
    return jsonify(filtered_df.to_dict(orient='list'))


# Fetch Data and Plot
@app.route('/plot', methods=['POST'])
def plot():
    print(request.json)  # for debugging
    selected_features = request.json['selected_features']
    plot_type = request.json['plot_type']  # to specify the type of plot
    model_type = request.json.get('model_type', 'random_forest')

    data_path = get_model_data(model_type)
    df = pd.read_csv(data_path)  # Load the data dynamically

    # Check for NaNs and infinities (keeping this part the same for now)
    if df.isnull().values.any() or df.isin([np.inf, -np.inf]).values.any():
        return jsonify({"error": "DataFrame contains NaN or Inf values"}), 400

    target_name = 'Real Oil Prices' or 'Cushing, OK WTI Spot Price FOB (Dollars per Barrel)' or 'y'
    if target_name not in df.columns:
        return jsonify({"error": "'Real Oil Prices' not found in DataFrame"}), 400

    # Create the plot and set its size
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

    # Loop through the features, plotting each one
    handles = []
    for feature in selected_features:
        if feature in df.columns:
            if plot_type == 'line':
                # Line Plot
                line, = ax.plot(df[feature], df['Real Oil Prices'], label=feature, linewidth=2, linestyle='--',
                                marker='o', markersize=3, alpha=0.7)
                ax.plot(df[feature].iloc[0], df['Real Oil Prices'].iloc[0], color='green', label=f"{feature} (Start)",
                        marker='o', markersize=5, zorder=5)
                ax.plot(df[feature].iloc[-1], df['Real Oil Prices'].iloc[-1], color='red', label=f"{feature} (End)",
                        marker='o', markersize=5, zorder=5)
                handles.append(line)
            elif plot_type == 'scatter':
                # Create a scatter plot
                scatter = ax.scatter(df[feature], df['Real Oil Prices'], label=f"{feature} (Others)", alpha=0.7, s=10)

                # Highlight the first and last point for each feature
                ax.scatter(df[feature].iloc[0], df['Real Oil Prices'].iloc[0], color='green',
                           label=f"{feature} (Start)",
                           zorder=5)
                ax.scatter(df[feature].iloc[-1], df['Real Oil Prices'].iloc[-1], color='red', label=f"{feature} (End)",
                           zorder=5)
                handles.append(scatter)

    ax.set_xlabel('Features')
    ax.set_ylabel('Real Oil Prices')
    ax.set_title('Real Oil Prices vs. Features')
    # Grid Lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save just the plot (no legend) to a BytesIO object
    img_plot = BytesIO()
    plt.savefig(img_plot, format='png')
    img_plot.seek(0)

    # Create a separate figure just for the legend
    fig_legend = plt.figure(figsize=(10, 3))
    axi = fig_legend.add_subplot(111)
    axi.axis('off')
    fig_legend.legend(handles, selected_features, loc='center', ncol=5)

    # Save just the legend to a BytesIO object
    img_legend = BytesIO()
    plt.savefig(img_legend, format='png')
    img_legend.seek(0)

    # Convert both to base64
    plot_url = base64.b64encode(img_plot.getvalue()).decode()
    legend_url = base64.b64encode(img_legend.getvalue()).decode()

    # Cleanup
    plt.close(fig)
    plt.close(fig_legend)

    return jsonify({'plot_url': 'data:image/png;base64,{}'.format(plot_url),
                    'legend_url': 'data:image/png;base64,{}'.format(legend_url)})



# Fetch number of days to forecast
@app.route('/forecast', methods=['POST'])
def get_forecast_days():
    global num_steps
    num_steps = request.json['days']
    # convert to int
    num_steps = int(num_steps)
    return jsonify(num_steps)


@app.route('/data_explorer')
def data_explorer():
    return render_template('data_explorer.html')


@app.route('/predict')
def predict():
    # Send the default dataset (or based on a default model) to the Predict page
    default_model = 'random_forest'  # Example default model
    default_data_path = get_model_data(default_model)
    default_data = pd.read_csv(default_data_path)  # Load the data into a DataFrame
    feature_names = default_data.columns.tolist()[0:-1]
    return render_template('predict.html', feature_names=feature_names)


@app.route('/update_features', methods=['POST'])
def update_features():
    model_type = request.json['model_type']
    if model_type not in model_data:
        return jsonify({'error': f"No model found for type: {model_type}"}), 400

    data_path = get_model_data(model_type)
    df = pd.read_csv(data_path)  # Load the data into a DataFrame
    feature_names = df.columns.tolist()[0:-1]
    return jsonify(feature_names)


def calculate_metrics(true_values, prediction):
    mse = mean_squared_error(true_values, prediction)
    mae = mean_absolute_error(true_values, prediction)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((true_values - prediction) / true_values)) * 100
    r2 = r2_score(true_values, prediction)
    return mse, mae, rmse, mape, r2


# Evaluate Endpoint
# Making this a POST because we are sending the selected features in the request body which is long
@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        print(f"Request: {request.get_json()}")
        model_type = request.get_json().get('model').lower()

        if model_type not in model_data:
            raise ValueError(f"No model found for type: {model_type}")

        model = load_model(model_type)
        data_path = get_model_data(model_type)
        data = pd.read_csv(data_path)  # Load the data into a DataFrame

        print(f"Evaluating Model type: {model_type}, data: {data_path}")

        target_column_options = ['Real Oil Prices', 'Cushing, OK WTI Spot Price FOB (Dollars per Barrel)', 'y']
        target_column = next((col for col in target_column_options if col in data.columns), None)
        if target_column is None:
            raise ValueError(f"None of the target columns {target_column_options} were found in DataFrame")

        # Check if the model is a time series model
        if model_data[model_type]['is_time_series']:
            # drop the 'Unnamed: 0' column
            if 'Unnamed: 0' in data.columns:
                data = data.drop(columns=['Unnamed: 0'])

            print("Time Series Model Detected")

            # For time series model, use only the target column
            # Prepare the data for prediction
            print(f"Target column: {target_column}")
            time_series_data = data[target_column]
            print(f"Time Series Data: {time_series_data}")

            if model_type == 'arima':
                # Make predictions using get_forecast
                forecast = model.get_forecast(steps=num_steps)

                # Generate and save the future forecast plot
                plt.figure(figsize=(12, 6))
                historical_data = pd.read_csv('data/Combined_Log_Clean.csv')[target_column]
                future_forecast_mean = forecast.predicted_mean
                future_conf_int = forecast.conf_int()

                plt.plot(historical_data.index, historical_data, label='Historical')
                future_index = np.arange(len(historical_data), len(historical_data) + len(future_forecast_mean))
                plt.plot(future_index, future_forecast_mean, color='green', label='Future Forecast')
                plt.fill_between(future_index, future_conf_int.iloc[:, 0], future_conf_int.iloc[:, 1],
                                 color='lightgreen', alpha=0.5, label='95% Confidence Interval')
                plt.xlabel('Time')
                plt.ylabel('Real Oil Prices')
                plt.title('Future Forecast with Confidence Intervals')
                plt.legend()
                future_forecast_plot_file = 'static/plots/arima-future-forecast-plot.png'
                plt.savefig(future_forecast_plot_file)
                plt.close()

                # Calculate metrics - can't give metrics for the future
                mse, mae, rmse, mape, r2 = [None] * 5

                result = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'r2': r2,
                    'future_forecast_plot': url_for('static', filename='plots/arima-future-forecast-plot.png'),
                    'prediction': future_forecast_mean.tolist(),
                }
            elif model_type == 'prophet':
                print("hello")
                # Create a DataFrame with future dates for forecasting
                future = model.make_future_dataframe(periods=num_steps)
                forecast = model.predict(future)

                # Plotting code for Prophet
                fig, ax = plt.subplots(figsize=(10, 6))
                prophet_df = pd.read_csv('data/prophet/prophet_df.csv')
                prophet_df['ds'] = pd.to_datetime(prophet_df['ds']).dt.tz_localize(None)
                ax.plot(prophet_df['ds'], prophet_df['y'], 'k.', label='Historical Data')
                ax.plot(forecast['ds'], forecast['yhat'], ls='-', color='blue', label='Forecast')
                ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='blue', alpha=0.2,
                                label='Uncertainty Interval')
                ax.axvline(x=prophet_df['ds'].iloc[-1], color='red', linestyle='--', lw=1, label='Start of Forecast')
                ax.set_xlabel('Date')
                ax.set_ylabel('Oil Prices')
                ax.set_title(f"{num_steps}-Day Forecast with Prophet")
                ax.legend()
                future_forecast_plot_file = 'static/plots/prophet-future-forecast-plot.png'
                plt.savefig(future_forecast_plot_file)
                plt.show()

                # Calculate metrics - can't give metrics for the future
                mse, mae, rmse, mape, r2 = [None] * 5
                print(f"Metrics: mse [Time Series]={mse}, mae={mae}, rmse={rmse}, mape={mape}, r2={r2}")

                result = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'r2': r2,
                    'future_forecast_plot': url_for('static', filename='plots/prophet-future-forecast-plot.png'),
                    'prediction': forecast['yhat'].tolist(),  # Keeping the raw prediction values in case they are needed,
                }

        else:
            request_data = request.get_json()
            selected_features = request_data.get('selected_features')
            if not selected_features:
                raise ValueError("No features selected")

            filtered_data = data[selected_features]
            if filtered_data.empty:
                raise ValueError("No data available for selected features")

            # Debugging: Print feature names
            print("Features used for prediction:", selected_features)
            print("Features used in training:", data.columns.tolist())

            print(f"Target column: {target_column}")
            print(f"Target column values: {data[target_column]}")
            print(f"Target column values type: {type(data[target_column])}")
            print(f"Filtered data: {filtered_data}")
            print(f"Filtered data type: {type(filtered_data)}")

            print("Before prediction")
            true_values = data[target_column]
            print("True Values:", true_values)

            # Convert to numpy array
            true_values_array = true_values.values  # or true_values.to_numpy()
            print("True Values Array:", true_values_array)

            prediction = model.predict(filtered_data)
            print("Prediction:", prediction)

            print(f"Calculating metrics for model type: {model_type}")
            # Compute metrics
            mse, mae, rmse, mape, r2 = calculate_metrics(true_values_array, prediction)
            print(f"Metrics: mse={mse}, mae={mae}, rmse={rmse}, mape={mape}, r2={r2}")

            # Plotting actual vs predicted
            plt.figure(figsize=(10, 6))
            plt.plot(prediction, label='Predictions', color='blue')
            plt.plot(true_values.values, label='Actual', color='red')
            plt.plot([0, len(true_values)], [np.mean(prediction), np.mean(prediction)], '--', lw=2, color='green',
                     label='Mean Prediction')
            plt.legend(loc='upper left')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title('Actual vs Predicted')
            actual_vs_predicted_plot_file = 'static/plots/actual-vs-predicted-plot.png'
            plt.savefig(actual_vs_predicted_plot_file)
            plt.close()

            # Feature Importance
            importances = model.feature_importances_
            indices = np.argsort(importances)[-NUM_FEATURES:]  # Get the indices of the top NUM_FEATURES features
            plt.figure()
            plt.title('Feature Importances')
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [selected_features[i] for i in indices])
            plt.xlabel('Relative Importance')
            feature_importance_plot_file = 'static/plots/feature-importance-plot.png'
            plt.savefig(feature_importance_plot_file)
            plt.close()

            result = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                'actual_vs_predicted_plot': url_for('static', filename='plots/actual-vs-predicted-plot.png'),
                'feature_importance_plot': url_for('static', filename='plots/feature-importance-plot.png'),
                'prediction': prediction.tolist(),  # Keeping the raw prediction values in case they are needed,
            }

    except ValueError as e:
        print(f'Value error: {e}')
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        print(f'Error while predicting: {e}')
        return jsonify({'error': 'Error while predicting'}), 400

    return jsonify(result)


def get_features_names(model_type):
    data_path = get_model_data(model_type)
    data = pd.read_csv(data_path)  # Load the data into a DataFrame

    feature_names = data.columns.tolist()[0:-1]  # Use 'data' instead of 'df'
    return feature_names  # Return list of feature names


if __name__ == '__main__':
    app.run(debug=True)
