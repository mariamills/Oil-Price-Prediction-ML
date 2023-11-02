import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from flask import Flask, render_template, jsonify, request, url_for
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = Flask(__name__)

# Load models into memory on app startup
models = {
    'random_forest': joblib.load('models/random_forest/random_forest_model.pkl'),
    # ... load other models ...
}

# Load Data from CSV when app is launched
df = pd.read_csv(
    'data/macro_features_and_real_oil_prices_log_tranferred_dropped_Nan_50_neg_and_zeros_for_log.csv',
    skiprows=0)


@app.route('/')
def index():
    return render_template('index.html')


# Feature Names Endpoint to get feature names for dropdown menu
@app.route('/features', methods=['GET'])
def get_features():
    feature_names = df.columns.tolist()[0:-1]
    return jsonify(feature_names)


# Data Endpoint to filter DataFrame based on selected features
@app.route('/data', methods=['POST'])
def get_data():
    selected_features = request.json['selected_features']
    filtered_df = df[selected_features]
    return jsonify(filtered_df.to_dict(orient='list'))


# Fetch Data and Plot
@app.route('/plot', methods=['POST'])
def plot():
    print(request.json)  # for debugging
    selected_features = request.json['selected_features']
    plot_type = request.json['plot_type']  # to specify the type of plot

    # Check for NaNs and infinities (keeping this part the same for now)
    if df.isnull().values.any() or df.isin([np.inf, -np.inf]).values.any():
        return jsonify({"error": "DataFrame contains NaN or Inf values"}), 400

    if 'Real Oil Prices' not in df.columns:
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


@app.route('/data_explorer')
def data_explorer():
    return render_template('data_explorer.html')


@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/predict_model', methods=['POST'])
def predict_model():
    try:
        model_type = request.get_json().get('model_type').lower()  # get model type from request
        model_type_dict = {"1": "random_forest", "2": "xgboost", "3": "polynomial_regression"}
        model_name = model_type_dict.get(model_type)
        if model_name is None:
            raise ValueError(f"No model found for type: {model_type}")
        model_file = f'models/{model_name}/{model_name}_model.pkl'
        model = joblib.load(model_file)

        selected_features = get_features_names()  # Get the selected features using your existing function
        print("SELECTED:", selected_features)
        if not selected_features:
            raise ValueError("No features selected")

        data = df[selected_features]  # filter DataFrame using selected features
        if data.empty:
            raise ValueError("No data available for selected features")

        # Assuming you have true values in a variable named true_values
        true_values = df['Real Oil Prices']  # Replace 'Real Oil Prices' with the name of your target column

        prediction = model.predict(data)

        # Compute metrics
        mse = mean_squared_error(true_values, prediction)
        mae = mean_absolute_error(true_values, prediction)
        rmse = np.sqrt(mse)

        # Plotting actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.plot(prediction, label='Predictions', color='blue')
        plt.plot(true_values.values, label='Actual', color='red')
        plt.plot([0, len(true_values)], [np.mean(prediction), np.mean(prediction)], '--', lw=2, color='green', label='Mean Prediction')
        plt.legend(loc='upper left')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Actual vs Predicted')
        actual_vs_predicted_plot_file = 'static/plots/actual-vs-predicted-plot.png'
        plt.savefig(actual_vs_predicted_plot_file)
        plt.close()

        # Feature Importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:]  # Get the indices of the top 10 features
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
            'actual_vs_predicted_plot': url_for('static', filename='plots/actual-vs-predicted-plot.png'),
            'feature_importance_plot': url_for('static', filename='plots/feature-importance-plot.png'),
            'prediction': prediction.tolist()  # Keeping the raw prediction values in case they are needed
        }

    except ValueError as e:
        print(f'Value error: {e}')
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print('Error while predicting:', str(e))
        return jsonify({'error': 'Error while predicting'}), 400

    return jsonify(result)


def get_features_names():
    feature_names = df.columns.tolist()[0:-1]
    return feature_names # return list of feature names


if __name__ == '__main__':
    app.run(debug=True)

