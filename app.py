import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# Load Data from CSV when app is launched
df = pd.read_csv(
    'data/macro_features_and_real_oil_prices_log_tranferred_dropped_Nan_skipped_neg_and_zeros_for_log.csv',
    skiprows=1)

@app.route('/')
def index():
    return render_template('index.html')


# Feature Names Endpoint to get feature names for dropdown menu
@app.route('/get_features', methods=['GET'])
def get_features():
    feature_names = df.columns.tolist()[2:-1]
    return jsonify(feature_names)


# Data Endpoint to filter DataFrame based on selected features
@app.route('/get_data', methods=['POST'])
def get_data():
    selected_features = request.json['selected_features']
    filtered_df = df[selected_features]
    return jsonify(filtered_df.to_dict(orient='list'))


# Fetch Data and Plot
@app.route('/get_plot', methods=['POST'])
def plot():
    print(request.json) # for debugging
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
                line, = ax.plot(df[feature], df['Real Oil Prices'], label=feature, linewidth=2, linestyle='--', marker='o', markersize=3, alpha=0.7)
                ax.plot(df[feature].iloc[0], df['Real Oil Prices'].iloc[0], color='green', label=f"{feature} (Start)", marker='o', markersize=5, zorder=5)
                ax.plot(df[feature].iloc[-1], df['Real Oil Prices'].iloc[-1], color='red', label=f"{feature} (End)", marker='o', markersize=5, zorder=5)
                handles.append(line)
            elif plot_type == 'scatter':
                # Create a scatter plot
                scatter = ax.scatter(df[feature], df['Real Oil Prices'], label=f"{feature} (Others)", alpha=0.7, s=10)

                # Highlight the first and last point for each feature
                ax.scatter(df[feature].iloc[0], df['Real Oil Prices'].iloc[0], color='green', label=f"{feature} (Start)",
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


if __name__ == '__main__':
    app.run()