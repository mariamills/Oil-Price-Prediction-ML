
# Oil Price Prediction: Data Exploration & Prediction Web Interface

This web interface is a crucial component of the larger __Oil Price Prediction Project__. Initially designed for data exploration, the site offers users a comprehensive platform for visualizing, filtering, and understanding historical trends in oil prices. In future releases, predictive analytics capabilities will be integrated into the platform, providing users with actionable insights into future oil price fluctuations.


## Project Documentation

⚠️ This README focuses primarily on the web interface portion of our Oil Price Prediction ML project. __It doesn't go into great detail about the technical aspects of the primary project, like data cleansing, model training, our development approach, and other technical details.__

If you're interested in a deeper understanding of our project—including challenges faced, solutions implemented, and technical details—we encourage you to visit the comprehensive documentation linked below.

[Project Documentation](https://mariamills.github.io/Oil-Price-Prediction-Documentation/)


## Features

- __Data Visualization__: Utilize interactive charts to explore historical oil price data, gaining a clearer picture of market trends over time.

- __Data Filtering__: Customize your data views with dynamic filtering options, _currently_ limited to only feature selection.

- __Graph Type Selection__: Choose between Scatter Plot and Line Plot visualizations to better suit your analysis needs.

- __Upcoming Predictions__: In future releases, users will have the ability to select from a range of predictive models and algorithms to forecast oil prices.


## API Reference

#### Get all features

Retrieves a list of all features that can be used for data visualization and filtering.

```http
  GET /features
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| N/A       | `string` |  List of all features      |



## Run Locally

#### Prerequisites
- Git
- Python 3.x
- Pip
- A web browser

#### Steps
1. Clone the project

```bash
  git clone https://github.com/mariamills/Oil-Price-Prediction-ML.git
```

2. Go to the project directory

```bash
  cd Oil-Price-Prediction-ML
```

3. Install dependencies

```bash
  pip install -r requirements.txt
```

4. Start the server

```bash
  python app.py
```
or in your IDE, run the app.py file.

5. Access the Web Interface
Open your web browser and navigate to the following address:

```
http://127.0.0.1:5000
```


## Tech Stack

**Frontend:** TailwindCSS, HTML, CSS, Javascript

**Backend:** Javascript, Flask(Python)

## Usage/Examples

### Overview
The Oil Price Prediction Web Interface is designed to be user-friendly and intuitive. This section provides you with some examples and scenarios to help you make the most out of the data exploration functionalities.

### Accessing the Website
To quickly begin, navigate to the website by clicking on the following link: [Oil Price Prediction Data Explorer](https://oil-price-prediction.onrender.com/data_explorer)

### User Guide
A quick user guide is available within the website to assist you in navigating the various features and functionalities. Look on the page and please read the _Important Notes_

### Example Scenarios
#### Exploring Data with Filters
__Choose Graph Type__: Decide whether you want to see the data as a scatter plot or a line plot using the dropdown selection.

### Future Features
In the upcoming releases, you will be able to select various predictive models to get future oil price estimates. This feature along with others such as time range selection and more is currently under development and will be announced when available.


## Screenshots

Coming soon
![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Demo

You can access a live demo of our Oil Price Prediction Web Interface by clicking [here](https://oil-price-prediction.onrender.com/).

#### Hosting Limitations
⚠️ __Please Note__: The website is currently hosted on [Render's](https://render.com/) free tier. Due to the limitations of this plan, the following should be noted:

- __Initial Load Time__: The app will shut down upon inactivity, so it may take a while to initially load.

- __Resource Constraints__: Performing too many operations, such as selecting a large number of features multiple times for visualization, may result in memory overflow.

#### Troubleshooting
If the chart takes longer than 3 minutes to generate, it's likely that the server is struggling with the load. In such cases, refreshing the page and selecting fewer features may resolve the issue.

#### Recommended Exploration
__Feature Selection__: Start by selecting just a few features for your initial exploration to avoid any potential memory issues.

__User Guide__: View the quick & simple user guide on the webpage for a quick overview of the website functionalities.

## FAQ
#### Where did you get the data that is being used here?
The data was provided by our 'sponsor,' a professor in the Economics department of our school. The data files include __Macroeconomic Data.csv__, which contains macroeconomic indicators from January 1986 to June 2023. We also received __RWTCm.xls__, containing data on the Cushing, OK WTI Spot Price FOB (Dollars per Barrel) from January 1986 to July 2023.

#### How up-to-date is the data?
The data spans from January 1986 to June 2023 for macroeconomic indicators and until July 2023 for oil prices. __Please note that this data is not regularly updated__ as it serves the educational purposes of a school project, designed as an introduction to machine learning.

#### What is this for?
This project is part of our Software Engineering class (CPSC 4175), where we were assigned a machine learning project focusing on oil price prediction. We were also assigned a 'sponsor' to simulate a real-world software engineering environment. Our team, "The Oval Table," comprises four members. The main requirement of this project is to use the provided data to train machine learning models capable of predicting oil prices.

#### Why build a whole web interface?
Initially, we built the web interface for data exploration to visually analyze the correlation between the various features and the 'real oil price'—which we calculate by adjusting the nominal price using the CPI. __Our ambition extends beyond just completing the class project; we aimed to provide an easy-to-use interface for our 'sponsor,' who is not a computer science major__. Rather than just delivering raw code or notebooks, we wanted to offer an intuitive, user-friendly experience.
## Feedback

If you have any feedback, please reach out to us at maria@mariamills.org


## Contributing

Contributions are always welcome!

Just fork the repo and submit a pull request. Please be as descriptive as possible in your pull request.

If you encounter any issues or have questions, please report them using the "Issues" section of the GitHub repository. Your input is valuable to us!

