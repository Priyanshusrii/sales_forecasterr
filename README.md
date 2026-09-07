Multi-Model Sales Forecaster
An interactive sales forecasting web application built with Streamlit that evaluates and compares three distinct machine learning algorithms across multiple business scenarios to identify the most accurate model.

Overview
Traditional sales forecasting tools often apply a single predictive algorithm regardless of data patterns. This project implements a comparative approach: it evaluates Linear Regression, Decision Tree, and Random Forest models against distinct market environments (such as marketing spend variations, seasonal spikes, and holiday rushes) and automatically highlights the best-performing model based on evaluation metrics.

Features
Multi-Model Evaluation: Automatically benchmarks Linear Regression, Decision Tree, and Random Forest.

Intelligent Model Selection: Determines the "Winner" model per scenario using R 
2
  Score (Goodness of Fit) and Mean Absolute Error (MAE).

Interactive UI: Dynamic dashboards powered by Streamlit for instant real-time predictions based on user inputs (e.g., budget, units, marketing spend).

Data Visualization: Plots historical data alongside model trendlines using Matplotlib for transparent interpretation.

Diverse Market Scenarios: Includes datasets modeling various retail behaviors and economic patterns.

Tech Stack
Language: Python

Web Framework: Streamlit

Machine Learning: Scikit-Learn

Data Manipulation & Math: Pandas, NumPy

Visualization: Matplotlib

Model Serialization: Joblib

Project Structure
Plaintext
├── datasets/             # Market scenario datasets (.csv)
├── models/               # Serialized pre-trained models (.pkl)
├── app.py                # Main Streamlit dashboard application
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
Installation & Setup
Clone the Repository

Bash
git clone https://github.com/Priyanshusrii/sales_forecasterr.git
cd Sales-Forecaster
Create a Virtual Environment (Optional but Recommended)

Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
Install Dependencies

Bash
pip install -r requirements.txt
Run the Application

Bash
streamlit run app.py
Evaluation Metrics
R 
2
  Score: Measures the proportion of variance in sales explained by the independent variables.

MAE (Mean Absolute Error): Measures the average magnitude of prediction errors in actual sales units/revenue.

Author
Priyanshu Srivastava – BCA, Parul University
