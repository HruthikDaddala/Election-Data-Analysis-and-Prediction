from flask import Flask, jsonify, request
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
import json

# Initialize Flask App and Spark Session
app = Flask(__name__)
spark = SparkSession.builder.appName("Election_Prediction_API").getOrCreate()

# Load saved models (adjust paths if needed)
lr_model = LogisticRegressionModel.load("hdfs://localhost:9000/user/election_data/logistic_model")
rf_model = RandomForestClassificationModel.load("hdfs://localhost:9000/user/election_data/random_forest_model")

@app.route('/')
def home():
    return "Welcome to the Election Prediction API!"

@app.route('/predict_lr', methods=['POST'])
def predict_lr():
    try:
        # Get data from POST request
        data = request.get_json()  # Expecting input like {'feature1': value, 'feature2': value, ...}

        # Convert the input features into a DataFrame
        feature_names = list(data.keys())
        feature_values = list(data.values())

        # Create a DataFrame for prediction (feature columns must match the input names)
        input_data = [(tuple(feature_values))]
        input_df = spark.createDataFrame(input_data, feature_names)

        # Assemble features into a vector
        assembler = VectorAssembler(inputCols=feature_names, outputCol="features")
        assembled_df = assembler.transform(input_df)

        # Make predictions using the Logistic Regression model
        lr_prediction = lr_model.transform(assembled_df)
        lr_predicted_class = lr_prediction.select("prediction").collect()[0][0]  # Get prediction

        return jsonify({'prediction': int(lr_predicted_class)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    try:
        # Get data from POST request
        data = request.get_json()  # Expecting input like {'feature1': value, 'feature2': value, ...}

        # Convert the input features into a DataFrame
        feature_names = list(data.keys())
        feature_values = list(data.values())

        # Create a DataFrame for prediction (feature columns must match the input names)
        input_data = [(tuple(feature_values))]
        input_df = spark.createDataFrame(input_data, feature_names)

        # Assemble features into a vector
        assembler = VectorAssembler(inputCols=feature_names, outputCol="features")
        assembled_df = assembler.transform(input_df)

        # Make predictions using the Random Forest model
        rf_prediction = rf_model.transform(assembled_df)
        rf_predicted_class = rf_prediction.select("prediction").collect()[0][0]  # Get prediction

        return jsonify({'prediction': int(rf_predicted_class)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
