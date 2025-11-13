from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, GBTClassifier, NaiveBayes
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
import pandas as pd

# Initialize Spark Session
spark = SparkSession.builder.appName("Election_Model_Evaluation").getOrCreate()

# Load preprocessed data from HDFS
data_path = "hdfs://localhost:9000/user/election_data/preprocessed_election_data"
df = spark.read.parquet(data_path)

# Ensure WINNER column has no NULL values and convert to integer
df = df.na.drop(subset=["WINNER"])
df = df.withColumn("WINNER", df["WINNER"].cast("integer"))

# Select features and target column
df = df.select("features", "WINNER")

# Split data into training (80%) and testing (20%)
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Function to evaluate models with multiple metrics
def evaluate_model(predictions, model_name):
    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="WINNER", metricName="accuracy")
    accuracy = accuracy_evaluator.evaluate(predictions)

    # Precision, Recall, F1 Score
    precision_evaluator = MulticlassClassificationEvaluator(labelCol="WINNER", metricName="weightedPrecision")
    precision = precision_evaluator.evaluate(predictions)

    recall_evaluator = MulticlassClassificationEvaluator(labelCol="WINNER", metricName="weightedRecall")
    recall = recall_evaluator.evaluate(predictions)

    f1_evaluator = MulticlassClassificationEvaluator(labelCol="WINNER", metricName="f1")
    f1 = f1_evaluator.evaluate(predictions)

    # AUC (Area Under the Curve)
    auc_evaluator = BinaryClassificationEvaluator(labelCol="WINNER", metricName="areaUnderROC")
    auc = auc_evaluator.evaluate(predictions)

    # MAE, MSE (Regression Metrics)
    mae_evaluator = RegressionEvaluator(labelCol="WINNER", predictionCol="prediction", metricName="mae")
    mae = mae_evaluator.evaluate(predictions)

    mse_evaluator = RegressionEvaluator(labelCol="WINNER", predictionCol="prediction", metricName="mse")
    mse = mse_evaluator.evaluate(predictions)

    return accuracy, precision, recall, f1, auc, mae, mse

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="WINNER"),
    "Gradient Boosting": GBTClassifier(featuresCol="features", labelCol="WINNER"),
    "SVM": LinearSVC(featuresCol="features", labelCol="WINNER"),
    "Naive Bayes": NaiveBayes(featuresCol="features", labelCol="WINNER")
}

# Store evaluation results
results = []

# Train, test, and evaluate models
for model_name, model in models.items():
    model_instance = model.fit(train_data)
    predictions = model_instance.transform(test_data)
    
    accuracy, precision, recall, f1, auc, mae, mse = evaluate_model(predictions, model_name)
    
    # Store the results
    results.append([model_name, accuracy, precision, recall, f1, auc, mae, mse])
    
    # Save the model
    model_instance.write().overwrite().save(f"/path/to/local/storage/{model_name.lower().replace(' ', '_')}_model")

# Convert results to DataFrame for easier visualization
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC", "MAE", "MSE"])

# Print the comparison table
print("\nModel Evaluation Comparison:")
print(results_df)

# Find the best model based on the highest accuracy
best_model_row = results_df.loc[results_df['Accuracy'].idxmax()]
best_model = best_model_row['Model']
best_accuracy = best_model_row['Accuracy']

print(f"\nBest Model based on Accuracy: {best_model} with Accuracy: {best_accuracy:.4f}")

print("\nModel Evaluation Completed and Models Saved Successfully!")
