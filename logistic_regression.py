from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark Session
spark = SparkSession.builder.appName("Election_LR_Model").getOrCreate()

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

# Train Logistic Regression Model
lr = LogisticRegression(featuresCol="features", labelCol="WINNER")
lr_model = lr.fit(train_data)

# Make predictions on test data
predictions = lr_model.transform(test_data)

# Evaluate model accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="WINNER", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Logistic Regression Model Accuracy: {accuracy:.4f}")

# Save trained model locally for Flask to use
model_path = "/path/to/local/storage/logistic_model"  # Replace with your desired path
lr_model.write().overwrite().save(model_path)

print("Logistic Regression Model Trained and Saved Successfully!")
