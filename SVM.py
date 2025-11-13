from pyspark.sql import SparkSession
from pyspark.ml.classification import LinearSVC
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark Session
spark = SparkSession.builder.appName("SVM_Classifier").getOrCreate()

# Load preprocessed dataset
data_path = "hdfs://localhost:9000/user/election_data/preprocessed_election_data"
df = spark.read.parquet(data_path)

# Ensure WINNER column has no NULL values and convert to integer
df = df.na.drop(subset=["WINNER"])
df = df.withColumn("WINNER", df["WINNER"].cast("integer"))

# Define feature columns
feature_columns = ["AGE", "ASSETS", "LIABILITIES", "TOTAL VOTES"]

# Drop existing 'features' column if it exists
if "features" in df.columns:
    df = df.drop("features")

# Assemble feature vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df = assembler.transform(df)

# Apply StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

# Split the data into training and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train SVM Model on scaled features
svm = LinearSVC(featuresCol="scaledFeatures", labelCol="WINNER", maxIter=50)
svm_model = svm.fit(train_data)

# Make predictions
predictions = svm_model.transform(test_data)

# Evaluate model accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="WINNER", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"SVM Model Accuracy: {accuracy:.4f}")

# Feature importance (absolute value of coefficients)
coefficients = svm_model.coefficients
print("\nFeature Importance (absolute value of coefficients):")
for i, feature in enumerate(feature_columns):
    print(f"{feature}: {abs(coefficients[i]):.4f}")