from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark Session
spark = SparkSession.builder.appName("NaiveBayes_Classifier").getOrCreate()

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

# Split data
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train Naive Bayes Model
nb = NaiveBayes(featuresCol="features", labelCol="WINNER", modelType="multinomial")
nb_model = nb.fit(train_data)

# Make predictions
predictions = nb_model.transform(test_data)

# Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="WINNER", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Naive Bayes Model Accuracy: {accuracy:.4f}")
