from pyspark.sql import SparkSession
import pandas as pd
import matplotlib.pyplot as plt

# Initialize Spark Session
spark = SparkSession.builder.appName("Visualization").getOrCreate()

# Load dataset
data_path = "hdfs://localhost:9000/user/election_data/preprocessed_election_data"
df = spark.read.parquet(data_path).toPandas()  # Convert Spark DataFrame to Pandas

# Group by Age and calculate the winning probability
age_win_data = df.groupby("AGE")["WINNER"].mean().reset_index()

# Plot the graph
plt.figure(figsize=(10, 5))
plt.plot(age_win_data["AGE"], age_win_data["WINNER"], marker='o', linestyle='-', color='b', label="Winning Rate")

# Graph Labels
plt.xlabel("Age")
plt.ylabel("Winning Probability")
plt.title("Age vs Winning Rate")
plt.legend()
plt.grid(True)

# Show plot
plt.show()
