from pyspark.sql import SparkSession
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Spark Session
spark = SparkSession.builder.appName("Visualization").getOrCreate()

# Load dataset from HDFS
data_path = "hdfs://localhost:9000/user/election_data/preprocessed_election_data"
df = spark.read.parquet(data_path).toPandas()  # Convert Spark DataFrame to Pandas

# Create the boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x="WINNER", y="TOTAL VOTES", data=df)

# Add labels and title
plt.xlabel("Winner (0 = Lost, 1 = Won)")
plt.ylabel("Total Votes")
plt.title("Distribution of Total Votes by Election Outcome")

# Show the plot
plt.show()
