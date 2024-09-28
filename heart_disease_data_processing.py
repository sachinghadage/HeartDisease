# Import required libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql.functions import pow, when

# Step 1: Data Loading
spark = SparkSession.builder.appName("Heart Disease Data Preprocessing").getOrCreate()
file_path = "DataSet.csv"  # Update this path
df = spark.read.csv(file_path, header=True, inferSchema=True)
df.show()

# Step 2: One-Hot Encoding
indexer = StringIndexer(inputCol="cp", outputCol="cp_index")
df_indexed = indexer.fit(df).transform(df)
encoder = OneHotEncoder(inputCol="cp_index", outputCol="cp_vector")
df_encoded = encoder.fit(df_indexed).transform(df_indexed)
df_encoded.select("cp", "cp_index", "cp_vector").show()

# Step 3: Feature Derivation
df_transformed = df_encoded.withColumn("powerOfTrestbps", pow(df_encoded["trestbps"], 2))
df_transformed.select("trestbps", "powerOfTrestbps").show()

# Step 4: Data Filtering
df_filtered = df_transformed.filter((df_transformed["age"] > 50) & (df_transformed["trestbps"] > 140))
df_filtered.show()

# Step 5: Quantization of Cholesterol Levels
df_quantized = df_filtered.withColumn("cholesterol_level",
    when(df_filtered["chol"] < 200, "Low")
    .when((df_filtered["chol"] >= 200) & (df_filtered["chol"] <= 239), "Medium")
    .otherwise("High")
)
df_quantized.select("chol", "cholesterol_level").show()

# Step 6: Data Reduction
high_chol_count = df_quantized.filter(df_quantized["cholesterol_level"] == "High").count()
print(f"Number of patients with High cholesterol: {high_chol_count}")

# Drop the complex column
df_quantized = df_quantized.drop("cp_vector")

# Step 7: Data Export (Save as CSV)
output_path = "output"  # Update this path
df_quantized.write.csv(output_path, header=True, mode="overwrite")
print(f"Processed dataset saved to {output_path}")

# Stop the Spark session
spark.stop()
