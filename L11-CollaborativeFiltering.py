# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:55:08 2020

@author: wchen
"""

# In[1]:
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('collaborative filtering').getOrCreate()
        
# Loads data
ratings = spark.read.csv("D:/#DevCourses-GWU/#5_IoT_BigData/sample_movielens_ratings.csv", header=True, inferSchema=True)
ratings.show()
# 30 users
# 100 movies


# In[2]:
# Split the data in training and test sets
(training, test) = ratings.randomSplit([0.8, 0.2], seed=12345)

training.describe().show()
test.describe().show()
# In[3]:
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

"""
Collaborative filtering is commonly used for recommender systems. These techniques aim to fill in the missing entries of a user-item association matrix. 
spark.ml currently supports model-based collaborative filtering, in which users and products are described by a small set of latent factors that can 
be used to predict missing entries. spark.ml uses the alternating least squares (ALS) algorithm to learn these latent factors. The implementation in 
spark.ml has the following parameters:

1. numBlocks is the number of blocks the users and items will be partitioned into in order to parallelize computation (defaults to 10).
2. rank is the number of latent factors in the model (defaults to 10).
3. maxIter is the maximum number of iterations to run (defaults to 10).
4. regParam specifies the regularization parameter in ALS (defaults to 1.0).
5. implicitPrefs specifies whether to use the explicit feedback ALS variant or one adapted for implicit feedback data (defaults to false which means using explicit feedback).
6. alpha is a parameter applicable to the implicit feedback variant of ALS that governs the baseline confidence in preference observations (defaults to 1.0).
7. nonnegative specifies whether or not to use nonnegative constraints for least squares (defaults to false).
"""

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating")
model = als.fit(training)

# In[4]:
# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
predictions.show()
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# In[5]:
# Visualize how good we are
import pandas as pd
import seaborn as sns
predictions_pd = predictions.toPandas()
df = pd.DataFrame({'RealScore': predictions_pd['rating'], 
                   'Predicted': predictions_pd['prediction']})
sns.violinplot(x="RealScore", y="Predicted", data=df)


# In[6]:
# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)
userRecs.show(truncate=False)
# Generate top 10 user recommendations for each movie
movieRecs = model.recommendForAllItems(10)
movieRecs.show(truncate=False)


# In[7]:
# Generate top 10 movie recommendations for a subset of users
users = ratings.select('userId').distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
userSubsetRecs.show(truncate=False)

# Generate top 10 user recommendations for a subset of movies
movies = ratings.select(als.getItemCol()).distinct().limit(3)
movieSubSetRecs = model.recommendForItemSubset(movies, 10)
movieSubSetRecs.show(truncate=False)

# In[8]:
# Generate top 10 movie recommendations for a specified set of users defined by you
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col
df = spark.createDataFrame([1, 2, 3], IntegerType())
df.show()
df = df.select(col("value").alias("userId"))
df.show()

userSubsetRecs = model.recommendForUserSubset(df, 10)
userSubsetRecs.show(truncate=False)


# In[9]:
# Generate top 10 user recommendations for a specified set of movies defined by you
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col
df = spark.createDataFrame([1, 2, 3], IntegerType())
df = df.select(col("value").alias("movieId"))
df.show()

movieSubSetRecs = model.recommendForItemSubset(df, 10)
movieSubSetRecs.show(truncate=False)