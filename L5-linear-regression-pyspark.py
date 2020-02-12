#!/usr/bin/env python
# coding: utf-8

# In[1]:
# # Create entry points to spark
from pyspark.sql import SparkSession
spark = SparkSession.builder\
           .appName("Python Spark Linear Regression example")\
           .config("spark.some.config.option", "some-value")\
           .getOrCreate()

# In[2]:
# Import data

ad = spark.read.csv('C:/Users/wei_chen/Desktop/SKwk5/Advertising.csv', header=True, inferSchema=True)
ad.show(5)


# In[3]:
# Transform data structure

from pyspark.ml.linalg import Vectors
ad_df = ad.rdd.map(lambda x: [Vectors.dense(x[1:4]), x[-1]]).toDF(['features', 'label'])
ad_df.show(5)


# In[4]:
# Build linear regression model

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol = 'label')


# In[5]:
# Fit the model
lr_model = lr.fit(ad_df)


# In[6]:
# Prediction
pred = lr_model.transform(ad_df)
pred.show(5)


# In[7]:
# Module evaluation

from pyspark.ml.evaluation import RegressionEvaluator 
evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='label')
evaluator.setMetricName('r2').evaluate(pred)

# In[8]:
def modelsummary(model, param_names):
    import numpy as np
    print ("Note: the last rows are the information for Intercept")
    print ("##","-------------------------------------------------")
    print ("##","  Estimate   |   Std.Error | t Values  |  P-value")
    coef = np.append(list(model.coefficients), model.intercept)
    Summary=model.summary
    param_names.append('intercept')

    for i in range(len(Summary.pValues)):
        print ("##",'{:10.6f}'.format(coef[i]),\
        '{:14.6f}'.format(Summary.coefficientStandardErrors[i]),\
        '{:12.3f}'.format(Summary.tValues[i]),\
        '{:12.6f}'.format(Summary.pValues[i]), \
        param_names[i])

    print ("##",'---')
    print ("##","Mean squared error: % .6f" \
           % Summary.meanSquaredError, ", RMSE: % .6f" \
           % Summary.rootMeanSquaredError )
    print ("##","Multiple R-squared: %f" % Summary.r2, "," )
    print ("##","Multiple Adjusted R-squared: %f" % Summary.r2adj, ", \
            Total iterations: %i"% Summary.totalIterations)

param_names = ad.columns[1:4]
modelsummary(lr_model, param_names)
# In[9]:
# ## Linear regression with cross-validation
# ## Training and test datasets

training, test = ad_df.randomSplit([0.8, 0.2], seed=123)
lr_model = lr.fit(training)
pred = lr_model.transform(training)
pred.show(5)
param_names = ad.columns[1:4]
modelsummary(lr_model, param_names)

# Make predictions.
pred_test = lr_model.transform(test)
pred_test.show(5)

from pyspark.ml.evaluation import RegressionEvaluator
# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(labelCol="label",
                                predictionCol="prediction",
                                metricName="rmse")
# metricName Supports: - "rmse" (default): root mean squared error - 
# "mse": mean squared error - 
# "r2": R^2^ metric - 
# "mae": mean absolute error

rmse = evaluator.evaluate(pred_test)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)




# In[10]:
# The Second Example
# Use Spark to read in the Ecommerce Customers csv file.
data = spark.read.csv("C:/Users/wei_chen/Desktop/SKwk5/Ecommerce_Customers.csv",inferSchema=True,header=True)
data.show(5)
data.columns
for item in data.head():
    print(item)


# In[11]:
# A few things we need to do before Spark can accept the data!
# It needs to be in the form of two columns
# ("label","features")

# Import VectorAssembler and Vectors
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=["Avg Session Length", "Time on App", 
               "Time on Website",'Length of Membership'],
    outputCol="features")

output = assembler.transform(data)

output.select("features").head()

final_data = output.select("features",'Yearly Amount Spent')


# In[15]:

train_data,test_data = final_data.randomSplit([0.7,0.3])
train_data.show(5)
train_data.describe().show()
test_data.show(5)
test_data.describe().show()


# In[18]:

# Create a Linear Regression Model object
lr = LinearRegression(labelCol='Yearly Amount Spent')

# Fit the model to the data and call this model lrModel
lrModel = lr.fit(train_data,)

# Print the coefficients and intercept for linear regression
print("Coefficients: {} Intercept: {}".format(lrModel.coefficients,lrModel.intercept))
param_names = ["Avg Session Length", "Time on App", 
               "Time on Website",'Length of Membership']
modelsummary(lrModel, param_names)

# In[21]:

test_results = lrModel.evaluate(test_data)
test_results.predictions.show(5)

# In[23]:

unlabeled_data = test_data.select('features')
predictions = lrModel.transform(unlabeled_data)
predictions.show(5)


# In[26]:

print("RMSE: {}".format(test_results.rootMeanSquaredError))
print("MSE: {}".format(test_results.meanSquaredError))













