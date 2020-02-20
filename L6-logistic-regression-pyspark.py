#!/usr/bin/env python
# coding: utf-8

# In[0]:
# ## Logistic regression with pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder\
           .appName("Python Spark Linear Regression example")\
           .config("spark.some.config.option", "some-value")\
           .getOrCreate()

# ## Import data
cuse = spark.read.csv('cuse_binary.csv', header=True, inferSchema=True)
cuse.show(5)


# In[1]:

# ## Process categorical columns

# The following code does three things with pipeline:
# 
# * **`StringIndexer`** all categorical columns
# * **`OneHotEncoder`** all categorical index columns
# * **`VectorAssembler`** all feature columns into one vector column

# ### Categorical columns

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
import pyspark.sql.functions as F

# categorical columns
categorical_columns = cuse.columns[0:3]
stage_string = [StringIndexer(inputCol= c, outputCol= c+"_string_encoded") for c in categorical_columns]
stage_one_hot = [OneHotEncoder(inputCol= c+"_string_encoded", outputCol= c+ "_one_hot") for c in categorical_columns]

ppl = Pipeline(stages=stage_string + stage_one_hot)
df = ppl.fit(cuse).transform(cuse)
df.toPandas().to_csv('cuse_afterTransform.csv')
df.select("age", 'age_string_encoded').distinct().sort(F.asc("age_string_encoded")).show()
df.select("age").groupBy("age").count().orderBy(F.desc('count')).show()
#df.select("education").distinct().show()
df.select("education", 'education_string_encoded').distinct().sort(F.asc("education_string_encoded")).show()
df.select("education").groupBy("education").count().orderBy(F.desc('count')).show()
#df.select("wantsMore").distinct().show()
df.select("wantsMore", 'wantsMore_string_encoded').distinct().sort(F.asc("wantsMore_string_encoded")).show()
df.select("wantsMore").groupBy("wantsMore").count().orderBy(F.desc('count')).show()

# In[2]:
# ### Build VectorAssembler stage
df.columns

assembler = VectorAssembler(
  inputCols=['age_one_hot',
             'education_one_hot',
             'wantsMore_one_hot',
             ],
    outputCol="features")

cuse_df = assembler.transform(df)
cuse_df = cuse_df.withColumn('label', F.col('y'))
cuse_df.select("features", "label").show()

cuse_df.show(5)


# In[3]:
# ## Split data into training and test datasets
training, test = cuse_df.randomSplit([0.8, 0.2], seed=1234)

# In[4]:
# ## Build Logistic Regression model

from pyspark.ml.regression import GeneralizedLinearRegression
logr = GeneralizedLinearRegression(family="binomial", link="logit", regParam=0.0)

# Fit the model to the data and call this model logr_Model
logr_Model = logr.fit(training)

# Print the coefficients and intercept for linear regression
summary = logr_Model.summary
print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
print("T Values: " + str(summary.tValues))
print("P Values: " + str(summary.pValues))


# #### Prediction on training data
pred_training_cv = logr_Model.transform(training)
pred_training_cv.show(5, truncate=False)

# #### Prediction on test data
pred_test_cv = logr_Model.transform(test)
pred_test_cv.show(5, truncate=False)

# In[5]:
# #### Cost Matrix

from sklearn.metrics import confusion_matrix
y_true = pred_test_cv.select("label")
y_true = y_true.toPandas()

y_pred = pred_test_cv.select("prediction").withColumn('pred_label', F.when(F.col('prediction') >0.5, 1).otherwise(0))
y_pred = y_pred.select('pred_label').toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred)
cnf_matrix

# In[6]:
# #### Model Summary
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

param_names = ['age_30-39',
             'age_25-29',
             'age_<25',
             'education_high',
             'wantsMore_yes']

modelsummary(logr_Model, param_names)


# In[6]:
# #### Second Example
data = spark.read.csv('titanic.csv',inferSchema=True,header=True)

data.printSchema()
data.columns


# In[7]:
# #### Remove the rows with missing data
my_cols = data.select(['Survived',
 'Pclass',
 'Sex',
 'Age',
 'SibSp',
 'Parch',
 'Fare',
 'Embarked'])

my_final_data = my_cols.na.drop()

# In[9]:
# ### Working with Categorical Columns
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer

gender_indexer = StringIndexer(inputCol='Sex',outputCol='SexIndex')
gender_encoder = OneHotEncoder(inputCol='SexIndex',outputCol='SexVec')

embark_indexer = StringIndexer(inputCol='Embarked',outputCol='EmbarkIndex')
embark_encoder = OneHotEncoder(inputCol='EmbarkIndex',outputCol='EmbarkVec')

assembler = VectorAssembler(inputCols=['Pclass',
 'SexVec',
 'Age',
 'SibSp',
 'Parch',
 'Fare',
 'EmbarkVec'],outputCol='features')

from pyspark.ml import Pipeline
ppl = Pipeline(stages=[gender_indexer,embark_indexer,
                       gender_encoder,embark_encoder,
                       assembler])
data_titanic = ppl.fit(my_final_data).transform(my_final_data)

data_titanic.show()

# In[9]:
# ### Build Logistic Regression model
from pyspark.ml.classification import LogisticRegression

train_titanic_data, test_titanic_data = data_titanic.randomSplit([0.7, 0.3])
log_reg_titanic = LogisticRegression(featuresCol='features',labelCol='Survived')
logr_titanic = log_reg_titanic.fit(train_titanic_data)

results = logr_titanic.transform(test_titanic_data)
results.select('Survived','prediction').show()


# In[10]:
# #### Cost Matrix

from sklearn.metrics import confusion_matrix
y_true = results.select("Survived")
y_true = y_true.toPandas()

y_pred = results.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred)
print(cnf_matrix)

print(y_true['Survived'].value_counts())
print(y_pred['prediction'].value_counts())
