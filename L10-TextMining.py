# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 19:58:19 2020

@author: wchen
"""

# In[1]:
# Read Text Data
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('text mining').getOrCreate()
data = spark.read.csv("D:/#DevCourses-GWU/#5_IoT_BigData/SMSSpamCollection.csv", inferSchema=True, sep='\t')
data = data.withColumnRenamed('_c0','class').withColumnRenamed('_c1','text')
data.show(truncate=False)


# In[2]:
# Count number of Words in each Text
from pyspark.sql.functions import length
data = data.withColumn('length', length(data['text']))
data.show()

# In[3]:
# Compare the lenght difference between ham and spam
data.groupby('class').mean().show()

# In[4]:
# Treat TF-IDF features for each text
# TF: Term Frequency
# IDF: Inverse Document Frequency
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer, VectorAssembler

tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')
idf = IDF(inputCol="c_vec", outputCol="tf_idf")
ham_spam_to_num = StringIndexer(inputCol='class',outputCol='label')
final_feature = VectorAssembler(inputCols=['tf_idf', 'length'],outputCol='features')

from pyspark.ml import Pipeline
data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,count_vec,idf,final_feature])
clean_data = data_prep_pipe.fit(data).transform(data)

clean_data.show()
clean_data.take(1)
clean_data.take(1)[0][-1]


# In[5]: 
# ## Split data into training and test datasets
training, test = clean_data.randomSplit([0.8, 0.2], seed=12345)

# Build Logistic Regression Model
from pyspark.ml.classification import LogisticRegression

log_reg = LogisticRegression(featuresCol='features', labelCol='label')
logr_model = log_reg.fit(training)

results = logr_model.transform(test)
results.select('label','prediction').show()


# In[10]:
# #### Confusion Matrix
from sklearn.metrics import confusion_matrix
y_true = results.select("label")
y_true = y_true.toPandas()

y_pred = results.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred)
print(cnf_matrix)
print("Prediction Accuracy is ", (cnf_matrix[0,0]+cnf_matrix[1,1])/sum(sum(cnf_matrix)) )
