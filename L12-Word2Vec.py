# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 20:16:03 2020

@author: wchen
"""
# In[1]:
# Load Very Huge Data about 5Gb
# Data can be downloaded from 
"""
https://transferxl.com/00j9pbsbF3jRDg
"""
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('word2vec').getOrCreate() 
data = spark.read.option("quote", "\"") \
                 .option("escape", "\"") \
                 .csv("D:/#DevCourses-GWU/#5_IoT_BigData/L12-Word2Vec/yelp_review.csv", inferSchema=True, sep=',', header=True)
data.show()
data.count()

# In[2]:
# random split, we just take a small porpotion for the demo
reviews_small, reviews_big = data.randomSplit([0.01, 0.99])
print(reviews_small.count()) # 66874
reviews_small.show()
reviews_small.toPandas().to_csv('D:/#DevCourses-GWU/#5_IoT_BigData/L12-Word2Vec/reviews_small.csv')

# In[3]:
# Text contains extra character b, which is not necessary
from pyspark.sql.functions import col, substring, length, expr
first_date = reviews_small.select(col('Date')).take(1)
print('Original Date String:', first_date[0][0])

# Remove the first two letters and the last letter
reviews_small = reviews_small.withColumn("NewDate", expr("substring(Date, 3, length(Date)-3)"))
first_date_new = reviews_small.select(col('NewDate')).take(1)
print('New Date String:',first_date_new[0][0])

# In[4]:
# Compare the Date strings, the latter one is more appropriate
from pyspark.sql.functions import date_format, col
reviews_small.select(col('Date')).show(5, truncate=False)      
reviews_small.select(date_format(col('Date'),"yyyy-MM-dd").alias('date')).show(5, truncate=False) 

reviews_small.select(col('NewDate')).show(5, truncate=False)      
reviews_small.select(date_format(col('NewDate'),"yyyy-MM-dd").alias('date')).show(5, truncate=False) 

 # In[5]:  
# Repeat removing not necessary b character in all first five columns        
reviews_small = reviews_small.withColumn("user_id", expr("substring(user_id, 3, length(user_id)-3)"))
reviews_small = reviews_small.withColumn("text", expr("substring(text, 3, length(text)-3)"))
reviews_small = reviews_small.withColumn("date", expr("substring(date, 3, length(date)-3)"))
reviews_small = reviews_small.withColumn("review_id", expr("substring(review_id, 3, length(review_id)-3)"))
reviews_small = reviews_small.withColumn("business_id", expr("substring(business_id, 3, length(business_id)-3)"))
reviews_small.show()
            
# In[6]:   
# Count Reviews by Date                  
date_table = reviews_small.select(date_format(col('NewDate'),"yyyy-MM-dd").alias('date'))                      
date_table.count()                      
date_table2 = date_table.groupBy("date").count().sort("date", ascending=True)
date_table2.show()

# Plot Review Counts along with Time
df = date_table2.toPandas()
import seaborn as sns
import pandas as pd
df['date'] = pd.to_datetime(df['date'])
ax = sns.lineplot(x="date", y="count", data=df)
   
# In[7]:
# Tokenize the text
from pyspark.ml.feature import RegexTokenizer
regexTokenizer = RegexTokenizer(gaps = False, pattern = '\w+', inputCol = 'text', outputCol = 'text_token')
reviews_token = regexTokenizer.transform(reviews_small)
reviews_token.show(3)

# In[8]:
# Remove stopwords
from pyspark.ml.feature import StopWordsRemover
swr = StopWordsRemover(inputCol = 'text_token', outputCol = 'text_sw_removed')
reviews_swr = swr.transform(reviews_token)
reviews_swr.show(3)

# In[9]:
# Word Term Frequency
from pyspark.ml.feature import CountVectorizer
cv = CountVectorizer(inputCol="text_sw_removed", outputCol="tf")
cv_model = cv.fit(reviews_swr)
reviews_cv = cv_model.transform(reviews_swr)
reviews_cv.show(3)

# In[10]:
# TF-IDF
from pyspark.ml.feature import IDF
idf = IDF(inputCol="tf", outputCol="features")
idf_model = idf.fit(reviews_cv)
reviews_tfidf = idf_model.transform(reviews_cv)
reviews_tfidf.show(3)

# In[11]:
# Predict Rating Score (Repeat What we did in Lecture 10)
gradings = reviews_tfidf.select('funny','cool','useful','stars').toPandas()
sns.distplot(gradings['funny'])
sns.distplot(gradings['cool'])
sns.distplot(gradings['useful'])
sns.distplot(gradings['stars'])

from pyspark.ml.feature import StringIndexer
stringIdx = StringIndexer(inputCol="stars", outputCol="label")
final = stringIdx.fit(reviews_tfidf).transform(reviews_tfidf)
final.show(3)

# ## Split data into training and test datasets
training, test = final.randomSplit([0.8, 0.2], seed=12345)

# Build Logistic Regression Model
from pyspark.ml.classification import LogisticRegression

log_reg = LogisticRegression(featuresCol='features', labelCol='label')
logr_model = log_reg.fit(training)

results = logr_model.transform(test)
results.select('label','prediction').show()

# #### Confusion Matrix
from sklearn.metrics import confusion_matrix
y_true = results.select("label")
y_true = y_true.toPandas()

y_pred = results.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred)
print(cnf_matrix)
print("Prediction Accuracy is ", (cnf_matrix[0,0]+cnf_matrix[1,1])/sum(sum(cnf_matrix)) )
sns.heatmap(cnf_matrix, annot=True)

# In[12]:
# Fit & Train Word2Vec
from pyspark.ml.feature import Word2Vec

#create an average word vector for each document
word2vec = Word2Vec(vectorSize = 100, minCount = 5, inputCol = 'text_sw_removed', outputCol = 'result')
model = word2vec.fit(reviews_swr)
reviews_w2v = model.transform(reviews_swr)
reviews_w2v.show(3)

# In[13]:
#test similarity between words
synonyms = model.findSynonyms("great", 5)
synonyms.show(5)    

# In[14]:
# Find Similar Business
business_id = reviews_w2v.select('business_id').take(1)[0][0]
input_vec = reviews_w2v.select('result').filter(reviews_w2v['business_id'] == business_id).collect()[0][0]   

# Calculate cosine similarity between two vectors 
import numpy as np
from pyspark.sql.functions import udf
@udf("float")
def cossim_udf(v1): 
    v2 = input_vec
    similarity = np.dot(v1, v2) / np.sqrt(np.dot(v1, v1)) / np.sqrt(np.dot(v2, v2)) 
    return float(similarity)
similarity = reviews_w2v.select('business_id', cossim_udf('result').alias("similarity"), 'text')
similarity = similarity.orderBy("similarity", ascending = False)
similarity.show(truncate=False)
similarity.na.drop().show(truncate=False)


# In[15]:
# Recommend Business based on Keyword
key_word = "sushi"
docvecs = reviews_w2v
x = spark.createDataFrame([('newbusinessid', key_word)]).\
    withColumnRenamed('_1', 'business_id').\
    withColumnRenamed('_2', 'text')
x.show()
x_token = regexTokenizer.transform(x)
x_swr = swr.transform(x_token)
input_vec = model.transform(x_swr)
input_vec = input_vec.select('result').collect()[0][0]

# Calculate cosine similarity between two vectors 
from pyspark.sql.functions import udf
@udf("float")
def cossim_udf(v1): 
    v2 = input_vec
    similarity = np.dot(v1, v2) / np.sqrt(np.dot(v1, v1)) / np.sqrt(np.dot(v2, v2)) 
    return float(similarity)
similarity2 = reviews_w2v.select('business_id', cossim_udf('result').alias("similarity"), 'text')
similarity2 = similarity2.orderBy("similarity", ascending = False)
similarity2.na.drop().show(truncate=False)


