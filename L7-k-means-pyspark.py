# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 11:42:53 2020
# Kmeans with PySpark
@author: wchen
"""

# # Clustering Code Along
# 
# We'll be working with a real data set about seeds, from UCI repository: https://archive.ics.uci.edu/ml/datasets/seeds.

# The examined group comprised kernels belonging to three different varieties of wheat: Kama, Rosa and Canadian, 70 elements each, randomly selected for 
# the experiment. High quality visualization of the internal kernel structure was detected using a soft X-ray technique. It is non-destructive and considerably cheaper than other more sophisticated imaging techniques like scanning microscopy or laser technology. The images were recorded on 13x18 cm X-ray KODAK plates. Studies were conducted using combine harvested wheat grain originating from experimental fields, explored at the Institute of Agrophysics of the Polish Academy of Sciences in Lublin. 
# 
# The data set can be used for the tasks of classification and cluster analysis.
# 
# 
# Attribute Information:
# 
# To construct the data, seven geometric parameters of wheat kernels were measured: 
# 1. area A, 
# 2. perimeter P, 
# 3. compactness C = 4*pi*A/P^2, 
# 4. length of kernel, 
# 5. width of kernel, 
# 6. asymmetry coefficient 
# 7. length of kernel groove. 
# All of these parameters were real-valued continuous.
# 
# Let's see if we can cluster them in to 3 groups with K-means!

# In[1]:

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
spark = SparkSession.builder.appName('k-means clustering').getOrCreate()
# Loads data
dataset = spark.read.csv("seeds_dataset.csv",header=True,inferSchema=True)
dataset.head()
dataset.describe().show()


# In[2]:
# ## Format the Data
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

print(dataset.columns)
vec_assembler = VectorAssembler(inputCols = dataset.columns, outputCol='features')
feature_data = vec_assembler.transform(dataset)


# In[3]:
# ## Scale the Data
# It is a good idea to scale our data to deal with the curse of dimensionality: https://en.wikipedia.org/wiki/Curse_of_dimensionality
from pyspark.ml.feature import StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(feature_data)
# Normalize each feature to have unit standard deviation.
final_data = scalerModel.transform(feature_data)


# In[4]:
# ## Train the Model and Evaluate
from pyspark.ml.clustering import KMeans
# Trains a k-means model.
kmeans = KMeans(featuresCol='scaledFeatures', k=3)
model = kmeans.fit(final_data)

# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse = model.computeCost(final_data)
print("Within Set Sum of Squared Errors = " + str(wssse))


# In[5]:
# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# Predict the label of each seed
model.transform(final_data).select('prediction').show()
# # Great Job!

# In[6]:
clusters = model.transform(final_data).select('*')
clusters.groupBy("prediction").count().orderBy(F.desc("count")).show()
clusters.show()
clusters_pd = clusters.toPandas()

import seaborn as sns
ax = sns.scatterplot(x="area", y="compactness", hue="prediction", data=clusters_pd)


# In[7]:
# A Second Example
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("Read Big Data Example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
    
ds = spark.read.format('com.databricks.spark.csv').\
                            options(header='true', \
                               inferschema='true').\
                                    load("2019-Nov.csv",header=True)     
ds.show(5)                                         
              
ds[ds.brand=="jessnail"].show()
ds.select("brand").distinct().show(1000) 
ds.groupBy("brand").count().orderBy(F.desc("count")).show()

ds.select("brand").distinct().count()
ds.select("user_id").distinct().count()


# In[8]:
# Make Features for View and Cart

alan = ds.select(["user_id","event_type"]).groupBy("user_id","event_type").count()
alan.show(5)

table1 = alan.filter(alan['event_type']=='view')
table1 = table1.withColumnRenamed("event_type", "view_type")
table1 = table1.withColumnRenamed("count", "view_count")
#table1 = table1.select("user_id", F.col("event_type").alias("view_type"), F.col("count").alias("view_count"))
table1.show(5)
table2 = alan.filter(alan['event_type']=='cart')
table2 = table2.withColumnRenamed("event_type", "cart_type")
table2 = table2.withColumnRenamed("count", "cart_count")
#table2 = table2.select("user_id", F.col("event_type").alias("cart_type"), F.col("count").alias("cart_count"))
table2.show(5)
table3 = table1.join(table2, on=['user_id'], how='outer')
table3.show(5)

# fill in null
table3 = table3.fillna({ 'view_type': 'view', 'view_count':0, 'cart_type': 'cart', 'cart_count':0 })
table3.show(5)
table1.unpersist()
table2.unpersist()

# In[9]:
# Make Features for Brand
alan = ds.select(["user_id","brand"]).groupBy("user_id","brand").count()
alan.show(5)

brand_ranking = ds.groupBy("brand").count().orderBy(F.desc("count"))
brand_ranking.show()
brand_names = brand_ranking.select("brand").collect()
brand_names = brand_names[1:6]

table1 = alan.filter(alan['brand']==brand_names[0][0])
table1 = table1.withColumnRenamed("count", brand_names[0][0])
table1 = table1.drop('brand')
table1.show(5)
for i in range(1, len(brand_names)):
    table2 = alan.filter(alan['brand']==brand_names[i][0])
    try:
        table2 = table2.withColumnRenamed("count", brand_names[i][0])
        table2 = table2.drop('brand')
        table1 = table1.join(table2, on=['user_id'], how='outer')
    except:
        print("This brand is not existed", brand_names[i][0])
    print(i)

table1.show(5)
table4 = table1.fillna(0)
table1.unpersist()
table2.unpersist()
table4.show(5)

# In[10]:
# Make Features for Price
totalPrice = ds.select(["user_id","price"]).groupBy("user_id").sum()
totalPrice = totalPrice.drop('sum(user_id)')
totalPrice.show(5)
meanPrice = ds.select(["user_id","price"]).groupBy("user_id").mean()
meanPrice = meanPrice.drop('avg(user_id)')
meanPrice.show(5)

table5 = totalPrice.join(meanPrice, on=['user_id'], how='outer')
table5 = table5.fillna(0)
table5.show(5)


# In[11]:
# Combine All Columns
table_final = table3.join(table4, on=['user_id'], how='outer')
table_final = table_final.join(table5, on=['user_id'], how='outer')
table_final = table_final.fillna(0)
table_final.show(5)


# In[12]:
# ## Format the Data
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

table_final = table_final.withColumnRenamed('bpw.style', 'bpw_style')
print(table_final.columns)
vec_assembler = VectorAssembler(inputCols = ['view_count','cart_count',
            'runail','irisk','grattol','masura', 'bpw_style', 
            'sum(price)', 'avg(price)'], outputCol='features')
feature_data = vec_assembler.transform(table_final)

# ## Scale the Data
# It is a good idea to scale our data to deal with the curse of dimensionality: https://en.wikipedia.org/wiki/Curse_of_dimensionality
from pyspark.ml.feature import StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(feature_data)
# Normalize each feature to have unit standard deviation.
final_data = scalerModel.transform(feature_data)

# ## Train the Model and Evaluate
from pyspark.ml.clustering import KMeans
# Trains a k-means model.
kmeans = KMeans(featuresCol='scaledFeatures', k=10)
model = kmeans.fit(final_data)

# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse = model.computeCost(final_data)
print("Within Set Sum of Squared Errors = " + str(wssse))

# In[13]:
# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

"""
Cluster Centers: 
[0.12848912 0.10515011 0.05943032 0.05882532 0.04934216 0.02851872
 0.03562338 0.12179758 0.26447539]
[1.62960591 2.27170762 1.47987743 1.92464732 1.03469603 0.87907283
 0.73123557 2.26787059 0.27089544]
[8.61498048e-02 7.28577955e-03 1.49099538e-02 1.82613019e-02
 1.35126726e-03 5.25244113e-04 7.72944451e-04 1.10507221e+00
 5.69237868e+00]
[ 4.70109783  7.5300588   4.19063882  3.74216854  1.53729493 29.02312139
  1.96946773  4.73807204  0.1254464 ]
[ 4.48011665  5.05690202  2.358661    2.2332741  20.61600802  1.12192474
  2.25025749  4.98534234  0.18819486]
[10.25571666 15.21002938 11.89095778 20.36430084  4.30254038  3.81470474
  6.52590851 16.32770342  0.26981907]
[ 2.65576477  3.99853171  1.55363076  2.07596682  1.28426382  0.90351688
 11.83429241  2.28959772  0.12039254]
[2.60822824e+02 5.43106629e+01 1.01571459e+00 0.00000000e+00
 1.05836021e-01 0.00000000e+00 1.00344615e+01 5.11297595e+01
 1.05960591e-01]
[ 3.57160203  5.89601208 10.11761527  4.47534179  1.60053472  1.61230745
  1.65175192  4.45790969  0.1783953 ]
[0.10579869 0.01815489 0.04832014 0.05267022 0.00526731 0.0029164
 0.00242624 0.49879117 1.99763664]
"""


# Predict the label of each seed
model.transform(final_data).select('prediction').show()

# Now you are ready for your howework!