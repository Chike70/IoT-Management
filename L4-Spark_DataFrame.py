
# In[1]:
"""Usually, there are two popular ways to create the RDD (Resilient Distributed Dataset): loading an external dataset, 
or distributing a set of collection of objects. The following examples show some simplest ways to create RDDs by using
parallelize() fucntion which takes an already existing collection in your program and pass the same to the Spark Context."""

# Pandas Data Size Limit <2 gigabytes
# PySpark Data Size Limit >100 gigabytes
# There is no hard limit on the Data size you can process with PySpark, however when your RDD size 
# exceeds the size of your RAM then the data will be moved to Disk. Even after the data is moved to the Disk,
# Spark will be equally capable of processing it.

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("Python Spark create RDD example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
df = spark.sparkContext.parallelize([(1, 2, 3, 'a b c'),
                                     (4, 5, 6, 'd e f'),
                                     (7, 8, 9, 'g h i')]).toDF(['col1', 'col2', 'col3','col4'])

df.show()

# In[2]:
from pyspark.sql import SparkSession
spark = SparkSession \
        .builder \
        .appName("Python Spark create RDD example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
myData = spark.sparkContext.parallelize([(1,2), (3,4), (5,6), (7,8), (9,10)])

myData.collect()

# In[3]:
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark create RDD example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

Employee = spark.createDataFrame([
                        ('1', 'Joe',   '70000', '1'),
                        ('2', 'Henry', '80000', '2'),
                        ('3', 'Sam',   '60000', '2'),
                        ('4', 'Max',   '90000', '1')],
                        ['Id', 'Name', 'Sallary','DepartmentId']
                       )

Employee.show()
# In[4]:
# Use read and load functions
## set up  SparkSession
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark create RDD example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

ds = spark.read.format('com.databricks.spark.csv').\
                            options(header='true', \
                               inferschema='true').\
                load("D:/#DevCourses-GWU/#5_IoT_BigData/Advertising.csv",header=True)

ds.show(5)
ds.printSchema()

import pandas as pd
dp = pd.read_csv('D:/#DevCourses-GWU/#5_IoT_BigData/Advertising.csv')

# In[5]:
# Column Names
ds.columns

# Data Types
ds.dtypes

# In[6]:
# Fill Null
my_list = [['male', 1, None], ['female', 2, 3],['male', 3, 4]]
df = spark.createDataFrame(my_list, ['A', 'B', 'C'])
df.show()

df2 = df.fillna(-99)
df2.show()

# In[7]:
# Replace Value
df.na.replace(['male','female'],['1','0']).show()

# In[8]:
# Rename Columns
ds.show(4)

ds.withColumnRenamed('Newspaper','Paper').show(4)

# In[9]:
# Drop Columns
drop_name = ['Newspaper','Sales']
ds.drop(*drop_name).show(4)

# In[10]:
# Filter
ds[ds.Newspaper<20].show(4)

ds[(ds.Newspaper<20)&(ds.TV>100)].show(4)

# In[11]:
# With New Column
import pyspark.sql.functions as F
ds.withColumn('cond',F.when((ds.TV>100)&(ds.Radio<40),1)\
                      .when(ds.Sales>10, 2)\
                      .otherwise(3)).show(4)


ds.withColumn('tv_norm', ds.TV/ds.groupBy().agg(F.sum("TV")).collect()[0][0]).show(4)

#ds.groupBy().agg(F.sum("TV")).collect()[0][0]
#sum(dp.TV)

ds.withColumn('log_tv',F.log(ds.TV)).show(4)

ds.withColumn('tv+10', ds.TV+10).show(4)

# In[12]:
# Joins
leftp = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                    index=[0, 1, 2, 3])

rightp = pd.DataFrame({'A': ['A0', 'A1', 'A6', 'A7'],
                       'F': ['B4', 'B5', 'B6', 'B7'],
                       'G': ['C4', 'C5', 'C6', 'C7'],
                       'H': ['D4', 'D5', 'D6', 'D7']},
                       index=[4, 5, 6, 7])

lefts = spark.createDataFrame(leftp)
rights = spark.createDataFrame(rightp)

# Left Join
leftp.merge(rightp,on='A',how='left')
#
lefts.join(rights,on='A',how='left')\
     .orderBy('A',ascending=True).show()

# Right Join
leftp.merge(rightp,on='A',how='right')
#
lefts.join(rights,on='A',how='right')\
     .orderBy('A',ascending=True).show()

# Inner Join
leftp.merge(rightp,on='A',how='inner')
#
lefts.join(rights,on='A',how='inner')\
     .orderBy('A',ascending=True).show()
     
# Full Join
leftp.merge(rightp,on='A',how='outer')
#
lefts.join(rights,on='A',how='full')\
     .orderBy('A',ascending=True).show()     
    
# In[13]:
# Concatenate Columns
my_list = [('a', 2, 3),
           ('b', 5, 6),
           ('c', 8, 9),
           ('a', 2, 3),
           ('b', 5, 6),
           ('c', 8, 9)]
col_name = ['col1', 'col2', 'col3']

ds = spark.createDataFrame(my_list,schema=col_name)     
    
ds.withColumn('concat',F.concat('col1','col2')).show()


# In[14]:
# Group By

ds.groupBy(['col1']).agg({'col2': 'min', 'col3': 'avg'}).show()


# In[15]:
from pyspark.sql import SparkSession
# May take a little while on a local computer
spark = SparkSession.builder.appName("Saies_Info").getOrCreate()

df = spark.read.csv('D:/#DevCourses-GWU/#5_IoT_BigData/sales_info.csv',inferSchema=True,header=True)

df.printSchema()
df.show()
df.groupBy("Company").mean().show()
df.groupBy("Company").count().show()
df.groupBy("Company").max().show()
df.groupBy("Company").min().show()
df.groupBy("Company").sum().show()
df.agg({'Sales':'max'}).show()


import pyspark.sql.functions as F
df.select(F.countDistinct("Sales")).show()

df.select(F.avg('Sales')).show()

df.select(F.stddev("Sales")).show()


sales_std = df.select(F.stddev("Sales").alias('std'))
sales_std.select(F.format_number('std',2).alias('std')).show()

# OrderBy
df.orderBy("Sales").show()
df.orderBy(df["Sales"].desc()).show()

# In[16]:
# Pandas Data Size Limit <2 gigabytes
# PySpark Data Size Limit >100 gigabytes
# There is no hard limit on the Data size you can process with PySpark, however when your RDD size 
# exceeds the size of your RAM then the data will be moved to Disk. Even after the data is moved to the Disk,
# Spark will be equally capable of processing it.

# https://www.kaggle.com/mkechinov/ecommerce-behavior-data-from-multi-category-store
import pandas as pd
dp = pd.read_csv('D:/#DevCourses-GWU/#5_IoT_BigData/2019-Novs.csv')
dp.head()                 
         
import numpy as np   
np.sort(dp.brand.unique()[1:])

# In[17]:             
# https://www.kaggle.com/mkechinov/ecommerce-behavior-data-from-multi-category-store    
## set up  SparkSession
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("Read Big Data Example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
    
ds = spark.read.format('com.databricks.spark.csv').\
                            options(header='true', \
                               inferschema='true').\
                                    load("D:/#DevCourses-GWU/#5_IoT_BigData/2019-Nov.csv",header=True)     
ds.show(5)                                         
              
ds[ds.brand=="jessnail"].show()
ds.select("brand").distinct().show(1000)                           
