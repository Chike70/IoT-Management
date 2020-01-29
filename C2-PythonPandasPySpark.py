# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 18:30:01 2020

@author: wchen
"""

# In[0]:
import numpy

numpy.random.seed(11)
no1 = numpy.random.randint(0, 100)
no2 = numpy.random.randint(0, 100)
no3 = numpy.random.randint(0, 100)
print("  Random No. 1: ", no1)
print("  Random No. 2: ", no2)
print("  Random No. 3: ", no3)

# In[1]:

a = numpy.array(  [ 1, 21, 11, 2, 22, 12, 3, 51 ]  )
b = numpy.array(   [	[1, 21], [11, 2], [22, 12], [3, 51] ]   )
c = numpy.array( [	[1, 21, 11, 2, 22, 12, 3, 51] ] )

print(a.shape)
print(b.shape)
print(c.shape)

print(numpy.array(   [	[1], [21], [11], [2], [22], [12], [3], [51] ]   ).shape)

print(numpy.array(  [	[[1, 21], [11, 2]], [[22, 12], [3, 51]] ]  ).shape)

print(numpy.array(   [	[[1], [21], [11], [2]], [[22], [12], [3], [51]] ]   ).shape)


# In[2]:

print(b.flatten())

# In[3]:

pythonArray = [	[1, 21, 11],
            [2, 22, 12], 
			[3, 23, 13], 
			[4, 24, 14], 
			[5, 25, 15], 
			[6, 26, 5], 
			[7, 27, 4], 
			[8, 28, 3], 
			[9, 29, 2], 
			[10, 30, 1]]

print("  pythonArray: ", pythonArray)
numpyArray = numpy.array(pythonArray)
print("  numpyArray: ")
print(numpyArray)

print("  numpyArray.shape: ", numpyArray.shape)
print("  numpyArray.ndim : ", numpyArray.ndim)

# In[4]:
firstThreeRecord = numpyArray[0:3, :]
print("firstThreeRecord (with all columns): ")
print(firstThreeRecord)
	
lastTwoRecord = numpyArray[8:10, :]
print("lastTwoRecord (with all columns): ")
print(lastTwoRecord)

# In[5]:
firstTwoColumn = numpyArray[:, 0:2]
print("firstTwoColumn (with all rows): ")
print(firstTwoColumn)
	
lastColumn = numpyArray[:, 2:3]
print("lastColumn (with all rows): ")
print(lastColumn)
		
firstThreeRecordFirstTwoColumn = numpyArray[0:3, 0:2]
print("firstThreeRecordFirstTwoColumn: ")
print(firstThreeRecordFirstTwoColumn)	
    
# In[6]:
newNumpyArrayFlatten = numpyArray.flatten()
print("newNumpyArrayFlatten: ")
print(newNumpyArrayFlatten)


# In[7]:
pythonArray = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; 
print("  pythonArray: ", pythonArray)

numpyArray = numpy.array(pythonArray)

print("  numpyArray: ")
print(numpyArray)

print("  numpyArray.shape: ", numpyArray.shape)

# In[8]:

numpyArrayReshape = numpyArray.reshape(1, 10, 1)

print("  numpyArrayReshape: ")
print(numpyArrayReshape)

print("  numpyArrayReshape.shape: ", numpyArrayReshape.shape)


# In[9]:
pythonArray = [	[1, 10],
            [2, 9], 
			[3, 8], 
			[4, 7], 
			[5, 6], 
			[6, 5], 
			[7, 4], 
			[8, 3], 
			[9, 2], 
			[10, 1]]
print("  pythonArray : ", pythonArray)

numpyArray = numpy.array(pythonArray)

print("  numpyArray: ")
print(numpyArray)

print("  numpyArray.shape: ", numpyArray.shape)

# In[10]:
numpyArrayReshape = numpyArray.reshape(1, 10, 2)

print("  numpyArrayReshape: ")
print(numpyArrayReshape)

print("  numpyArrayReshape.shape: ", numpyArrayReshape.shape)

# In[11]:
import pandas
filename = "data-3-numeric.csv"
	
dataframe = pandas.read_csv(filename, header=None)
print("  dataframe:")
print(dataframe)
	
values_int = dataframe.values
print("  values_int: ")
print(values_int)
	
values_float = values_int.astype(float)	
print("  values_float: ")
print(values_float)


# In[12]:
filename = "data-3-numeric-1-string.csv"	
	
dataframe = pandas.read_csv(filename, header=None, dtype="str")
	
print("  dataframe: ")
print(dataframe)
	
dataset = dataframe.values
print("  dataset: ")
print(dataset)
	
X_str = dataset[:, 0:3]
X_float = X_str.astype(float)
Y_str = dataset[:, 3]
	
print("  X_str: ")
print(X_str)
print("  X_float: ")
print(X_float)
print("  Y_str: ")
print(Y_str)
    
# In[13]:
datasetX = pandas.read_csv(filename, header=None, dtype="float", usecols=range(0, 3))
datasetY = pandas.read_csv(filename, header=None, dtype="str", usecols=[3])
print("  datasetX: ")
print(datasetX)
print("  datasetY: ")
print(datasetY)

# In[14]:
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

# In[15]:





