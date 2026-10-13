import pandas as pd

mydata = pd.read_csv("students data.csv")

print(mydata)
print("-------------------------------------------------------")
print("shape: ")
print(mydata.shape)

#1- Inspect the data with shape and isnull().sum().

print(mydata.isnull().sum())
#--------------------------------------------
#2- Remove the duplicates.

print("number of duplicates:", mydata.duplicated().sum() )
mydata = mydata.drop_duplicates()

print(mydata.shape)

#---------------------------------------------
#3- Fill the missing values with the median.
from sklearn.impute import SimpleImputer 
import numpy as np
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(mydata.iloc[:, 2:5])
mydata.iloc[:, 2:5] = imputer.transform(mydata.iloc[:, 2:5])

#-----------------------
#missing value after imputations
print(mydata.isnull().sum())

print("\n final data:")
print(mydata.head())
