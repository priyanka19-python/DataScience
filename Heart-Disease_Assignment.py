import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #for plotting
"""
1.Read in the data from a freely available source on the internet.  
2.Account for outlier values in numeric columns (at least 1 column).
3.Replace missing numeric data (at least 1 column).
4.Normalize numeric values (at least 1 column, but be consistent with numeric data).
5.Bin numeric variables (at least 1 column).
6.Consolidate categorical data (at least 1 column).
7.One-hot encode categorical data with at least 3 categories (at least 1 column).
8.Remove obsolete columns.
"""
#https://www.kaggle.com/tanersekmen/basic-analyzing-of-heart-disease --Categorical
#https://www.kaggle.com/fabijanbajo/heart-disease-prediction

 #1.****Read in the data from a freely available source on the internet. 
print("1.Read in the data") 
Heart = pd.read_csv('C:\Priyanka\DS\Project\Milestone2\heart-disease-uci/heartV1.csv')
Heart.head()
Heart.columns
Heart.info()
# Check the data types
print('Data type:',Heart.dtypes)
print('Shape:',Heart.shape)
Heart.isnull().any()
#This graph shows how many target there are in data.
sns.countplot(x="target", data=Heart, palette="deep")
plt.show()
sns.countplot(x='sex', data=Heart, palette="pastel")
plt.xlabel("Sex (0 = female, 1= male)") # Representing values 
plt.show()
#It gives male and female relationship

threshold = sum(Heart.age)/len(Heart.age)
print(threshold)
Heart["age_situation"] = ["old" if i > threshold else "young" for i in Heart.age]
Heart.loc[:10,["age_situation","age"]]
#Calculator of Age Scale and gives information which one old or young for this illness

sns.barplot(x=Heart.age.value_counts()[:10].index,y=Heart.age.value_counts()[:10].values)
plt.xlabel('Age')
plt.ylabel('Number of People')
plt.title('Age Calculation')
plt.show()
#Bar Plot and this is providing distribution of age
#######################****************************************888
print("###############2.Account for outlier values in numeric columns" ) 
#********2.Account for outlier values in numeric columns (at least 1 column)
Heart.loc[:, "age"].unique()
sort_by_age = Heart.sort_values("age")
#copy age column ro x
x=Heart.loc[:, "age"]

# The high limit for acceptable values is the mean plus 2 standard deviations    
LimitHi = np.mean(x) + 2*np.std(x)
# The high limit is the cutoff for good values
LimitHi

# The low limit for acceptable values is the mean plus 2 standard deviations
LimitLo = np.mean(x) - 2*np.std(x)
# The low limit is the cutoff for good values
LimitLo
# Create Flag for values within limits 
FlagGood = (x >= LimitLo) & (x <= LimitHi)
# What type of variable is FlagGood? Check the Variable explorer.

# present the flag
FlagGood
#######################

# We can present the values of the items within the limits
x[FlagGood]

# Overwrite x with the selected values
x = x[FlagGood]

# present the data set
print("Dataset after removing outliers from age column:")
x
#sort_by_age = Heart.sort_values("age")
#######################
print("#############3.Replace missing numeric data")
# 3.******Replace missing numeric data (at least 1 column).*********
#################
# Check the unique values
Heart.loc[:, "chol"].unique()
##################
# Check the number of rows and columns
Heart.shape
# Replace >Question Marks< with NaNs
Heart = Heart.replace(to_replace="?", value=float("NaN"))

# Corece to numeric and impute medians for Chol column
Heart.loc[:, "chol"] = pd.to_numeric(Heart.loc[:, "chol"], errors='coerce')
HasNan = np.isnan(Heart.loc[:,"chol"])
Heart.loc[HasNan, "chol"] = np.nanmedian(Heart.loc[:,"chol"])
##############
# Count NaNs
Heart.isnull().sum()
y=Heart.loc[:,"chol"]
# Check distinct values and verify if there are any missing values
Heart.loc[:,"chol"].unique()
####################
#4.Normalize numeric values (at least 1 column, but be consistent with numeric data).
x=Heart.loc[:,"chol"]
offset = np.mean(x)
spread = np.std(x)
xNormZ = (x - offset)/spread
print(" The Z-normalized variable:[Chol column]", xNormZ)

############*****************######################################
print("5.Bin numeric variables (at least 1 column")
#********5.Bin numeric variables (at least 1 column).
#  2 ]Equal-width Binning
# Determine the boundaries of the bins
NumberOfBins = 3
BinWidth = (max(y) - min(y))/NumberOfBins
MinBin1 = float('-inf')
MaxBin1 = min(y) + 1 * BinWidth
MaxBin2 = min(y) + 2 * BinWidth
MaxBin3 = float('inf')

print(" Bin 1 is greater than", MinBin1, "up to", MaxBin1)
print(" Bin 2 is greater than", MaxBin1, "up to", MaxBin2)
print(" Bin 3 is greater than", MaxBin2, "up to", MaxBin3)
# Create the categorical variable
# Start with an empty array that is the same size as x
xBinnedEqW = np.empty(len(x), object) 
# np.full(len(x), "    ")

# The conditions at the boundaries should consider the difference 
# between less than (<) and less than or equal (<=) 
# and greater than (>) and greater than or equal (>=)
xBinnedEqW[(x > MinBin1) & (x <= MaxBin1)] = "Low"
xBinnedEqW[(x > MaxBin1) & (x <= MaxBin2)] = "Med"
xBinnedEqW[(x > MaxBin2) & (x <= MaxBin3)] = "High"
print(" x binned into 3 equal-width bins:", xBinnedEqW)
print("6.Consolidate categorical data")
##********6.Consolidate categorical data (at least 1 column).
Heart.loc[:,"thal"].unique()
Heart.loc[:, "thal"].astype(str)
# Check the data types
Heart.dtypes
#convert "Thal" column to string
#add for value 0
Heart.loc[:,"thal"] = Heart.loc[:,"thal"].astype(str)
Heart.loc[Heart.loc[:, "thal"] == "1", "thal"] = "fixed defect"
Heart.loc[Heart.loc[:, "thal"] == "2", "thal"] = "normal"
Heart.loc[Heart.loc[:, "thal"] == "3", "thal"] = "reversable defect"

# Check the first rows of the data frame
Heart.head(10)
print("7.One-hot encode categorical data with at least 3 categories")
##********7.One-hot encode categorical data with at least 3 categories (at least 1 column).

# Check the unique values
Heart.loc[:, "slope"].unique()
Heart.loc[:,"slope"] = Heart.loc[:,"slope"].astype(str)
Heart.loc[ Heart.loc[:, "slope"] == "0", "slope"] = "downsloping"
Heart.loc[Heart.loc[:, "slope"] == "1", "slope"] = "flat"
Heart.loc[Heart.loc[:, "slope"] == "2", "slope"] = "upsloping"
Heart.loc[:, "slope"]
Heart.columns
# Create 3 new columns, one for each state in "Shape"
Heart.loc[:, "downsl"] = (Heart.loc[:, "slope"] == "downsloping").astype(int)
Heart.loc[:, "flat"] = (Heart.loc[:, "slope"] == "flat").astype(int)
Heart.loc[:, "upsl"] = (Heart.loc[:, "slope"] == "upsloping").astype(int)
print("8.Remove obsolete column slope.")
##*******8.Remove obsolete column slope.
print("Removing column slope from the dataset:")
Heart = Heart.drop("slope", axis=1)
print("Columns in the dataset after removing slope column:",Heart.columns)
##############

