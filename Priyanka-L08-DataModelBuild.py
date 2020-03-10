#https://www.kaggle.com/amolbhivarkar/knn-for-classification-using-scikit-learn/notebook
#https://github.com/kb22/Heart-Disease-Prediction/blob/master/Heart%20Disease%20Prediction.ipynb
#https://www.kaggle.com/fabijanbajo/heart-disease-prediction
#https://www.educba.com/knn-algorithm/ to write comments
"""
Assignment 08
--Read in the dataset from a freely and easily available source on the internet.
Show data preparation. Normalize some numeric columns, one-hot encode some categorical columns with 3 or more categories, remove or replace missing values, remove or replace some outliers.
Ask a binary-choice question that describes your classification. Write the question as a comment. Specify an appropriate column as your expert label for a classification (include decision comments).
Apply K-Means on some of your columns, but make sure you do not use the expert label. Add the K-Means cluster labels to your dataset.
--Split your data set into training and testing sets using the proper function in sklearn (include decision comments).
Create a classification model for the expert label based on the training data (include decision comments).
Apply your (trained) classifiers to the test data to predict probabilities.
Write out to a csv a dataframe of the test data, including actual outcomes, and the probabilities of your classification.
Determine accuracy rate, which is the number of correct predictions divided by the total number of predictions (include brief preliminary analysis commentary).
"""
#import  python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
dataset = pd.read_csv('Heart.csv')
dataset.info()
dataset.describe()
#understanding data

rcParams['figure.figsize'] = 20, 14
plt.matshow(dataset.corr())
plt.yticks(np.arange(dataset.shape[1]), dataset.columns)
plt.xticks(np.arange(dataset.shape[1]), dataset.columns)
plt.colorbar()
#Histogram
dataset.hist()


rcParams['figure.figsize'] = 8,6
plt.bar(dataset['target'].unique(), dataset['target'].value_counts(), color = ['red', 'green'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')
plt.show()

#The two classes are not exactly 50% each but the ratio is good enough to 
#continue without dropping/increasing our data.



###************Finding the optimum number of clusters for k-means classification
# 1.Importing the dataset
url='https://raw.githubusercontent.com/priyanka19-python/DataScience/master/heartV1.csv'
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
Heart = pd.read_csv(url, header=None) 
Heart = pd.read_csv(url, error_bad_lines=False)
Heart.head()
Heart.columns
Heart.info()
dataset=Heart

x = dataset.iloc[:, [0, 4]].values

#2a********Normalize numeric values 
x=Heart.loc[:,"cp"]
offset = np.mean(x)
spread = np.std(x)
xNormZ = (x - offset)/spread
print(" The Z-normalized variable:[cp column]", xNormZ)

#2b****** Otlier in column age
print("###############2.Account for outlier values in numeric columns" ) 
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
print("Dataset after removing outliers from age column:\n ", x)

#2c
Heart.loc[:, "chol"].unique()
# Check the number of rows and columns
Heart.shape
# Replace >Question Marks< with NaNs
Heart = Heart.replace(to_replace="?", value=float("NaN"))
# Corece to numeric and impute medians for Chol column
Heart.loc[:, "chol"] = pd.to_numeric(Heart.loc[:, "chol"], errors='coerce')
HasNan = np.isnan(Heart.loc[:,"chol"])
Heart.loc[HasNan, "chol"] = np.nanmedian(Heart.loc[:,"chol"])
print("After replacing the values ",Heart.loc[:, "chol"].unique())

#2d**One-hot encode some categorical columns with 3 or more categories
# Check the unique values
Heart.loc[:, "slope"].unique()
Heart.loc[:,"slope"] = Heart.loc[:,"slope"].astype(str)
Heart.loc[ Heart.loc[:, "slope"] == "0", "slope"] = "downsloping"
Heart.loc[Heart.loc[:, "slope"] == "1", "slope"] = "flat"
Heart.loc[Heart.loc[:, "slope"] == "2", "slope"] = "upsloping"
Heart.loc[:, "slope"]
# Create 3 new columns, one for each state in "Shape"
Heart.loc[:, "downsl"] = (Heart.loc[:, "slope"] == "downsloping").astype(int)
Heart.loc[:, "flat"] = (Heart.loc[:, "slope"] == "flat").astype(int)
Heart.loc[:, "upsl"] = (Heart.loc[:, "slope"] == "upsloping").astype(int)
print("After One-hot encode columns: \n",Heart.columns)
# 3] Ask a binary-choice question that describes your classification. Write the question as a comment.
# Specify an appropriate column as your expert label for a classification (include decision comments).

#4]Apply K-Means on some of your columns, but make sure you do not use the expert label. 
#Add the K-Means cluster labels to your dataset.
dataset=Heart
x = dataset.iloc[:, [0, 4]].values
x
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
   
#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Within Cluster Sum of Squares')
plt.show()
#Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)
y_kmeans
#Visualising the clusters
plt.title('Applied Kmeans with sklearn')
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Chol Mid ')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Chol Low')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Chol High')
#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
plt.legend()
plt.show()

# 5]Split your data set into training and testing sets using the proper function
# in sklearn (include decision comments).
#split dataset
dataset.dtypes
####***************Data Processing
dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
dataset.columns
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

y = dataset['target']
X = dataset.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
#K Neighbors Classifier
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train, y_train)
    knn_scores.append(knn_classifier.score(X_test, y_test))

plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')
plt.show()


#From the plot above, it is clear that the maximum score achieved was 0.87 
#for the 8 neighbors.
print("The score for K Neighbors Classifier is {}% with {} nieghbors.".format(knn_scores[7]*100, 8))


#8]Write out to a csv a dataframe of the test data, including actual outcomes, 
#and the probabilities of your classification.
#Determine accuracy rate, which is the number of correct predictions divided by the
#total number of predictions (include brief preliminary analysis commentary).

data_test = pd.DataFrame(data=X_test, columns=['f{}'.format(i) for i in range(1, 11)])
data_test['y_test'] = y_test
data_test['y_pred'] = knn_classifier.predict(X_test)
data_test.columns
data_test.to_csv('KNNOutput_Lesson08assignment.csv', sep=',')




"""
Conclusion¶ 
In this project, I used Machine Learning to predict whether a person is 
suffering from a heart disease. After importing the data, 
I analysed it using plots. Then, I did generated dummy variables for 
categorical features and scaled other features. 
I then applied four Machine Learning algorithms, 
K Neighbors Classifier, Support Vector Classifier, Decision Tree Classifier 
and Random Forest Classifier. I varied parameters across each model to improve their scores.
 In the end, K Neighbors Classifier achieved the highest score of 87% with 8 nearest neighbors.
"""























