
"""
Name : Priyanka Kshatriya
Assignment 08
Short narrative on the data preparation :
This project aims to generate a model to predict the presence of a heart disease.
The UCI heart disease database contains 76 attributes, but all published experiments 
refer to using a subset of 14. 
The target attribute is an integer valued from 0 (no presence) to 4.
Attribute Information and distribution :
1. age 
2. sex 
3. chest pain type (4 values) 
4. resting blood pressure 
5. serum cholesterol in mg/dl 
6. fasting blood sugar > 120 mg/dl
7. resting electrocardiographic results (values 0,1,2)
8. maximum heart rate achieved 
9. exercise induced angina 
10. oldpeak = ST depression induced by exercise relative to rest 
11. the slope of the peak exercise ST segment 
12. number of major vessels (0-3) colored by flourosopy 
13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
14. target yes/no diagonised 
"""
#import  python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import matplotlib.pyplot as plt
import matplotlib

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
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
# 1.#####################Importing the dataset#################333
print("Read dataset from online : \n")
url='https://raw.githubusercontent.com/priyanka19-python/DataScience/master/heartV1.csv'
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
Heart = pd.read_csv(url, header=None) 
Heart = pd.read_csv(url, error_bad_lines=False)
Heart.head()
Heart.columns
Heart.info()
dataset=Heart
#######################understanding data#####################

rcParams['figure.figsize'] = 20, 14
plt.matshow(dataset.corr())
plt.yticks(np.arange(dataset.shape[1]), dataset.columns)
plt.xticks(np.arange(dataset.shape[1]), dataset.columns)
plt.colorbar()
#Histogram
dataset.hist()

#Count of  target the two classes are not exactly 50% each but the ratio is good enough to 
#continue without dropping/increasing our data.
rcParams['figure.figsize'] = 8,6
plt.bar(dataset['target'].unique(), dataset['target'].value_counts(), color = ['red', 'green'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')
plt.show()

x = dataset.iloc[:, [0, 4]].values

#2a********Normalize numeric values *****************************
x=Heart.loc[:,"cp"]
offset = np.mean(x)
spread = np.std(x)
xNormZ = (x - offset)/spread
print(" The Z-normalized variable:[cp column]\n", xNormZ)

#2b****** Outlier in column**************************
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

#2c remove or replace missing values
print("Replace missing values :/n")
Heart.loc[:, "chol"].unique()
# Check the number of rows and columns
Heart.shape
# Replace >Question Marks< with NaNs
Heart = Heart.replace(to_replace="?", value=float("NaN"))
# Corece to numeric and impute medians for Chol column
Heart.loc[:, "chol"] = pd.to_numeric(Heart.loc[:, "chol"], errors='coerce')
HasNan = np.isnan(Heart.loc[:,"chol"])
Heart.loc[HasNan, "chol"] = np.nanmedian(Heart.loc[:,"chol"])
print("After replacing the values :\n ",Heart.loc[:, "chol"].unique())

#2d**One-hot encode some categorical columns with 3 or more categories

print("One-hot encode  foe slope column :\n")
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

################Unsupervised Learning##################################
# 3] Ask a binary-choice question that describes your classification. Write the question as a comment.
# Specify an appropriate column as your expert label for a classification (include decision comments).
"""Target variable is binary variable  whether a patient has heart diesease .Target column is expert label """

#4]Apply K-Means on some of your columns, but make sure you do not use the expert label. 
#Add the K-Means cluster labels to your dataset.
print ("K-Means cluster labels on the  dataset.\n")
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

#############Supervised Learning###############################
"""
Supervised Learning:

Ask a binary-choice question that describes your classification. Write the question as a comment. 
Split your data set into training and testing sets using the proper function in sklearn. 
Use sklearn to train two classifiers on your training set, like logistic regression and random forest. 
Apply your (trained) classifiers to the test set. 
Create and present a confusion matrix for each classifier. Specify and justify your choice of probability threshold. 
For each classifier, create and present 2 accuracy metrics based on the confusion matrix of the classifier. 
For each classifier, calculate the ROC curve and it's AUC using sklearn. Present the ROC curve. Present the AUC in the ROC's plot.
"""
# 5]Split your data set into training and testing sets using the proper function split dataset.
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
    #model.fit(OldInputs, OldTarget)                          
    knn_scores.append(knn_classifier.score(X_test, y_test))
    print ("predictions for test set:")
    print ( knn_classifier.predict(X_test))
    print ('actual class values:')
    print (y_test)

plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')
plt.show()

#From the plot above, it is clear that the maximum score achieved was 0.87 for  8 neighbors.


#Determine accuracy rate, which is the number of correct predictions divided by the
#total number of predictions .
print("The score for K Neighbors Classifier is {}% with {} nieghbors.".format(knn_scores[7]*100, 8))
print ("Probability estimates:\n",knn_classifier.predict_proba(X_test))
y_pred = knn_classifier.predict(X_test)
prob= knn_classifier.predict_proba(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
CM = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = CM.ravel()
print ("TP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(y_test, y_pred)
print ("Accuracy rate:", np.round(AR, 2))
P = precision_score(y_test, y_pred)
print ("Precision:", np.round(P, 2))
R = recall_score(y_test, y_pred)
print ("Recall:", np.round(R, 2))

 # False Positive Rate, True Posisive Rate, probability thresholds
fpr, tpr, th = roc_curve(y_test, y_pred)
AUC = auc(fpr, tpr)

plt.rcParams["figure.figsize"] = [8, 8] # Square
font = {'family' : 'normal', 'weight' : 'bold', 'size' : 18}
matplotlib.rc('font', **font)
plt.figure()
plt.title('ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr, tpr, LW=3, label='ROC curve (AUC = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
plt.legend(loc="lower right")
plt.show()
#8]Write out to a csv a dataframe of the test data, including actual outcomes, 
#and the probabilities of your classification.

data_test = pd.DataFrame(data=X_test, columns=['f{}'.format(i) for i in range(1, 11)])
data_test['y_test'] = y_test
data_test['y_pred'] = knn_classifier.predict(X_test)


data_test.columns
print("KNN classification output to csv :\n")
data_test.to_csv('Priyanka_Lesson08assignment.csv', sep=',')

"""
Observations:
Gender has a strong influence on the target variable, where men have a higher frequency of heart disease patiënts
Respondents with the typical or asymptomatic chest pain type, seem to have the highest probability of getting a heart disease
A fasting blood sugar (fbs) level of higher or lower than 120 mg/dl is not a good predictor of the target
From the restecg (Resting electrocardiography results), a normal seems to have the highest impact on the target
The presence of exang (exercise induced angina) has a relative strong impact on the target
There is a high decrease in the probability of having a heart disease where respondents have an upsloping peak excercise slope
thal might be a good predictor of the target variable
 #####################################################3
Applied Machine Learning algorithm K Neighbors Classifier. K Neighbors Classifier achieved the 
highest score of 87% with 8 nearest neighbors.
"""
##############ROC with mutliple classifier #####


# Instantiate the classfiers and make a list
classifiers = [LogisticRegression(random_state=1234), 
               GaussianNB(), 
               KNeighborsClassifier(), 
               DecisionTreeClassifier(random_state=1234),
               RandomForestClassifier(random_state=1234)]

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

# Train the models and record the results
for cls in classifiers:
    model = cls.fit(X_train, y_train)
    yproba = model.predict_proba(X_test)[::,1]
    
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)
#Plot the figure
fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()

fig.savefig('multiple_roc_curve.png')



















