# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('kidney_disease.csv')

#Encoding the Categorical data
dataset['rbc'] = dataset['rbc'].map({'normal':1,'abnormal':0})
dataset['pc'] = dataset['pc'].map({'normal':1,'abnormal':0})
dataset['pcc'] = dataset['pcc'].map({'present':1,'notpresent':0})
dataset['ba'] = dataset['ba'].map({'present':1,'notpresent':0})
dataset['htn'] = dataset['htn'].map({'yes':1,'no':0})
dataset['dm'] = dataset['dm'].map({'yes':1,'no':0})
dataset['cad'] = dataset['cad'].map({'yes':1,'no':0})
dataset['appet'] = dataset['appet'].map({'good':1,'poor':0})
dataset['pe'] = dataset['pe'].map({'yes':1,'no':0})
dataset['ane'] = dataset['ane'].map({'yes':1,'no':0})
dataset['classification'] = dataset['classification'].map({'ckd':1,'notckd':0})

#Calculate number of missing values
dataset.isnull().sum()

#Filling in missing data
dataset = dataset.replace('?', np.NaN)
dataset = dataset.replace('\t?', np.NaN)

dataset['pcv'] = dataset['pcv'].astype(float)
dataset['pcv'].fillna((dataset['pcv'].mean()), inplace=True)

dataset['wc'] = dataset['wc'].astype(float)
dataset['wc'].fillna((dataset['wc'].mean()), inplace=True)

dataset['rc'] = dataset['rc'].astype(float)
dataset['rc'].fillna((dataset['rc'].mean()), inplace=True)

dataset['age'].fillna((dataset['age'].mean()), inplace=True)
dataset['age'] = dataset['age'].astype(int)

dataset['bp'].fillna((dataset['bp'].mean()), inplace=True)
dataset['bp'] = dataset['bp'].astype(int)

dataset['bgr'].fillna((dataset['bgr'].mean()), inplace=True)
dataset['bgr'] = dataset['bgr'].astype(int)

dataset['bu'].fillna((dataset['bu'].mean()), inplace=True)
dataset['bu'] = dataset['bu'].astype(int)

dataset['sc'].fillna((dataset['sc'].mean()), inplace=True)
dataset['sc'] = dataset['sc'].astype(int)

dataset['sod'].fillna((dataset['sod'].mean()), inplace=True)
dataset['sod'] = dataset['sod'].astype(int)

dataset['pot'].fillna((dataset['pot'].mean()), inplace=True)
dataset['pot'] = dataset['pot'].astype(int)

dataset['hemo'].fillna((dataset['hemo'].mean()), inplace=True)
dataset['hemo'] = dataset['hemo'].astype(int)

dataset['sg'] = dataset['sg'].fillna(dataset['sg'].value_counts().index[0])
dataset['al'] = dataset['al'].fillna(dataset['al'].value_counts().index[0])
dataset['su'] = dataset['su'].fillna(dataset['su'].value_counts().index[0])
dataset['rbc'] = dataset['rbc'].fillna(dataset['rbc'].value_counts().index[0])
dataset['pc'] = dataset['pc'].fillna(dataset['pc'].value_counts().index[0])
dataset['pcc'] = dataset['pcc'].fillna(dataset['pcc'].value_counts().index[0])
dataset['ba'] = dataset['ba'].fillna(dataset['ba'].value_counts().index[0])
dataset['htn'] = dataset['htn'].fillna(dataset['htn'].value_counts().index[0])
dataset['dm'] = dataset['dm'].fillna(dataset['dm'].value_counts().index[0])
dataset['cad'] = dataset['cad'].fillna(dataset['cad'].value_counts().index[0])
dataset['appet'] = dataset['appet'].fillna(dataset['appet'].value_counts().index[0])
dataset['pe'] = dataset['pe'].fillna(dataset['pe'].value_counts().index[0])
dataset['ane'] = dataset['ane'].fillna(dataset['ane'].value_counts().index[0])

#Calculating number of missing values again
dataset.isnull().sum()

#Dropping rows with no class
dataset = dataset.dropna()

#Seperating Independent and Dependent features
X = dataset.iloc[:, 1:25].values
y = dataset.iloc[:, 25].values

# Splitting the dataset into the Training set and Test set with Seed 339
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 339)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier_lg = LogisticRegression(random_state = 0, solver = 'liblinear')
classifier_lg.fit(X_train, y_train)

#Predicting the Test set results
y_pred_lg = classifier_lg.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_lg = confusion_matrix(y_test, y_pred_lg) 

precision_lg = cm_lg[1][1]/(cm_lg[1][1] + cm_lg[0][1])
print("Precision for Logistic Regression is : ")
print(precision_lg)

recall_lg = cm_lg[1][1]/(cm_lg[1][1] + cm_lg[1][0])
print("Recall for Logistic Regression is : ")
print(recall_lg)

# Fitting KNN classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_knn.fit(X_train, y_train)

# Predicting the Test set results
y_pred_knn = classifier_knn.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)

precision_knn = cm_knn[1][1]/(cm_knn[1][1] + cm_knn[0][1])
print("Precision for KNN Classifier is : ")
print(precision_knn)

recall_knn = cm_knn[1][1]/(cm_knn[1][1] + cm_knn[1][0])
print("Recall for KNN Classifier is : ")
print(recall_knn)

print("Confusion Matrix for Logistic Regression is : ")
print(cm_lg)

f1_score_lg = 2*((precision_lg*recall_lg)/(precision_lg+recall_lg))
print("F1 score for Logistic Regression is : ")
print(f1_score_lg)

print("Confusion Matrix for KNN Classifier is : ")
print(cm_knn)
f1_score_knn = 2*((precision_knn*recall_knn)/(precision_knn+recall_knn))
print("F1 score for KNN Classifier is : ")
print(f1_score_knn)