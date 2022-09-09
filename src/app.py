#Import libraries
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from collections import Counter

#Import data
df_raw = pd.read_csv('../data/raw/healthcare-dataset-stroke-data.csv') 
 
#BMI Missing Value and outliers
#Create Age bins 
# The labels of bins
labels = ['0 - 4','5 - 9','10 - 19','20 - 29', '30 - 39', '40 - 49', '50 - 59','60 - 69', '70 - +']
# Define the ages between bins
bins = [0,5,10,20,30,40, 50, 60, 70, np.inf]

# pd.cut each column, with each bin closed on left and open on right
df_raw['age_bins'] = pd.cut(df_raw['age'], bins=bins, labels=labels, right=False)

#Calculate the bmi value depend of age bins and gender. using mean value.
df_raw['bmi_new'] = df_raw.groupby(["age_bins","gender"])['bmi'].transform(lambda x: x.fillna(x.mean()))
#Set the value of missing value
df_raw['bmi'].fillna(df_raw['bmi_new'], inplace = True)

#Remove 2 outliers
df_raw.drop(df_raw[(df_raw['bmi'] > 80)].index, inplace=True)

#Transformation of category feature, and remove feature
df_raw['stroke']=df_raw['stroke'].astype(int)
df_raw['age']=df_raw['age'].astype(int)
df_raw['heart_disease']=df_raw['heart_disease'].astype(int)
df_raw['hypertension']=df_raw['hypertension'].astype(int)

# Encoding the 'Gender' column
df_raw['gender'] = df_raw['gender'].map({'Male': 0, 'Female' : 1, 'Other': 2})
df_raw['gender'] = df_raw['gender'].astype(int)

# Encoding the 'Residence_type' column
df_raw['Residence_type'] = df_raw['Residence_type'].map({'Urban': 0, 'Rural' : 1})
df_raw['Residence_type']=df_raw['Residence_type'].astype(int)

# Encoding the 'smoking status' column
df_raw['smoking_status'] = df_raw['smoking_status'].map({'Unknown': 0, 'never smoked' : 1, 'smokes': 2 , 'formerly smoked':3})
df_raw['smoking_status']=df_raw['smoking_status'].astype(int)

# Encoding the 'ever_married' column
df_raw['ever_married'] = df_raw['ever_married'].apply(lambda x: 1 if x == 'Yes' else 0)
df_raw['ever_married'] =df_raw['ever_married'].astype(int)

# Encoding the 'work_type' column
df_raw['work_type'] = df_raw['work_type'].map({'Private' : 0, 'Self-employed': 1, 'children': 2 , 'Govt_job':3, 'Never_worked':4})
df_raw['work_type'] =df_raw['work_type'].astype(int)

df_raw.drop(["age_bins","bmi_new","id"],axis=1,inplace=True)


#we define our labels and features
y = df_raw['stroke']
X = df_raw.drop('stroke', axis=1)
#we divide into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13, stratify=y)

def run_model_balanced(X_train, X_test, y_train, y_test, weight={1:20,0:1}):
    clf = LogisticRegression(C=0.1,penalty='l2',random_state=1,solver="newton-cg",class_weight=weight)
    clf.fit(X_train, y_train)
    return clf
 
model_balanced = run_model_balanced(X_train, X_test, y_train, y_test) 
pred_y = model_balanced.predict(X_test)
print(confusion_matrix(y_test, pred_y))
print(classification_report(y_test, pred_y,zero_division=False))

print(X_test.info())


#Flask Dump
filename = '../models/stroke_model.pkl'
pickle.dump(model_balanced, open(filename,'wb'))



