# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 14:42:48 2022

@author: Babs
"""

import pandas as pd
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from time import time


df = pd.read_csv('Trainset.csv') # Load the data
from sklearn.model_selection import train_test_split

df.info()

# The target variable is 'Attack'
#factorising the Attack column for RFC model
df.Attack.value_counts()
df['Code'] = pd.factorize(df.Attack)[0]
df.Code.value_counts()
df = df.drop('Attack', axis=1)
df.head()
df = df.rename(columns={'Code':'Attack'})
df.head()
#factorising the Protocol column
df.protocol_type.value_counts()
df['Code'] = pd.factorize(df.protocol_type)[0]
df.Code.value_counts()
df = df.drop('protocol_type', axis=1)
df.head()
df = df.rename(columns={'Code':'protocol_type'})
df.head()
#Factorising the service column
df.service.value_counts()
df['Code'] = pd.factorize(df.service)[0]
df.Code.value_counts()
df = df.drop('service', axis=1)
df.head()
df = df.rename(columns={'Code':'service'})
df.head()
#Factorising the flag column
df.flag.value_counts()
df['Code'] = pd.factorize(df.flag)[0]
df.Code.value_counts()
df = df.drop('flag', axis=1)
df.head()
df = df.rename(columns={'Code':'flag'})
df.head()


Y = df['Attack']
#Normalisation of Dataset using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_df = scaler.fit_transform(df[['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','Attack']])
X_df = pd.DataFrame(X_df, columns=df.columns)
X = df[['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']]
# Split the data into train and test data:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4)

#Build the model with the random forest classifier algorithm:
model = RandomForestClassifier()
model.fit(X_train, Y_train)
model.score(X_test, Y_test)

#Confusion Matix
y_predict = model.predict(X_test)
cm2 = confusion_matrix(Y_test, y_predict)
cm2

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,5))
sns.heatmap(cm2, annot=True)
plt.xlabel('Predicted')
plt.ylabel('True')

import shap
shap_values = shap.TreeExplainer(model).shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")

shap.summary_plot(shap_values, X_train)

shap.dependence_plot('src_bytes', shap_values[0], X_train)

shap.dependence_plot('dst_bytes', shap_values[0], X_train)

#Dropping the not important columns after SHAP Analysis
df2 = df
y = df2['Attack']
x =  df2[['protocol_type','service','flag','src_bytes','dst_bytes','hot','logged_in','count','srv_count','rerror_rate','same_srv_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']]
# Split the data into train and test data:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

#Build the model with the random forest classifier algorithm:
df2_model = RandomForestClassifier()
df2_model.fit(x_train, y_train)
df2_model.score(x_test, y_test)

#Confusion Matix
y_pred = df2_model.predict(x_test)
CM = confusion_matrix(y_test, y_pred)
CM

#Explains the classification of the 500th entry of the test set 
explainer = shap.TreeExplainer(model)
choosen_instance = X_test.iloc[[500]]
shap_values = explainer.shap_values(choosen_instance)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance)
shap.force_plot(explainer.expected_value[1], shap_values[0], choosen_instance)

#Explains the classification of the 1000th entry of the test set
explainer = shap.TreeExplainer(model)
choosen_instance = X_test.iloc[[1000]]
shap_values = explainer.shap_values(choosen_instance)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance)
shap.force_plot(explainer.expected_value[1], shap_values[0], choosen_instance)

model_preds = model.predict(X_test) 
model_preds
len(X_test)
X_test.iloc[[421]]




def shap_plot(j):
    explainer = shap.TreeExplainer(model)
    shap_values_Model = explainer.shap_values(S)
    p = shap.force_plot(explainerModel.expected_value, shap_values_Model[j], S.iloc[[j]], matplotlib = True, show = False)
    plt.savefig('tmp.svg')
    plt.close()
    return(p)
