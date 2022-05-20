# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:30:23 2022

@author: VicboxMS
"""

##Instalaciones necesarias 
!pip install dpcpp-cpp-rt
!pip install scikit-learn-intelex
!pip install dpctl

#> ! Importante !
#> Llamar sklearnex.patch_sklearn(), antes de invocar al clasificador de sklearn sklearn.svm.SVC()

import dpctl
from sklearnex import patch_sklearn, config_context
patch_sklearn(['SVC'])
##
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time


data=load_breast_cancer()
X = data['data']#scaler.fit_transform(data['data'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.5, random_state=0)#



r=[]
for i in range(0,4):
    a = time.time()
    clf = SVC(kernel='linear')
    clf.fit(X_train,y_train)
    print(clf.score(X_test,y_test))
    b = time.time()
    r.append(b-a)
print(r)
print('Tiempo en segundos',np.mean(r),)





ruta = 'airline_standardscale.csv'
df=pd.read_csv(ruta)
labelencoder= LabelEncoder()
df.iloc[:,-1] = labelencoder.fit_transform(df.iloc[:,-1]) 
data=list()
data={'data':np.array(df.iloc[:,:-1]),'target':np.array(df.iloc[:,-1])}
X = data['data']#scaler.fit_transform(data['data'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.5, random_state=0)#
o , a = X_train.shape
print('# Observaciones: ',o,'Numero de Atributos:',a)




r=[]
for i in range(0,2):
    a = time.time()
    svc = SVC(kernel='rbf')
    parameters = {'gamma':[0.005,0.05,0.5]}
    clasificador_svm = GridSearchCV(svc, parameters,n_jobs=-1,cv=3)
    clasificador_svm.fit(X_train,y_train)
    print(clasificador_svm.score(X_test,y_test))
    b = time.time()
    r.append(b-a)
print(r,'\n')
print('Tiempo promedio  de ejecución : ', np.mean(r))
