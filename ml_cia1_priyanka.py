import pandas as pd
import numpy as np
import matplotlib as mlt

df = pd.read_csv(r"C:\Users\PRIYANKA\Downloads\Swarm_Behaviour.csv")
print(df.head(15))
print(df.info())
print(df.describe())

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,
                                                random_state=0)
from sklearn import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train)

#Logistic Regression

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_test,y_pred)
print("Accuracy: ", accuracy_score(y_test,y_pred))

#kNN 

from sklearn.neighbors import KNeighborsClassifier
best_Kvalue = 0
best_score = 0
for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    if knn.score(x_test, y_test) > best_score:
        best_score = knn.score(x_train, y_train)
        best_Kvalue = i

print("Best KNN Value: ",best_Kvalue)
print("Accuracy: ",best_score)

#Naive Bayes 

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
print("Accuracy: ",nb.score(x_test, y_test))    
        
