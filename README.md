# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the necessary packages using import statement.
2. Read the given csv file and print the number of contents to be displayed.
3. Split the dataset using train_test_split.
4. Calculate Y_Pred and accuracy.
5. Display the result. 
 

## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: HEMAROSHINI M
RegisterNumber: 212219220015

import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["EmailText"].values
y=data["Label"].values
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
Data.head():

![image](https://user-images.githubusercontent.com/107909531/174952316-09a0b7f1-aaa8-4402-a8ce-1d931fad8951.png)


Data.info():

![image](https://user-images.githubusercontent.com/107909531/174952359-0a91a732-978e-493c-bd2f-19fd411ad7f5.png)


Data.isnull().sum():

![image](https://user-images.githubusercontent.com/107909531/174952398-abe04470-b7f4-49cd-9507-002a1d14cb94.png)


Y_Pred:

![image](https://user-images.githubusercontent.com/107909531/174952440-846072b9-9016-4d9f-97f4-854933955b14.png)


Accuracy:

![image](https://user-images.githubusercontent.com/107909531/174952473-9c7e1c47-c9a4-4ca7-a0fe-dbb46d53b3da.png)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
