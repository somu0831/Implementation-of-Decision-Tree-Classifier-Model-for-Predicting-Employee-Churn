# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas as pd
2. from sklearn.preprocessing import LabelEncoder 
3. from sklearn.model_selection import train_test_split and split the training and testing data from original data
4. from sklearn import metrics to presict the y_prdict value
5. by using metrics.accuracy_score we can get the accuracy

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Panduru Somu
RegisterNumber:  212223240111
*/
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
le = LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
accuracy=metrics.accuracy_score(y_test,y_pred)
print(float(accuracy))
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
![Screenshot 2024-04-02 091412](https://github.com/somu0831/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/162110820/c25890f6-31ec-44b3-804f-ff0127dae175)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
