import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

df = pd.read_csv(r"C:\Users\HP\Desktop\madical premium\credit card project\clean_dataset.csv")


print(df)
print(df.duplicated().value_counts())

print(df.nunique())
df1 = df.copy()
df1.dtypes


# display the special character columns wise
import re


print(df.head())
print(df.tail())

print(df.info())
print(df.nunique())
print(df.describe())


df['Approved'].value_counts()
print(df['Approved'].value_counts())
df['Approved'].value_counts().plot(kind='pie',autopct='%1.1f%%',figsize=(5,5))
plt.show()

print(df.isnull().sum())# there are no null values if there are null values you can fill them by using bfill,ffill,mean,median,mode)


print(df.dtypes)
print(df.head(2))



from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in df.columns:
    if df[col].dtypes == object:
        df[col]= le.fit_transform(df[col])
x=df.drop('Approved',axis=1)
y=df['Approved']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test  = sc.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=42)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
y_pred

a=pd.DataFrame({'actual_value':y_test,'predicted_value':y_pred})
a

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm = confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
print(cm)

log_acc=accuracy_score(y_test,y_pred)*100
log_acc

result = classifier.predict(np.array([[1,22,5.6,1,1,5,3,4.5,0,2,0,1,203,450,1]]))
result

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,random_state=42,max_features=15)
rf.fit(x_train,y_train)

y_pred_rf = rf.predict(x_test)
y_pred_rf

pd.DataFrame({'actual_value':y_test,'predicted_value':y_pred_rf})
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm = confusion_matrix(y_test,y_pred_rf)
print(classification_report(y_test,y_pred_rf))
print(cm)

rf_acc = accuracy_score(y_test,y_pred_rf)*100# it defines how much correctly the model is predicting the actual value
rf_acc

result = rf.predict(np.array([[1,22,5.6,1,1,5,3,4.5,0,2,0,1,203,450,1]]))
result


from sklearn import svm
svm = svm.SVC(kernel='linear',C = 0.01)
svm.fit(x_train,y_train)
# Note: there four types of kernels : linear,rbf,poly and sigmoid the most reliable and to get best accuracy we use linear kernel

y_pred_svm = svm.predict(x_test)
y_pred_svm

pd.DataFrame({'actual_value':y_test,'predicted_value':y_pred_svm})

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm = confusion_matrix(y_test,y_pred_svm)
print(classification_report(y_test,y_pred_svm))
print(cm)

svm_acc = accuracy_score(y_test,y_pred_svm)*100
svm_acc

result = svm.predict(np.array([[1,29,9,0,0,6,3,4.5,0,3,0,1,203,590,1]]))
result

plt.rcParams["figure.figsize"] = [8, 4]
plt.rcParams["figure.autolayout"] = True

x=['logistic accuracy','randomforest_accuracy','support vector machine_accuracy']
y=[log_acc,rf_acc,svm_acc]

width = 0.75
fig, ax = plt.subplots()

pps = ax.bar(x, y, width, align='center')

for p in pps:
   height = p.get_height()
   ax.text(x=p.get_x() + p.get_width() / 2, y=height+1,
      s="{}%".format(height),
      ha='center')
plt.title('Accuracy of models')
plt.show()

import joblib

# Save RandomForest and StandardScaler
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(sc, 'scaler.pkl')