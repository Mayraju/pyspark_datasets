import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.svm import SVC


dataset = "/opt/dkube/dataset"

df = pd.read_csv(dataset+'/train_data_1.csv',sep=',')



X = df.drop(['label'],axis=1).values  
y = df['label'].values
X_train60,X_test40,y_train60,y_test40 = train_test_split(X,y,test_size=0.40,random_state=32)
#X_train70,X_test30,y_train70,y_test30 = train_test_split(X, y, test_size=0.30, random_state=42)
#X_train80,X_test20,y_train80,y_test20 = train_test_split(X,y,test_size=20,random_state=52)
#X_train90,X_test10,y_train90,y_test10 = train_test_split(X,y,test_size=10,random_state=42)
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
clf4 = tree.DecisionTreeClassifier()
clf5 = SVC(kernel='rbf', probability=True)

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gba', clf3),('dt',clf4),('svm',clf5)],voting='soft',weights=[0.1,0.2,0.3,0.1,0.3])

print("voting classifier")
vot = eclf.fit(X_train60,y_train60)
p_vot = vot.predict(X_test40)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt   

print("vot")
labels = ['attack', 'normal']
cm = confusion_matrix(y_test40, p_vot, labels)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("accuracy")
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test40, p_vot))

print("precision")
from sklearn.metrics import precision_score
print(precision_score(y_test40, p_vot, average='macro'))
print(precision_score(y_test40, p_vot, average='micro'))
print(precision_score(y_test40, p_vot, average='weighted'))

print("recall")

from sklearn.metrics import recall_score

print(recall_score(y_test40, p_vot, average='macro'))
print(recall_score(y_test40, p_vot, average='micro'))
print(recall_score(y_test40, p_vot, average='weighted'))

print("f1-score")
from sklearn.metrics import f1_score
print(f1_score(y_test40, p_vot, average='macro'))
print(f1_score(y_test40, p_vot, average='micro'))
print(f1_score(y_test40, p_vot, average='weighted'))
