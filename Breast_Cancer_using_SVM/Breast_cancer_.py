import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()
type(cancer_dataset)

cancer_dataset.keys()

cancer_dataset['data']

cancer_dataset['target_names']
print(cancer_dataset['DESCR'])


cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],
             columns = np.append(cancer_dataset['feature_names'], ['target']))

cancer_df.info()
#data preprocessing

X = cancer_df.drop(['target'], axis = 1)
X.head(6)

y = cancer_df['target']
y.head(6)

# split dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 5)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# Support vector classifier
from sklearn.svm import SVC
svc_classifier = SVC()
svc_classifier.fit(X_train, y_train)
y_pred_scv = svc_classifier.predict(X_test)
accuracy_score(y_test, y_pred_scv)

cm = confusion_matrix(y_test, y_pred_scv)
plt.title('Heatmap of Confusion Matrix', fontsize = 15)
sns.heatmap(cm, annot = True)
plt.show()

print(classification_report(y_test, y_pred_scv))

import pickle


pickle.dump(X_train, open('breast_cancer_detector.pickle', 'wb'))

breast_cancer_detector_model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))

y_pred = breast_cancer_detector_model.predict(X_test)

print('Confusion matrix of XGBoost model: \n', confusion_matrix(y_test, y_pred), '\n')


print('Accuracy of XGBoost model = ', accuracy_score(y_test, y_pred))
