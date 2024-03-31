import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


preprocessed_data = pickle.load(open('./preprocess_numbers.pickle', 'rb'))

data = np.asarray(preprocessed_data['data'])
labels = np.asarray(preprocessed_data['labels'])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_pred, y_test)

print("Accuracy of the model is : {0} %".format(accuracy*100))

cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm, annot=True)
plt.title("Confusion Matrix for Sign Language - Alphabets\nAccuracy of the model is : {0} %".format(accuracy*100))
plt.show()

f = open('model_numbers.p', 'wb')
pickle.dump({'model': model}, f)
f.close()