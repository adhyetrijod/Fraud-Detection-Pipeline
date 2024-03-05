import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import logging
logging.basicConfig(level=logging.INFO)
try:
  basedata = np.genfromtxt('../data/data-eng.csv', skip_header=1, delimiter=',', dtype='f8')
except FileNotFoundError:
  logging.error("Data file not found!")
  exit(1)
except Exception as e:
  logging.error(f"Error loading data: {e}")
  exit(1)

basedata = basedata[:, 1:]
X = basedata[:, 1:]
y = basedata[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

try:
  clf.fit(X_train, y_train)
except Exception as e:
  logging.error(f"Error training model: {e}")
  exit(1)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model accuracy on test set: {accuracy:.4f}")

import pickle
with open('../private/gb-clf.pkl', 'wb') as sv_file:
    pickle.dump(clf, sv_file)
