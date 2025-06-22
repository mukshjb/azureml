# train_model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data and train model
iris = load_iris()
X, y = iris.data, iris.target
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

# Save model
with open('model/iris_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
