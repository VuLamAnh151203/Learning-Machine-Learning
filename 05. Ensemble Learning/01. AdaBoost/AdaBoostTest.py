import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from AdaBoost import DecisionStump, AdaBoost
dataset = sklearn.datasets.load_breast_cancer()
X = dataset["data"]
y = dataset["target"]

# change the class = 0 -> -1
y[y == 0] = -1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

adaboost = AdaBoost(n_clf = 50)
adaboost.fit(X_train, y_train)

print(accuracy_score(adaboost.predict(X_test), y_test))
