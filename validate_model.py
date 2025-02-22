from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from joblib import load
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = load('model.pkl')

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
threshold = 1.1
assert accuracy > threshold, "Accuracy is below given threshold"
print(f'Accuracy: {accuracy}')