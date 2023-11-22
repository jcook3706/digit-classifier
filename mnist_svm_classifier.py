from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the MNIST dataset
digits = datasets.load_digits()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Create an SVM classifier
svm_classifier = SVC(kernel='poly')  # Linear kernel, can also use 'rbf' or 'poly'

# Train the SVM classifier
svm_classifier.fit(X_train, y_train)

# Predict on the test set
predictions = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
