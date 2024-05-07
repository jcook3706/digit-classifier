from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

# Load the MNIST dataset
digits = datasets.load_digits()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Create an SVM classifier
svm_classifier = SVC(kernel='rbf')  # Linear kernel, can also use 'rbf' or 'poly'

# Train the SVM classifier
start_time = time.time()
svm_classifier.fit(X_train, y_train)
print(f'Time to train SVM classifier on training partition of MNIST: {(time.time()-start_time):.4f} seconds')

# Predict on the test set
start_time = time.time()
predictions = svm_classifier.predict(X_test)
print(f'Time to run inference on test partition of MNIST: {(time.time()-start_time):.4f} seconds')

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Test partition accuracy: {accuracy:5f}")
