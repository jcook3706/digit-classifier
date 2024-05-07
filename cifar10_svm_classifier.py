from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Load CIFAR-10 dataset
cifar = datasets.fetch_openml('CIFAR_10_small')

# Extract images and labels
X = cifar.data
y = cifar.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier (you can choose the kernel and other parameters)
clf = svm.SVC(kernel='poly')

# Train the classifier
start_time = time.time()
clf.fit(X_train, y_train)
print(f'Time to train SVM classifier on training partition of CIFAR10: {(time.time()-start_time):.4f} seconds')

# Predict on the test set
start_time = time.time()
y_pred = clf.predict(X_test)
print(f'Time to run inference on test partition of CIFAR10: {(time.time()-start_time):.4f} seconds')

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Partition Accuracy: {accuracy:.5f}")
