from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Scale data
X = StandardScaler().fit(X).transform(X)

# Split data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Train model
clf = SVC(kernel="linear")
clf.fit(X_train, y_train)

# Classify test data using model
y_pred = clf.predict(X_test)

# Compute the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=iris.target_names)
disp.plot()
plt.show()

# Get the model classification metrics (will only show after the confusion matrix display window is closed)
print(classification_report(y_test, y_pred, target_names=iris.target_names))
