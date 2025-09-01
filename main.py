from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np

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

# Plot points and hyperplane as orthogonal projection
# Based on graphs previously plotted, it seems that feature 0 (sepal length) does not differ much between different targets
# so it does not seem to be helpful to plot it
N = clf.coef_
d = clf.intercept_
fig = plt.figure()
gs = fig.add_gridspec(2, 2)
(ax1, ax2), (ax3, ax4) = gs.subplots(sharex="col", sharey="row")
hyperplane_color_list = ["b", "g", "r"]
target_color_list = ["c", "m", "y"]

# Custom function for making plots
def make_plot(ax, h_axis, v_axis):
    """Custom function for making plots with a given horizontal and vertical axis"""
    x_pts = X[:,h_axis]
    for hyperplane in range(d.size):
        ax.plot(x_pts, (d[hyperplane]-N[hyperplane][h_axis]*x_pts)/N[hyperplane][v_axis], hyperplane_color_list[hyperplane])
    for target in set(y):
        ax.plot(x_pts[y==target], X[:,v_axis][y==target], target_color_list[target]+"o")

# YZ-plane
ax1.set(ylabel=iris.feature_names[2])
make_plot(ax1, h_axis=1, v_axis=2)

# XY-plane
ax3.set(xlabel=iris.feature_names[1], ylabel=iris.feature_names[3])
make_plot(ax3, h_axis=1, v_axis=3)

# XZ-plane
ax4.set(xlabel=iris.feature_names[2])
make_plot(ax4, h_axis=2, v_axis=3)

plt.show()

# Plot points and hyperplane in 3D
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
xv = X[:,1]
yv = X[:,2]

# Plot points in 3D
for target in set(y):
    ax.plot(xv[y==target], yv[y==target], X[:,3][y==target], target_color_list[target]+"o")

# Plot hyperplanes in 3D
xv = np.linspace(np.min(xv), np.max(xv), num=3)
yv = np.linspace(np.min(yv), np.max(yv), num=3)
xv, yv = np.meshgrid(xv, yv)
for hyperplane in range(d.size):
    ax.plot_surface(xv, yv, (d[hyperplane]-N[hyperplane][1]*xv-N[hyperplane][2]*yv)/N[hyperplane][3], color=hyperplane_color_list[hyperplane])

# Label axes
ax.set_xlabel(iris.feature_names[1])
ax.set_ylabel(iris.feature_names[2])
ax.set_zlabel(iris.feature_names[3])

plt.show()
