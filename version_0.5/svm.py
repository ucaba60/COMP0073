from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Load the data
data = pd.read_csv('Model_Ready_Data/combined_data_reg_ready.csv')

# Split the data into features and labels
X = data.drop('label', axis=1)
y = data['label']

# Initialize the scaler
scaler = MinMaxScaler()  # or StandardScaler()

# Fit the scaler using the training data and scale it
X_scaled = scaler.fit_transform(X)

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create a SVM Classifier
clf = svm.SVC(kernel='linear')  # You can change the kernel based on your needs.

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for the test dataset
y_pred = clf.predict(X_test)

# Print the classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))