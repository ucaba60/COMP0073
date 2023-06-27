import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# Load the data
data = pd.read_csv('data_matrix.csv')

# Extract features and labels
X = data.drop(columns=['label'])
y = data['label']


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
base_models = [
    ('lr', make_pipeline(StandardScaler(), LogisticRegression())),
    ('svm', make_pipeline(StandardScaler(), SVC())),
    ('rf', RandomForestClassifier())
]


# Define stacking classifier
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())


# Define hyperparameters for grid search
param_grid = [
    {
        'lr__logisticregression__C': [0.1, 1.0, 10.0],
        'svm__svc__C': [0.1, 1.0, 10.0],
        'svm__svc__gamma': ['scale', 'auto'],
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [None, 5, 10],
    }
]


# Perform grid search
grid_search = GridSearchCV(stacking_clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print(f"Best parameters: {grid_search.best_params_}")

# Evaluate the model on the test set
test_score = grid_search.score(X_test, y_test)
print(f"Test accuracy: {test_score}")


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from matplotlib import pyplot as plt

# Make predictions with the best model from the grid search
y_pred = grid_search.predict(X_test)
y_prob = grid_search.predict_proba(X_test)[:,1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# Print metrics
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"AUC-ROC: {auc}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC-ROC = {auc:.2f}")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Print confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix: \n{confusion_mat}")