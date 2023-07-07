from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import joblib


def load_and_process_data(filepath):
    data = pd.read_csv(filepath)
    X = data.drop('label', axis=1)
    y = data['label']
    return X, y


def split_and_scale_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler


def train_model(X_train, y_train):
    clf = LogisticRegression()
    param_grid = [
        {
            'solver': ['newton-cg', 'lbfgs', 'sag'],
            'penalty': ['l2', 'none'],
            'C': np.logspace(-4, 4, 20),
            'max_iter': [10000, 20000, 50000]  # increased max_iter values even more
        },
        {
            'solver': ['liblinear', 'saga'],
            'penalty': ['l1', 'l2'],
            'C': np.logspace(-4, 4, 20),
            'max_iter': [10000, 20000, 50000]  # increased max_iter values even more
        },
        {
            'solver': ['saga'],
            'penalty': ['elasticnet'],
            'C': np.logspace(-4, 4, 20),
            'l1_ratio': np.linspace(0, 1, 10),  # Only applicable for 'elasticnet'
            'max_iter': [10000, 20000, 50000]  # increased max_iter values even more
        },
    ]
    n_iter_search = 50
    random_search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=n_iter_search, cv=5, verbose=0,
                                       n_jobs=-1)
    random_search.fit(X_train, y_train)
    model_best = random_search.best_estimator_
    return model_best


# Fit on training set only.

def make_predictions(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred


def save_model(model, filename):
    joblib.dump(model, filename)


def load_model(filename):
    model = joblib.load(filename)
    return model


def evaluate_model(y_test, y_pred):
    print('Classification Report:')
    print(classification_report(y_test, y_pred))


X, y = load_and_process_data("data_matrix_gpt-3.5-turbo.csv")
X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
try:
    model_best = load_model('model_best.joblib')
except:
    model_best = train_model(X_train, y_train)
    save_model(model_best, 'model_best.joblib')
y_pred = make_predictions(model_best, X_test)
evaluate_model(y_test, y_pred)


# Function for confusion matrix
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix', size=15)
    plt.show()


def plot_coefficients(model, X):
    cols = X.columns

    # For binary classification
    if len(model.coef_) == 1:
        plt.figure(figsize=(10, 6))
        coef = model.coef_[0]
        sns.barplot(x=cols, y=coef)
        plt.xticks(rotation=90)
        plt.title('Feature Coefficients')

    # For multi-class classification
    else:
        plt.figure(figsize=(10, 6))
        coef = model.coef_
        sns.heatmap(coef, xticklabels=cols, cmap="RdBu_r")
        plt.xticks(rotation=90)
        plt.title('Feature Coefficients')

    plt.show()


# Function for ROC-AUC curve
def plot_roc_auc_curve(y_test, y_scores):
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()

#
# plot_confusion_matrix(y_test, y_pred)
# plot_coefficients(model_best, X)
# plot_roc_auc_curve(y_test, y_pred)
