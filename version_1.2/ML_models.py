import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
import os

# Create a directory for saving models and scaler
if not os.path.exists('model_data'):
    os.makedirs('model_data')

# Create a directory for output images if it does not exist
if not os.path.exists('output_images'):
    os.makedirs('output_images')


def load_data(data_path, training_data_name):
    # Load the dataset
    data = pd.read_csv(data_path)

    # Split the data into X (the input features) and y (the target)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Save feature names
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, f'model_data/{training_data_name}/feature_names.pkl')

    return X, y


def scale_and_split_data(X, y, training_data_name, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the trained scaler
    joblib.dump(scaler, f'model_data/{training_data_name}/trained_scaler.pkl')

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_logreg(X_train_scaled, y_train, training_data_name):
    logreg = LogisticRegression(random_state=42)
    param_grid = {
        'solver': ['newton-cg', 'lbfgs', 'sag', 'saga', 'liblinear'],
        'penalty': ['l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]  # Inverse of regularization strength
    }

    param_grid_l1 = {
        'solver': ['liblinear', 'saga'],
        'penalty': ['l1'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]  # Inverse of regularization strength
    }

    param_grid_elasticnet = {
        'solver': ['saga'],
        'penalty': ['elasticnet'],
        'l1_ratio': [i / 10.0 for i in range(11)],  # Increments of 0.1 from 0 to 1
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]  # Inverse of regularization strength
    }

    param_grid = [param_grid, param_grid_l1, param_grid_elasticnet]
    grid = GridSearchCV(logreg, param_grid, cv=5, verbose=True, n_jobs=-1)

    grid.fit(X_train_scaled, y_train)
    # Save the trained model
    model_file = f'model_data/{training_data_name}/trained_model_logreg.pkl'
    if os.path.exists(model_file):
        overwrite = input("Model file already exists. Do you want to overwrite it? (yes/no) ")
        if overwrite.lower() != "yes":
            return
    joblib.dump(grid.best_estimator_, f'model_data/{training_data_name}/trained_model_logreg.pkl')

    return grid.best_estimator_


def load_model_and_scaler(model_file, scaler_file):
    # Load the model and scaler
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

    return model, scaler


def train_svm(X_train_scaled, y_train, training_data_name):
    svm = SVC(random_state=42,probability=True)
    # Define the parameter grid for SVM
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

    grid = GridSearchCV(svm, param_grid, refit=True, verbose=3)

    grid.fit(X_train_scaled, y_train)

    model_file = f'model_data/{training_data_name}/trained_model_svm.pkl'
    if os.path.exists(model_file):
        overwrite = input("Model file already exists. Do you want to overwrite it? (yes/no) ")
        if overwrite.lower() != "yes":
            return

    joblib.dump(grid.best_estimator_, f'model_data/{training_data_name}/trained_model_svm.pkl')
    return grid.best_estimator_


def train_rf(X_train_scaled, y_train, training_data_name):
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [10, 50, 100, 200],  # The number of trees in the forest
        'max_features': ['sqrt', 'log2'],  # The number of features to consider when looking for the best split
        'max_depth': [None, 10, 20, 30, 40, 50],  # The maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # The minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],  # The minimum number of samples required to be at a leaf node
        'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
    }
    grid = GridSearchCV(rf, param_grid, cv=5, verbose=True, n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    model_file = f'model_data/{training_data_name}/trained_model_rf.pkl'
    if os.path.exists(model_file):
        overwrite = input("Model file already exists. Do you want to overwrite it? (yes/no) ")
        if overwrite.lower() != "yes":
            return

    joblib.dump(grid.best_estimator_, f'model_data/{training_data_name}/trained_model_rf.pkl')
    return grid.best_estimator_


def train_ensemble(X_train_scaled, y_train, training_data_name):
    # Load the trained models and scalers
    svm = joblib.load(f'model_data/{training_data_name}/trained_model_svm.pkl')
    rf = joblib.load(f'model_data/{training_data_name}/trained_model_rf.pkl')
    log_reg = joblib.load(f'model_data/{training_data_name}/trained_model_logreg.pkl')

    # Create a list of tuples, each tuple containing the string identifier and the model
    models = [('svm', svm), ('rf', rf), ('log_reg', log_reg)]

    # Create the ensemble model
    ensemble = VotingClassifier(estimators=models, voting='soft')

    # Fit the ensemble model on the scaled training data
    ensemble.fit(X_train_scaled, y_train)
    model_file = f'model_data/{training_data_name}/trained_model_ensemble.pkl'
    if os.path.exists(model_file):
        overwrite = input("Model file already exists. Do you want to overwrite it? (yes/no) ")
        if overwrite.lower() != "yes":
            return
    joblib.dump(ensemble, f'model_data/{training_data_name}/trained_model_ensemble.pkl')
    return ensemble


def make_prediction(model, X_test_scaled):
    # Make predictions
    y_pred = model.predict(X_test_scaled)

    return y_pred



