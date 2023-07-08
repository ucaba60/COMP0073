import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
import os
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create a directory for saving models and scaler
if not os.path.exists('model_data'):
    os.makedirs('model_data')

# Create a directory for output images if it does not exist
if not os.path.exists('output_images'):
    os.makedirs('output_images')


def load_data(data_path):
    # Load the dataset
    data = pd.read_csv(data_path)

    # Split the data into X (the input features) and y (the target)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    return X, y


def scale_and_split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the trained scaler
    joblib.dump(scaler, 'model_data/trained_scaler.pkl')

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_logreg(X_train_scaled, y_train):
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
    model_file = 'trained_model_logreg.pkl'
    if os.path.exists(model_file):
        overwrite = input("Model file already exists. Do you want to overwrite it? (yes/no) ")
        if overwrite.lower() != "yes":
            return
    joblib.dump(grid.best_estimator_, 'model_data/trained_model_logreg.pkl')

    return grid.best_estimator_


def load_model_and_scaler(model_file, scaler_file):
    # Load the model and scaler
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

    return model, scaler


def train_svm(X_train_scaled, y_train):
    svm = SVC(random_state=42)
    # Define the parameter grid for SVM
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

    grid = GridSearchCV(svm, param_grid, refit=True, verbose=3)

    grid.fit(X_train_scaled, y_train)

    model_file = 'model_data/trained_model_svm.pkl'
    if os.path.exists(model_file):
        overwrite = input("Model file already exists. Do you want to overwrite it? (yes/no) ")
        if overwrite.lower() != "yes":
            return

    joblib.dump(grid.best_estimator_, 'model_data/trained_model_svm.pkl')
    return grid.best_estimator_


def train_rf(X_train_scaled, y_train):
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
    model_file = 'model_data/trained_model_rf.pkl'
    if os.path.exists(model_file):
        overwrite = input("Model file already exists. Do you want to overwrite it? (yes/no) ")
        if overwrite.lower() != "yes":
            return

    joblib.dump(grid.best_estimator_, 'model_data/trained_model_rf.pkl')
    return grid.best_estimator_


def train_ensemble(X_train_scaled, y_train):
    # Load the trained models and scalers
    svm = joblib.load('model_data/trained_model_svm.pkl')
    rf = joblib.load('model_data/trained_model_rf.pkl')
    log_reg = joblib.load('model_data/trained_model_logreg.pkl')

    # Create a list of tuples, each tuple containing the string identifier and the model
    models = [('svm', svm), ('rf', rf), ('log_reg', log_reg)]

    # Create the ensemble model
    ensemble = VotingClassifier(estimators=models, voting='hard')

    # Fit the ensemble model on the scaled training data
    ensemble.fit(X_train_scaled, y_train)
    model_file = 'model_data/trained_model_ensemble.pkl'
    if os.path.exists(model_file):
        overwrite = input("Model file already exists. Do you want to overwrite it? (yes/no) ")
        if overwrite.lower() != "yes":
            return
    joblib.dump(ensemble, 'model_data/trained_model_ensemble.pkl')
    return ensemble


def make_prediction(model, X_test_scaled):
    # Make predictions
    y_pred = model.predict(X_test_scaled)

    return y_pred


def plot_roc_curve(y_test, y_pred, model_name):
    # Calculate ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_pred)
    print("ROC-AUC Score:", roc_auc)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # Plot ROC curve
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')  # Plotting the random guess line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    fig.savefig(f'output_images/{model_name}_ROC_Curve.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_learning_curve(estimator, X_train_scaled, y_train, model_name):
    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(estimator,
                                                            X_train_scaled,
                                                            y_train,
                                                            cv=5,
                                                            scoring='accuracy',
                                                            n_jobs=-1,
                                                            train_sizes=np.linspace(0.01, 1.0, 50))

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Define colors
    train_color = "#1f77b4"  # Blue
    test_color = "#ff7f0e"  # Orange
    std_color = "#DDDDDD"  # Light gray

    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color=train_color, label="Training score")
    plt.plot(train_sizes, test_mean, color=test_color, label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color=std_color)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color=std_color)

    # Create plot
    fig = plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.grid(False)  # Disable grid
    plt.tight_layout()
    plt.gca().set_facecolor('white')  # Set background color to white
    fig.savefig(f'output_images/{model_name}_Learning_Curve.png', dpi=300, bbox_inches='tight')

    plt.show()

    # Save the plot


def print_accuracy(grid, X_train_scaled, X_test_scaled, y_train, y_test):
    # Compute accuracy on the training set
    train_acc = grid.score(X_train_scaled, y_train)
    print(f'Training Accuracy: {train_acc * 100:.2f}%')

    # Compute accuracy on the validation set
    val_acc = grid.score(X_test_scaled, y_test)
    print(f'Validation Accuracy: {val_acc * 100:.2f}%')


def plot_confusion_matrix(y_test, y_pred, model_name):
    # Generate confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)

    # Create a DataFrame from the confusion matrix
    conf_mat_df = pd.DataFrame(conf_mat, columns=np.unique(y_test), index=np.unique(y_test))
    conf_mat_df.index.name = 'Actual'
    conf_mat_df.columns.name = 'Predicted'

    fig = plt.figure()
    plt.figure(figsize=(10, 7))

    # Create a seaborn heatmap
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(conf_mat_df, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')

    plt.title("Confusion Matrix")
    fig.savefig(f'output_images/{model_name}_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    plot_and_save_fig(fig, model_name, "Confusion_Matrix")


def print_classification_report(y_test, y_pred):
    print('Classification Report: \n', classification_report(y_test, y_pred))


def plot_feature_importance(model, X, model_name):
    # Get the coefficients from the logistic regression model
    coefficients = model.coef_[0]

    # Create a DataFrame to store the feature names and their corresponding coefficients
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})
    feature_importance_df['Absolute Coefficient'] = feature_importance_df['Coefficient'].abs()

    # Sort the features by their absolute coefficients in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Absolute Coefficient', ascending=False)

    # Plot the feature importances
    fig = plt.figure()
    plt.figure(figsize=(16, 10))
    sns.barplot(x='Absolute Coefficient', y='Feature', data=feature_importance_df, palette='coolwarm')
    plt.title("Feature Importances (Logistic Regression)")
    plt.xlabel("Absolute Coefficient")
    plt.ylabel("Feature")
    fig.savefig(f'output_images/{model_name}_Feature_Importance.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_correlation_matrix(X, model_name):
    corr = X.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    fig = plt.figure()
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    fig.savefig(f'output_images/{model_name}_Correlation_Matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    plot_and_save_fig(fig, "Correlation_Matrix")


def plot_permutation_importance(model, X_test, y_test, model_name):
    from sklearn.inspection import permutation_importance

    # Convert input features to a NumPy array
    X_test_array = X_test.to_numpy()

    # Calculate permutation importances
    result = permutation_importance(model, X_test_array, y_test, n_repeats=10, random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    # Plot the permutation importances using a boxplot
    fig = plt.figure()
    fig, ax = plt.subplots(figsize=(10, 6))  # Increase the figsize as needed
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx])
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    fig.savefig(f'output_images/{model_name}_Permutation_Importance.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_and_save_fig(fig, model_name, chart_name):
    filename = f"output_images/{model_name}_{chart_name}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")




