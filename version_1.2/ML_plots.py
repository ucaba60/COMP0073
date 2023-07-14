import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve


def plot_text_perplexity_distribution(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Filter rows for label 0 and 1
    df_0 = df[df['label'] == 0]
    df_1 = df[df['label'] == 1]

    # Get the average perplexity for label 0
    avg_perplexity_0 = df_0['text_perplexity'].mean()

    avg_perplexity_1 = df_1['text_perplexity'].mean()

    # Set a theme for prettier plots
    sns.set(style="white")  # set style to white to remove grids

    # Create the KDE plot for label 0
    sns.kdeplot(data=df_0, x='text_perplexity', fill=True, color='lightblue', label='Human Text')

    # Create the KDE plot for label 1
    sns.kdeplot(data=df_1, x='text_perplexity', fill=True, color='salmon', label='GPT-3.5-turbo Text')

    # Add a vertical line at the average perplexity for label 0
    plt.axvline(x=avg_perplexity_0, color='gray', linestyle='--', alpha=0.8,
                linewidth=0.5)

    # Add a vertical line at the average perplexity for label 1
    plt.axvline(x=avg_perplexity_1, color='gray', linestyle='--', alpha=0.8,
                linewidth=0.5)

    # Add title and labels
    plt.title('[GPT2-large] Distribution of Text Perplexity for Human and LM-generated Text')
    plt.xlabel('Text Perplexity')
    plt.ylabel('Density')

    # Set x-axis limit
    plt.xlim(0.9, 1.4)

    # Show the legend and the plot
    plt.legend()
    plt.show()


# plot_text_perplexity_distribution("data_matrix_gpt2-large.csv")


def plot_column_frequencies(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Define columns to plot
    columns_to_plot = ['ADJ', 'ADV', 'CONJ', 'NOUN', 'NUM', 'VERB', 'COMMA', 'FULLSTOP', 'SPECIAL-', 'FUNCTION-A',
                       'FUNCTION-IN', 'FUNCTION-OF', 'FUNCTION-THE']

    # Create DataFrame with sums of each column, separately for each label
    df_sum = df.groupby('label')[columns_to_plot].sum().transpose()

    # Convert sums to proportions (in percentage)
    df_sum_percentage = (df_sum / df_sum.sum()) * 100

    # Calculate the combined proportions for sorting
    df_sum_percentage['combined'] = df_sum_percentage.sum(axis=1)

    # Sort by the combined proportions
    df_sum_percentage = df_sum_percentage.sort_values(by='combined', ascending=False)

    # Drop the combined proportions column
    df_sum_percentage = df_sum_percentage.drop(columns='combined')

    # Create a grouped bar chart
    ax = df_sum_percentage.rename(columns={0: 'Human', 1: 'GPT-J'}).plot(kind='bar', stacked=False)

    # Add percentage annotations on top of each bar
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                    xytext=(0, 10), textcoords='offset points', fontsize=8)

    # Add title and labels
    plt.title('[GPT-J] Parts-of-Speech Frequency Comparison between Human and LM-generated Texts')
    plt.xlabel('Columns')
    plt.ylabel('Proportion (%)')

    # Show the legend and the plot
    plt.legend(title='Label', loc='upper right')
    plt.show()


# plot_column_frequencies("data_matrix_gpt-j1x.csv")

def plot_prompt_text_cosine_similarity_distribution(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Filter rows for label 0 and 1
    df_0 = df[df['label'] == 0]
    df_1 = df[df['label'] == 1]

    # Set a theme for prettier plots
    sns.set(style="white")  # set style to white to remove grids

    # Create the KDE plot for label 0
    sns.kdeplot(data=df_0, x='prompt_text_cosine_similarity', fill=True, color='lightblue', label='Human')

    # Create the KDE plot for label 1
    sns.kdeplot(data=df_1, x='prompt_text_cosine_similarity', fill=True, color='salmon', label='GPT2-large')

    # Add title and labels
    plt.title(
        '[GPT2-large] Distribution of Cosine Similarity between Prompt & Response for Human and LM-generated Texts')
    plt.xlabel('Prompt Text Cosine Similarity')
    plt.ylabel('Density')

    # Add vertical lines for mean values
    plt.axvline(df_0['prompt_text_cosine_similarity'].mean(), color='blue', linestyle='dashed', linewidth=0.5)
    plt.axvline(df_1['prompt_text_cosine_similarity'].mean(), color='red', linestyle='dashed', linewidth=0.5)

    # Show the legend and the plot
    plt.legend()
    plt.show()


# plot_prompt_text_cosine_similarity_distribution("data_matrix_gpt2-large.csv")


def plot_avg_readability_metrics(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Compute the average flesch_kincaid_grade_level and flesch_reading_ease for each label
    avg_metrics = df.groupby('label')[['flesch_kincaid_grade_level', 'flesch_reading_ease']].mean()

    # Create a bar plot
    avg_metrics.plot(kind='bar', color=['lightblue', 'salmon'], alpha=0.7)

    # Add title and labels
    plt.title('Average Readability Metrics for Human and GPT-3.5-turbo')
    plt.xlabel('Label')
    plt.ylabel('Average Value')
    plt.xticks(ticks=[0, 1], labels=['Human', 'GPT-3.5-turbo'], rotation=0)  # replace numeric labels with text labels

    # Show the plot
    plt.show()


# Call the function
def plot_avg_word_length_distribution(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Filter rows for label 0 and 1
    df_0 = df[df['label'] == 0]
    df_1 = df[df['label'] == 1]

    # Calculate average word lengths
    avg_word_length_0 = df_0['avg_word_length'].mean()
    avg_word_length_1 = df_1['avg_word_length'].mean()

    # Set a theme for prettier plots
    sns.set(style="white")  # set style to white to remove grids

    # Create the KDE plot for label 0
    sns.kdeplot(data=df_0, x='avg_word_length', fill=True, color='lightblue', label='Human')

    # Create the KDE plot for label 1
    sns.kdeplot(data=df_1, x='avg_word_length', fill=True, color='salmon', label='GPT-3.5-turbo')

    # Add average lines
    plt.axvline(avg_word_length_0, color='blue', linestyle='--', linewidth=0.5)
    plt.axvline(avg_word_length_1, color='red', linestyle='--', linewidth=0.5)

    # Add title and labels
    plt.title('[GPT-3.5-turbo] Distribution of Average Word Length for Human and LM-generated Texts')
    plt.xlabel('Average Word Length')
    plt.ylabel('Density')

    # Show the legend and the plot
    plt.legend()
    plt.show()


# plot_avg_word_length_distribution("data_matrix_gpt-3.5-turbo.csv")


def plot_avg_sentence_length_distribution(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Filter rows for label 0 and 1
    df_0 = df[df['label'] == 0]
    df_1 = df[df['label'] == 1]

    # Calculate average sentence lengths
    avg_sentence_length_0 = df_0['avg_sentence_length'].mean()
    avg_sentence_length_1 = df_1['avg_sentence_length'].mean()

    # Set a theme for prettier plots
    sns.set(style="white")  # set style to white to remove grids

    # Create the KDE plot for label 0
    sns.kdeplot(data=df_0, x='avg_sentence_length', fill=True, color='lightblue', label='Human')

    # Create the KDE plot for label 1
    sns.kdeplot(data=df_1, x='avg_sentence_length', fill=True, color='salmon', label='GPT-3.5-turbo')

    # Add average lines
    plt.axvline(avg_sentence_length_0, color='blue', linestyle='--', linewidth=0.5)
    plt.axvline(avg_sentence_length_1, color='red', linestyle='--', linewidth=0.5)

    # Add title and labels
    plt.title('Distribution of Average Sentence Length for Human and GPT-3.5-turbo')
    plt.xlabel('Average Sentence Length')
    plt.ylabel('Density')

    # Show the legend and the plot
    plt.legend()
    plt.show()


def plot_avg_sentence_cosine_similarity_distribution(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Filter rows for label 0 and 1
    df_0 = df[df['label'] == 0]
    df_1 = df[df['label'] == 1]

    # Set a theme for prettier plots
    sns.set(style="white")  # set style to white to remove grids

    # Create the KDE plot for label 0
    sns.kdeplot(data=df_0, x='avg_sentence_cosine_similarity', fill=True, color='lightblue', label='Human')

    # Create the KDE plot for label 1
    sns.kdeplot(data=df_1, x='avg_sentence_cosine_similarity', fill=True, color='salmon', label='GPT2-large')

    # Add title and labels
    plt.title('[GPT2-large] Distribution of Average Cosine Similarity between Sentences for Human and LM-generated Text')
    plt.xlabel('Average Sentence Cosine Similarity')
    plt.ylabel('Density')

    # Add vertical lines for mean values
    plt.axvline(df_0['avg_sentence_cosine_similarity'].mean(), color='blue', linestyle='dashed', linewidth=0.5)
    plt.axvline(df_1['avg_sentence_cosine_similarity'].mean(), color='red', linestyle='dashed', linewidth=0.5)

    # Set x-axis limit
    plt.xlim(0.8, 1.1)

    # Show the legend and the plot
    plt.legend()
    plt.show()


# plot_avg_sentence_cosine_similarity_distribution("data_matrix_gpt2-large.csv")


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
