import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
from feature_extraction import compute_difference_tfidf_words


def plot_text_perplexity_distribution(file_path, save_as_pdf=False):
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
    sns.kdeplot(data=df_1, x='text_perplexity', fill=True, color='salmon', label='GPT-J Text')

    # Add a vertical line at the average perplexity for label 0
    plt.axvline(x=avg_perplexity_0, color='blue', linestyle='--', alpha=0.8,
                linewidth=1.5, label='Average Perplexity (Human Text)')

    # Add a vertical line at the average perplexity for label 1
    plt.axvline(x=avg_perplexity_1, color='red', linestyle='--', alpha=0.8,
                linewidth=1.5, label='Average Perplexity (GPT-J Text)')

    # Add title and labels
    plt.title('[GPT-J] Distribution of Text Perplexity for Human and LM-generated Text')
    plt.xlabel('Text Perplexity')
    plt.ylabel('Density')

    # Set x-axis limit
    plt.xlim(0.9, 1.4)

    # Show the legend and the plot with smaller legend text
    plt.legend(fontsize='small')

    # Save as PDF if desired
    if save_as_pdf:
        plt.savefig('plot_output.pdf', format='pdf', dpi=300, bbox_inches='tight')
    else:
        plt.show()


# plot_text_perplexity_distribution("data_matrix_gpt-j1x.csv", save_as_pdf=True)
# plot_text_perplexity_distribution("data_matrix_gpt-3.5-turbo.csv", save_as_pdf=True)
# plot_text_perplexity_distribution("data_matrix_gpt2-large.csv", save_as_pdf=True)

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

    # Create a grouped bar chart with specified bar width
    ax = df_sum_percentage.rename(columns={0: 'Human', 1: 'GPT-J'}).plot(kind='bar', stacked=False,
                                                                         figsize=(12, 10),
                                                                         width=0.7)  # Adjust width here

    # Add percentage annotations on top of each bar
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                    xytext=(0, 5), textcoords='offset points', fontsize=10)  # Increase fontsize from 8 to 10

    # Add title and labels with larger font size
    plt.title('[GPT-J] Parts-of-Speech Frequency Comparison between Human and LM-generated Texts', fontsize=16)
    plt.xlabel('Parts-of-Speech', fontsize=14)
    plt.ylabel('Frequency (%)', fontsize=14)

    # Set larger tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Show the legend with larger font size
    plt.legend(title='Label', loc='upper right', fontsize=12)  # Set fontsize in legend

    # Adjust layout
    plt.tight_layout(rect=[0, 0.01, 1, 0.95])  # Increase the bottom margin

    # Save the plot to a PDF file before displaying it
    plt.savefig('plot_output.pdf', format='pdf')

    # Display the plot
    plt.show()


# Call the function to display the plot with the updated adjustments
# plot_column_frequencies("data_matrix_gpt-3.5-turbo.csv")
# plot_column_frequencies("data_matrix_gpt-j1x.csv")
# plot_column_frequencies("data_matrix_gpt2-large.csv")

def plot_prompt_text_cosine_similarity_distribution(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Filter rows for label 0 and 1
    df_0 = df[df['label'] == 0]
    df_1 = df[df['label'] == 1]

    # Set a theme for prettier plots
    sns.set(style="white")  # set style to white to remove grids

    # Set a larger figure size
    plt.figure(figsize=(10, 6))  # Increase the width to 10 from the default size

    # Create the KDE plot for label 0
    sns.kdeplot(data=df_0, x='prompt_text_cosine_similarity', fill=True, color='lightblue', label='Human')

    # Create the KDE plot for label 1
    sns.kdeplot(data=df_1, x='prompt_text_cosine_similarity', fill=True, color='salmon', label='GPT2-large')

    # Add title and labels
    plt.title(
        '[GPT2-large] Distribution of Cosine Similarity between Prompt & Response for Human and LM-generated Texts')
    plt.xlabel('Cosine Similarity between Prompt & Response')
    plt.ylabel('Density')

    # Add vertical lines for mean values
    plt.axvline(x=df_0['prompt_text_cosine_similarity'].mean(), color='blue', linestyle='--', alpha=0.8, linewidth=1.5,
                label='Average Cosine Similarity (Human Text)')
    plt.axvline(x=df_1['prompt_text_cosine_similarity'].mean(), color='red', linestyle='--', alpha=0.8, linewidth=1.5,
                label='Average Cosine Similarity (GPT2-large)')

    # Show the legend and the plot
    plt.legend()
    plt.tight_layout()

    # Save the plot as a PDF file
    plt.savefig("cosine_similarity_distribution.pdf", bbox_inches='tight')

    # Display the plot
    plt.show()


# Call the function to display and save the plot
# plot_prompt_text_cosine_similarity_distribution("data_matrix_gpt-j1x.csv")
# plot_prompt_text_cosine_similarity_distribution("data_matrix_gpt-3.5-turbo.csv")
# plot_prompt_text_cosine_similarity_distribution("data_matrix_gpt2-large.csv")


def plot_avg_readability_metrics(file_paths):
    # Initialize an empty DataFrame to store average metrics
    avg_metrics = pd.DataFrame(columns=['Source', 'flesch_reading_ease'])

    # Calculate the average flesch_reading_ease for Human texts
    human_df = pd.read_csv(file_paths[0])
    human_avg = human_df[human_df['label'] == 0]['flesch_reading_ease'].mean()
    human_row = pd.DataFrame([['Human', human_avg]], columns=['Source', 'flesch_reading_ease'])
    avg_metrics = pd.concat([avg_metrics, human_row], ignore_index=True)

    # Define a dictionary to map file names to more readable names
    name_map = {
        'gpt-3.5-turbo': 'GPT-3.5-turbo',
        'gpt-j1x': 'GPT-J',
        'gpt2-large': 'GPT2-large'
    }

    # Calculate the average flesch_reading_ease for each file
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        gpt_avg = df[df['label'] == 1]['flesch_reading_ease'].mean()

        # Extract the model name from the file path and map it to a readable name
        gpt_name = re.search('data_matrix_(.*).csv', file_path).group(1)
        gpt_name_readable = name_map[gpt_name]

        # Append the readable name and the average reading ease score to the dataframe
        gpt_row = pd.DataFrame([[gpt_name_readable, gpt_avg]], columns=['Source', 'flesch_reading_ease'])
        avg_metrics = pd.concat([avg_metrics, gpt_row], ignore_index=True)

    # Set plot size
    plt.figure(figsize=(10, 6))

    # Create a bar plot
    bar_plot = sns.barplot(x='flesch_reading_ease', y='Source', data=avg_metrics, palette='viridis')

    # Add title and labels
    plt.title('Average Reading Ease for Human and Different GPT Models')
    plt.xlabel('Average Flesch Reading Ease')
    plt.ylabel('Language Model')

    # Add the average values at the end of the bars
    for bar in bar_plot.containers[0]:
        bar_plot.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.2f}',
                      va='center', ha='left')

    # Add a margin to the x-axis
    plt.xlim(0, max(avg_metrics['flesch_reading_ease']) * 1.1)

    # Save the plot as a PDF file
    plt.savefig("average_readability_metrics.pdf", bbox_inches='tight')

    # Show the plot
    plt.show()


# # Call the function to display the plot
# file_paths = ["data_matrix_gpt-3.5-turbo.csv", "data_matrix_gpt-j1x.csv", "data_matrix_gpt2-large.csv"]
# plot_avg_readability_metrics(file_paths)


def plot_avg_sentence_length(file_paths):
    # Initialize an empty DataFrame to store average metrics
    avg_sentence_lengths = pd.DataFrame(columns=['Source', 'avg_sentence_length'])

    # Calculate the average sentence length for Human texts
    human_df = pd.read_csv(file_paths[0])
    human_avg = human_df[human_df['label'] == 0]['avg_sentence_length'].mean()
    human_row = pd.DataFrame([['Human', human_avg]], columns=['Source', 'avg_sentence_length'])
    avg_sentence_lengths = pd.concat([avg_sentence_lengths, human_row], ignore_index=True)

    # Define a dictionary to map file names to more readable names
    name_map = {
        'gpt-3.5-turbo': 'GPT-3.5-turbo',
        'gpt-j1x': 'GPT-J',
        'gpt2-large': 'GPT2-large'
    }

    # Calculate the average sentence length for each file
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        gpt_avg = df[df['label'] == 1]['avg_sentence_length'].mean()

        # Extract the model name from the file path and map it to a readable name
        gpt_name = re.search('data_matrix_(.*).csv', file_path).group(1)
        gpt_name_readable = name_map[gpt_name]

        # Append the readable name and the average sentence length to the dataframe
        gpt_row = pd.DataFrame([[gpt_name_readable, gpt_avg]], columns=['Source', 'avg_sentence_length'])
        avg_sentence_lengths = pd.concat([avg_sentence_lengths, gpt_row], ignore_index=True)

    # Create a bar plot
    plt.figure(figsize=[10, 6])
    bar_plot = sns.barplot(x='avg_sentence_length', y='Source', data=avg_sentence_lengths, palette='viridis')

    # Add title and labels
    plt.title('Average Sentence Length for Human and Different GPT Models')
    plt.xlabel('Average Sentence Length')
    plt.ylabel('Language Model')

    # Add the average values at the end of the bars
    for bar in bar_plot.containers[0]:
        bar_plot.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.2f}',
                      va='center', ha='left', color='black')

    # Save the plot as a PDF file
    plt.savefig("average_sentence_length.pdf", bbox_inches='tight')

    # Show the plot
    plt.show()


# Call the function to display the plot
# file_paths = ["data_matrix_gpt-3.5-turbo.csv", "data_matrix_gpt-j1x.csv", "data_matrix_gpt2-large.csv"]
# plot_avg_sentence_length(file_paths)


def plot_avg_word_length(file_paths):
    # Initialize an empty DataFrame to store average metrics
    avg_word_lengths = pd.DataFrame(columns=['Source', 'avg_word_length'])

    # Calculate the average word length for Human texts
    human_df = pd.read_csv(file_paths[0])
    human_avg = human_df[human_df['label'] == 0]['avg_word_length'].mean()
    human_row = pd.DataFrame([['Human', human_avg]], columns=['Source', 'avg_word_length'])
    avg_word_lengths = pd.concat([avg_word_lengths, human_row], ignore_index=True)

    # Define a dictionary to map file names to more readable names
    name_map = {
        'gpt-3.5-turbo': 'GPT-3.5-turbo',
        'gpt-j1x': 'GPT-J',
        'gpt2-large': 'GPT2-large'
    }

    # Calculate the average word length for each file
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        gpt_avg = df[df['label'] == 1]['avg_word_length'].mean()

        # Extract the model name from the file path and map it to a readable name
        gpt_name = re.search('data_matrix_(.*).csv', file_path).group(1)
        gpt_name_readable = name_map[gpt_name]

        # Append the readable name and the average word length to the dataframe
        gpt_row = pd.DataFrame([[gpt_name_readable, gpt_avg]], columns=['Source', 'avg_word_length'])
        avg_word_lengths = pd.concat([avg_word_lengths, gpt_row], ignore_index=True)

    # Create a bar plot
    plt.figure(figsize=[10, 6])
    bar_plot = sns.barplot(x='avg_word_length', y='Source', data=avg_word_lengths, palette='viridis')

    # Add title and labels
    plt.title('Average Word Length for Human and Different GPT Models')
    plt.xlabel('Average Word Length')
    plt.ylabel('Language Model')

    # Add the average values at the end of the bars
    for bar in bar_plot.containers[0]:
        bar_plot.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.2f}',
                      va='center', ha='left', color='black')

    # Save the plot as a PDF file
    plt.savefig("average_word_length.pdf", bbox_inches='tight')

    # Show the plot
    plt.show()


# Call the function to display the plot
# file_paths = ["data_matrix_gpt-3.5-turbo.csv", "data_matrix_gpt-j1x.csv", "data_matrix_gpt2-large.csv"]
# plot_avg_word_length(file_paths)


def plot_avg_sentence_cosine_similarity_distribution(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Filter rows for label 0 and 1
    df_0 = df[df['label'] == 0]
    df_1 = df[df['label'] == 1]

    # Set a theme for prettier plots
    sns.set(style="white")  # set style to white to remove grids

    # Increase figure size
    plt.figure(figsize=(12, 6))  # Increase width to 12

    # Create the KDE plot for label 0
    sns.kdeplot(data=df_0, x='avg_sentence_cosine_similarity', fill=True, color='lightblue', label='Human')

    # Create the KDE plot for label 1
    sns.kdeplot(data=df_1, x='avg_sentence_cosine_similarity', fill=True, color='salmon', label='GPT2-large')

    # Add title and labels
    plt.title(
        '[GPT2-large] Distribution of Average Cosine Similarity between Sentences for Human and LM-generated Text')
    plt.xlabel('Average Cosine Similarity between Sentences')
    plt.ylabel('Density')

    # Add vertical lines for mean values
    plt.axvline(x=df_0['avg_sentence_cosine_similarity'].mean(), color='blue', linestyle='--', alpha=0.8, linewidth=1.5,
                label='Average Cosine Similarity (Human Text)')
    plt.axvline(x=df_1['avg_sentence_cosine_similarity'].mean(), color='red', linestyle='--', alpha=0.8, linewidth=1.5,
                label='Average Cosine Similarity (GPT2-large)')

    # Set x-axis limit
    plt.xlim(0.8, 1.1)

    # Add legend with a title
    plt.legend(title="Label")

    # Save the plot as a PDF file
    plt.savefig("avg_cosine_similarity_distribution.pdf", bbox_inches='tight')

    # Show the plot
    plt.show()


# plot_avg_sentence_cosine_similarity_distribution("data_matrix_gpt2-large.csv")
# plot_avg_sentence_cosine_similarity_distribution("data_matrix_gpt-j1x.csv")
# plot_avg_sentence_cosine_similarity_distribution("data_matrix_gpt-3.5-turbo.csv")


def plot_roc_curve(y_test, y_pred):
    # Calculate ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_pred)
    print("ROC-AUC Score:", roc_auc)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # Plot ROC curve
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC Score = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Plotting the random guess line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('ROC_Curve.pdf', bbox_inches='tight')
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


def print_classification_report(y_test, y_pred):
    print('Classification Report: \n', classification_report(y_test, y_pred))


def plot_feature_importance_lr(model, model_name):
    # Get the coefficients from the logistic regression model
    coefficients = model.coef_[0]

    # Load the feature names
    feature_names = joblib.load(f'model_data/{model_name}/feature_names.pkl')

    # Create a DataFrame to store the feature names and their corresponding coefficients
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    feature_importance_df['Absolute Coefficient'] = feature_importance_df['Coefficient'].abs()

    # Sort the features by their absolute coefficients in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Absolute Coefficient', ascending=False)

    # Plot the feature importances
    plt.figure(figsize=(16, 10))
    sns.barplot(x='Absolute Coefficient', y='Feature', data=feature_importance_df, palette='coolwarm')
    plt.title("Feature Importances (Logistic Regression)")
    plt.xlabel("Absolute Coefficient")
    plt.ylabel("Feature")
    plt.savefig('Feature_Importance.pdf', bbox_inches='tight')
    plt.show()


def plot_feature_importance_rf(model, model_name):
    # Get the feature importances from the Random Forest model
    feature_importances = model.feature_importances_

    # Load the feature names
    feature_names = joblib.load(f'model_data/{model_name}/feature_names.pkl')

    # Create a DataFrame to store the feature names and their corresponding importances
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

    # Sort the features by their importances in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot the feature importances
    plt.figure(figsize=(16, 10))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='coolwarm')
    plt.title("Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.savefig('Feature_Importance_RANDOMFOREST.pdf', bbox_inches='tight')
    plt.show()


def plot_correlation_matrix(X, title):
    corr = X.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20, 20))  # Increase the size of the figure

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    ax.set_title(title)  # Set the title for the diagram

    plt.subplots_adjust(bottom=0.28)  # Further adjust the bottom margin

    # Save the plot as a PDF file
    plt.savefig("correlation_matrix.pdf", bbox_inches='tight')

    # Show the plot
    plt.show()


def plot_permutation_importance(model, X_test, y_test, feature_names):
    # Calculate permutation importances
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort().tolist()  # Convert to list

    # Plot the permutation importances using a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))  # Increase the figsize as needed
    ax.barh([feature_names[i] for i in sorted_idx], result.importances_mean[sorted_idx])

    # Add labels to the axes
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')

    ax.set_title("[GPT-2-large] Permutation Importances (test set)")
    fig.tight_layout()

    # Save the plot as a .pdf
    plt.savefig("permutation_importance_plot.pdf", format='pdf', bbox_inches='tight')
    plt.show()


def plot_diff_tfidf_words(data_file, n_top_words=10):
    """
    This function reads the input data, computes the top n words with the largest average difference in
    TF-IDF scores between human-labelled text and AI-generated text, and plots the results.

    Args:
    data_file (str): The path to the .csv file which contains the texts and labels.
    n_top_words (int): The number of top words to return. Default is 10.

    Returns:
    None
    """

    # get the top words with the largest difference in tf-idf
    diff_words = compute_difference_tfidf_words(data_file, n_top_words)

    # plot the results using seaborn
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("coolwarm",
                                n_top_words)  # using coolwarm palette, you can use any palette of your liking
    sns.barplot(x=diff_words['tfidf_difference'][::-1], y=diff_words['word'][::-1],
                palette=palette)  # reverse order to have the largest bar at top
    plt.xlabel('TF-IDF Difference')
    plt.ylabel('Words')
    plt.title(
        '[GPT-J] Top {} Words with Largest Average Difference in TF-IDF Scores between Human and AI-Generated Text'.format(
            n_top_words))

    # Save the plot as a .pdf
    plt.savefig("tfidf_difference_plot.pdf", format='pdf', bbox_inches='tight')

    plt.show()


# plot_diff_tfidf_words('extracted_data/gpt-3.5-turbo_and_human_data.csv', n_top_words=10)
# plot_diff_tfidf_words('extracted_data/gpt2-large_and_human_data.csv', n_top_words=10)
# plot_diff_tfidf_words('extracted_data/gpt-j1x_and_human_data.csv', n_top_words=10)


def calculate_vif(X):
    vif = pd.DataFrame()
    vif["Features"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif


# # Exclude label column if it is not a feature
# features_df = df.drop(columns=['label'])
#
# vif_scores = calculate_vif(features_df)
# print(vif_scores)

# vif_scores = calculate_vif(df)


def plot_sentiment_frequencies(data_file, sentiment_method):
    # Load the saved data from the CSV file
    data = pd.read_csv(data_file)

    # Select the columns of interest for the specified sentiment extraction method
    column = f'text_sentiment_{sentiment_method}'

    # Map sentiment scores to corresponding labels
    sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

    # Filter data for Label 0 and Label 1 separately
    data_label0 = data[data['label'] == 0]
    data_label1 = data[data['label'] == 1]

    # Calculate the frequencies for each sentiment score and label
    frequencies_label0 = data_label0[column].map(sentiment_labels).value_counts().reindex(sentiment_labels.values(),
                                                                                          fill_value=0)
    frequencies_label1 = data_label1[column].map(sentiment_labels).value_counts().reindex(sentiment_labels.values(),
                                                                                          fill_value=0)

    # Set the order of sentiment scores for the x-axis
    sentiment_order = ['Negative', 'Neutral', 'Positive']

    # Plot the grouped bar chart using Matplotlib
    plt.figure(figsize=(12, 6))

    # Adjust width and offset for overlapping bars
    width = 0.35
    offset = width / 2

    plt.bar(np.arange(len(sentiment_order)) - offset, frequencies_label0, width, label='Human-Text', alpha=0.8,
            color='blue')
    plt.bar(np.arange(len(sentiment_order)) + offset, frequencies_label1, width, label='GPT-Text', alpha=0.8,
            color='red')

    # Add labels with the number of observations on top of each column
    for i, freq_label0, freq_label1 in zip(range(len(sentiment_order)), frequencies_label0, frequencies_label1):
        plt.text(i - offset, freq_label0 + 5, str(freq_label0), ha='center', color='black')
        plt.text(i + offset, freq_label1 + 5, str(freq_label1), ha='center', color='black')

    plt.xlabel('Sentiment Scores')
    plt.ylabel('Frequency')
    plt.title(f'[GPT-3.5-turbo] Frequency of Sentiment Scores ({sentiment_method.capitalize()}) by Label')
    plt.xticks(np.arange(len(sentiment_order)), sentiment_order)
    plt.legend()
    plt.gca().set_facecolor('white')  # Set white background
    plt.grid(False)  # Remove grid lines

    # Save the plot as a PDF file
    plt.savefig("sentiment_frequencies.pdf", bbox_inches='tight')

    # Show the plot
    plt.show()

#
# plot_sentiment_frequencies('text_sentiments/gpt2-large-sentiment.csv', 'roberta')
# # plot_sentiment_frequencies('text_sentiments/gpt2-large-sentiment.csv', 'textblob')
# plot_sentiment_frequencies('text_sentiments/gpt2-large-sentiment.csv', 'nltk')
#
# plot_sentiment_frequencies('text_sentiments/gpt-j1x-sentiment.csv', 'roberta')
# plot_sentiment_frequencies('text_sentiments/gpt-j1x-sentiment.csv', 'textblob')
# plot_sentiment_frequencies('text_sentiments/gpt-j1x-sentiment.csv', 'nltk')


# plot_sentiment_frequencies('text_sentiments/gpt-3.5-turbo-sentiment.csv', 'roberta')
# plot_sentiment_frequencies('text_sentiments/gpt-3.5-turbo-sentiment.csv', 'textblob')
# plot_sentiment_frequencies('text_sentiments/gpt-3.5-turbo-sentiment.csv', 'nltk')
