import gradio as gr
import pandas as pd
from ML_models import load_model_and_scaler
from training_matrix import prepare_single_text_for_regression
import joblib
import numpy as np


def preprocess_and_choose_model(prompt, response, model_choice, training_corpus):
    # Ensure inputs are strings
    assert isinstance(prompt, str), "Prompt must be a string."
    assert isinstance(response, str), "Response must be a string."

    # Remove newlines, strip leading/trailing white spaces and ensure there's only one white space between words
    prompt = ' '.join(prompt.replace("\n", " ").strip().split())
    response = ' '.join(response.replace("\n", " ").strip().split())

    # Preprocess inputs
    features = prepare_single_text_for_regression(response, prompt, training_corpus)
    print("Features:", features)  # print features for debugging

    # Load the saved feature names
    feature_names = joblib.load(f'model_data/{training_corpus}/feature_names.pkl')

    # Turn features into a DataFrame (assuming 'features' is a dict)
    # Use the loaded feature names as the columns
    X = pd.DataFrame([features], columns=feature_names)
    model_name_mapping = {
        'Logistic Regression': 'logreg',
        'SVM': 'svm',
        'Random Forest': 'rf',
        'Ensemble': 'ensemble'
    }

    model_file = f"model_data/{training_corpus}/trained_model_{model_name_mapping[model_choice]}.pkl"
    print(f"Loaded data from {training_corpus}")
    scaler_file = f"model_data/{training_corpus}/trained_scaler.pkl"

    model, scaler = load_model_and_scaler(model_file, scaler_file)
    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)

    # Create a message based on the prediction
    if y_pred[0] == 0:
        message = "This text is likely human-generated."
    else:
        message = "This text is likely AI-generated."

    # Add the log probabilities to the message
    log_proba = np.log(y_pred_proba[0])
    message += f"\n\nLog probability of being human-generated: {log_proba[0]:.2f}"
    message += f"\nLog probability of being AI-generated: {log_proba[1]:.2f}"

    # Add the line about the training corpus used
    message += f"\nTraining Corpus Used: {training_corpus}"

    return message


iface = gr.Interface(
    fn=preprocess_and_choose_model,
    inputs=[
        gr.inputs.Textbox(lines=2, label="Headline/Title of Article/Book etc."),
        gr.inputs.Textbox(lines=2, label="Response/Long-Text/Article etc."),
        gr.inputs.Dropdown(choices=['Logistic Regression', 'SVM', 'Random Forest', 'Ensemble'], label="Model Choice"),
        gr.inputs.Dropdown(choices=['gpt-3.5-turbo', 'gpt-j1x','gpt2-large'], label="Training Corpus")  # add more
        # choices if you have more training corpora
    ],
    outputs="text"
)
iface.launch()
