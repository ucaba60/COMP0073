import gradio as gr
import pandas as pd
from ML_models import load_model_and_scaler
from training_matrix import prepare_single_text_for_regression
import joblib
import numpy as np


def preprocess_and_choose_model(prompt, response, model_choice):
    # Ensure inputs are strings
    assert isinstance(prompt, str), "Prompt must be a string."
    assert isinstance(response, str), "Response must be a string."

    # Preprocess inputs
    features = prepare_single_text_for_regression(response, prompt)
    print("Features:", features)  # print features for debugging

    # Load the saved feature names
    feature_names = joblib.load('model_data/feature_names.pkl')

    # Turn features into a DataFrame (assuming 'features' is a dict)
    # Use the loaded feature names as the columns
    X = pd.DataFrame([features], columns=feature_names)
    print(X)
    model_name_mapping = {
        'Logistic Regression': 'logreg',
        'SVM': 'svm',
        'Random Forest': 'rf',
        'Ensemble': 'ensemble'
    }

    model_file = f"model_data/trained_model_{model_name_mapping[model_choice]}.pkl"
    scaler_file = "model_data/trained_scaler.pkl"

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

    return message


iface = gr.Interface(
    fn=preprocess_and_choose_model,
    inputs=[
        gr.inputs.Textbox(lines=2, label="Prompt"),
        gr.inputs.Textbox(lines=2, label="Response"),
        gr.inputs.Dropdown(choices=['Logistic Regression', 'SVM', 'Random Forest', 'Ensemble'], label="Model Choice")
    ],
    outputs="text"
)
iface.launch()
