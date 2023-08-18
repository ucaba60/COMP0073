from datetime import datetime
import os
import gradio as gr
import pandas as pd
from ML_models import load_model_and_scaler
from training_matrix import prepare_single_text_for_regression
import joblib


def save_feedback(text, prediction, feedback):
    timestamp = datetime.now()
    date_str = timestamp.strftime("%Y%m%d")
    time_str = timestamp.strftime("%H%M%S")
    data = {"text": text, "prediction": prediction, "feedback": feedback, "time": time_str}
    df = pd.DataFrame([data])
    directory = f"feedback/{date_str}/"
    os.makedirs(directory, exist_ok=True)
    file_path = f"{directory}/feedback.csv"

    # If file does not exist, write with header
    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False)
    else:  # else it exists so append without writing the header
        df.to_csv(file_path, mode='a', header=False, index=False)


def preprocess_and_choose_model(prompt, response, model_choice, training_corpus, feedback=None):
    # Ensure inputs are strings
    assert isinstance(prompt, str), "Prompt must be a string."
    assert isinstance(response, str), "Response must be a string."

    # Remove newlines, strip leading/trailing white spaces and ensure there's only one white space between words
    prompt = ' '.join(prompt.replace("\n", " ").strip().split())
    response = ' '.join(response.replace("\n", " ").strip().split())

    # Preprocess inputs
    features = prepare_single_text_for_regression(response, prompt, training_corpus)

    # Load the saved feature names
    feature_names = joblib.load(f'model_data/{training_corpus}/feature_names.pkl')

    # Turn features into a DataFrame (assuming 'features' is a dict)
    X = pd.DataFrame([features], columns=feature_names)
    model_name_mapping = {
        'Logistic Regression': 'logreg',
        'SVM': 'svm',
        'Random Forest': 'rf',
        'Ensemble': 'ensemble'
    }

    model_file = f"model_data/{training_corpus}/trained_model_{model_name_mapping[model_choice]}.pkl"
    scaler_file = f"model_data/{training_corpus}/trained_scaler.pkl"

    model, scaler = load_model_and_scaler(model_file, scaler_file)
    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)

    # Create a message based on the prediction
    if y_pred[0] == 0:
        message = f"This text is likely human-generated with {y_pred_proba[0][0] * 100:.2f}% probability."
    else:
        message = f"This text is likely AI-generated with {y_pred_proba[0][1] * 100:.2f}% probability."

    # Add the line about the training corpus used
    message += f"\n\nTraining Corpus Used: {training_corpus}"

    # Save feedback if provided
    if feedback is not None:
        # You might want to modify the save_feedback function to suit your requirements
        save_feedback({'prompt': prompt, 'response': response}, message, feedback)

    return message


iface = gr.Interface(
    fn=preprocess_and_choose_model,
    inputs=[
        gr.inputs.Textbox(lines=2, label="Headline/Title of Article/Book etc."),
        gr.inputs.Textbox(lines=2, label="Response/Long-Text/Article etc."),
        gr.inputs.Dropdown(choices=['Logistic Regression', 'SVM', 'Random Forest', 'Ensemble'], label="Model Choice"),
        gr.inputs.Dropdown(choices=['gpt-3.5-turbo', 'gpt-j1x', 'gpt2-large'], label="Training Corpus"),
        # add more choices if you have more training corpora
        gr.inputs.Radio(choices=["Agree", "Disagree"], label="Do you agree with the prediction?", optional=True)
    ],
    outputs="text",
    description="IMPORTANT: In order for your feedback to be recorded the 'Submit' button needs to be pressed twice. "
                "First, "
                "when computing the probabilities and the second time when you input your response as to whether you "
                "agree with the model's decision.",
    allow_flagging=False
)

iface.launch()
