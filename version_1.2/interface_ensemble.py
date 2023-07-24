import gradio as gr
import pandas as pd
import joblib
from datetime import datetime
import os
from ML_models import load_model_and_scaler
from training_matrix import prepare_single_text_for_regression


def preprocess_and_predict(prompt, response):
    assert isinstance(prompt, str), "Prompt must be a string."
    assert isinstance(response, str), "Response must be a string."

    prompt = ' '.join(prompt.replace("\n", " ").strip().split())
    response = ' '.join(response.replace("\n", " ").strip().split())

    # Preprocess inputs
    features = prepare_single_text_for_regression(response, prompt, 'gpt-3.5-turbo')

    # Load the saved feature names
    feature_names = joblib.load('model_data/gpt-3.5-turbo/feature_names.pkl')

    # Turn features into a DataFrame (assuming 'features' is a dict)
    X = pd.DataFrame([features], columns=feature_names)

    model_file = "model_data/gpt-3.5-turbo/trained_model_ensemble.pkl"
    scaler_file = "model_data/gpt-3.5-turbo/trained_scaler.pkl"

    model, scaler = load_model_and_scaler(model_file, scaler_file)
    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)

    # Create a message based on the prediction
    if y_pred[0] == 0 and y_pred_proba[0][0] > 0.7:
        message = f"The text is likely human-generated with {y_pred_proba[0][0] * 100:.2f}% probability."
    elif y_pred[0] == 1 and y_pred_proba[0][1] > 0.7:
        message = f"The text is likely AI-generated with {y_pred_proba[0][1] * 100:.2f}% probability."
    else:
        message = f"The model is unsure (confidence less than 70%) about the source of the text."

    return message


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
    else: # else it exists so append without writing the header
        df.to_csv(file_path, mode='a', header=False, index=False)


def test_and_get_feedback(prompt, response, feedback=None):
    prediction = preprocess_and_predict(prompt, response)
    if feedback is not None:
        save_feedback({'prompt': prompt, 'response': response}, prediction, feedback)
    return prediction


iface = gr.Interface(
    fn=test_and_get_feedback,
    inputs=[
        gr.inputs.Textbox(lines=2, label="Headline/Title of Article/Book etc."),
        gr.inputs.Textbox(lines=2, label="Response/Long-Text/Article etc."),
        gr.inputs.Radio(choices=["Agree", "Disagree"], label="Do you agree with the prediction?", optional=True)
    ],
    outputs="text",
    description="IMPORTANT: In order for your feedback to be recorded the 'Submit' button needs to be pressed twice. First "
                "when computing the probabilities and the second time when you input your response as to whether you "
                "agree with the model's decision.",
    allow_flagging=False
)
iface.launch()

