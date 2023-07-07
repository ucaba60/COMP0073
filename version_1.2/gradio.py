import pandas as pd
import gradio as gr
from training_matrix import prepare_single_text_for_regression
import joblib


def get_model_predictions(model, features, scaler):
    # Apply the same scaling as you did for your training data
    scaled_features = scaler.transform(features)

    # Make the prediction
    prediction = model.predict(scaled_features)
    prediction_proba = model.predict_proba(scaled_features)

    return prediction, prediction_proba


def format_prediction(prediction, prediction_proba):
    # Convert the output to a readable format
    if prediction[0] == 0:
        prediction_text = "Human generated"
    else:
        prediction_text = "AI generated"

    # Get the probabilities
    human_prob = prediction_proba[0][0]
    ai_prob = prediction_proba[0][1]

    # Combine all outputs into a string
    output = f"Prediction: {prediction_text}, Human generated Probability: {human_prob}, AI generated Probability: {ai_prob}"

    return output


def predict_text(model_name, input_text, prompt=None):
    # Extract features from the input text
    extracted_features = prepare_single_text_for_regression(input_text, prompt)

    # Convert the features dictionary to a DataFrame
    features_df = pd.DataFrame([extracted_features])

    # Get the model and scaler based on the model_name
    model, scaler = get_model_and_scaler(model_name)

    # Get the predictions
    prediction, prediction_proba = get_model_predictions(model, features_df, scaler)

    # Format the predictions into a readable string
    output = format_prediction(prediction, prediction_proba)

    return output


def get_model_and_scaler(model_name):
    if model_name == 'Logistic Regression':
        model = load_model('model_best.joblib')
        scaler = load_model('scaler.joblib')  # assuming you also saved the scaler with joblib
        return model, scaler


iface = gr.Interface(
    fn=predict_text,
    inputs=[
        gr.inputs.Dropdown(choices=["Logistic Regression"], label="Model"),
        gr.inputs.Textbox(lines=20, placeholder="Enter text here..."),
        gr.inputs.Textbox(lines=2, placeholder="Your prompt")],
    outputs=gr.outputs.Textbox(label="Prediction & Probabilities"),
)
iface.launch()
