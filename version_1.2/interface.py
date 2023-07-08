import gradio as gr
import pandas as pd
from ML_models import load_model_and_scaler
from training_matrix import prepare_single_text_for_regression


def preprocess_and_choose_model(prompt, response, model_choice):
    # Ensure inputs are strings
    assert isinstance(prompt, str), "Prompt must be a string."
    assert isinstance(response, str), "Response must be a string."

    # Preprocess inputs
    features = prepare_single_text_for_regression(response, prompt)
    print("Features:", features)  # print features for debugging

    # Turn features into a DataFrame (assuming 'features' is a dict)
    X = pd.DataFrame([features], columns=features.keys())

    model_file = f"trained_model_{model_choice.lower()}.pkl"
    scaler_file = "trained_scaler.pkl"

    model, scaler = load_model_and_scaler(model_file, scaler_file)
    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)

    return y_pred[0]


iface = gr.Interface(
    fn=preprocess_and_choose_model,
    inputs=[
        gr.inputs.Textbox(lines=2, label="Prompt"),
        gr.inputs.Textbox(lines=2, label="Response"),
        gr.inputs.Dropdown(choices=['LogReg', 'SVM', 'RF', 'Ensemble'], label="Model Choice")
    ],
    outputs="text"
)
iface.launch()
