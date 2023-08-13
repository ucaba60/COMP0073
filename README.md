# COMP0073
Summer Project Code

                                                          **  Supplementary Explanation of Code and File Structure**

The code is separated into .py files depending on the functionality. There are also folders where data is stored in a .csv format. Generally, running interface.py starts the application.

Here is a brief summary of what each .py file does and what each folder contains:

Datasets.py – this is functions related to extracting, processing and combining data from the human datasets (PubMedQA, WritingPrompts, CNN_Dailymail) in the format ‘Question:Answer’.

§ This requires the exitance of the data folder, where WritingPrompts data is located.

§ This creates a file prompts.csv in the folder extracted_data.

§ This creates a file combined_human_data.csv containing all human text in extracted_data.

Llm_sample.py – functions related to using the ‘Questions’ as prompts for various language models (GPT-3.5-trubo).

· This creates the file gpt-3.5-turbo_repsonses.csv in extracted_data. This is the raw responses.

· This creates the file gpt-3.5-turbo_responses_preprocessed in extracted_data. These are the responses with some pre-processing.

· This creates the gpt-3.5-turbo_and_human_data.csv in extracted_data. This contains the pre-processed texts from both human and AI sources. This is the file from which features are extracted.

Feature_extraction.py – these are the main functions that extract the features (POS-tags, cosine similarity etc.) from the texts.

Training_matrix.py – this creates a data_matrix.csv file which is the result of extracting all the features. These files are used for training.

ML_models.py – training and fine-tuning ML models.

· The models are saved in the folder model_data/[model name]

Interface.py – the function that uses Gradio to start the application.

Folders:

Extracted_data: Contains the prompts, the AI-responses, the human text and the combined text.

Model_data: Contains .pkl files for each machine learning model.
