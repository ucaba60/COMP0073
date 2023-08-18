# GPT-Text Detection via Machine Learning Classifiers

## Note on WritingPrompts
To run functions related to WritingPrompts download the dataset from [here](https://www.kaggle.com/datasets/ratthachat/writing-prompts). Save the data into a directory **data/writingPrompts**.

## File Descriptions
The below is a high-level description of the functionality contained in each .py file:


### interface.py
- **Purpose**: Launches the Gradio application for field-testing. THIS IS THE INTERFACE USED IN THE VIDEO REPORT
- **Steps**:
  1. Loads the pre-trained LR/SVM/RF/Ensemble GPT-3.5-turbo, GPT-J, GPT2-large model `.pkl` file from `model_data`.
  2. Pre-processes the input text and feeds it to the model.

### interface_fieldtest.py
- **Purpose**: Launches the Gradio application for field-testing.
- **Steps**:
  1. Loads the pre-trained SVM GPT-3.5-turbo model `.pkl` file from `model_data/gpt-3.5-turbo`.
  2. Pre-processes the input text and feeds it to the model.

### datasets_gathering.py
- **Purpose**: Functions related to extracting, processing, and combining data from human datasets.
- **Datasets**: PubMedQA, WritingPrompts, CNN_Dailymail (formatted as `Question: Answer`).
- **Dependencies**: Requires the existence of the `data` folder (contains WritingPrompts data).
- **Outputs**:
  - `prompts.csv` in `extracted_data` folder.
  - `combined_human_data.csv` in `extracted_data` folder with all human text.

### llm_sample.py
- **Purpose**: Uses 'Questions' as prompts for GPT-3.5-turbo and GPT2.
- **Outputs**:
  - `gpt-3.5-turbo_responses.csv` in `extracted_data` (raw responses).
  - `gpt-3.5-turbo_responses_preprocessed` in `extracted_data` (pre-processed responses).
  - `gpt-3.5-turbo_and_human_data.csv` in `extracted_data` (pre-processed texts from both human and AI sources for feature extraction).

### gpt-j.py
- **Purpose**: Specific code for extracting GPT-J outputs.

### feature_extraction.py
- **Purpose**: Contains main functions for extracting features (e.g., POS-tags, cosine similarity) from texts.

### training_matrix.py
- **Purpose**: Constructs the data matrix after feature extraction.
- **Output**: `data_matrix.csv` (used for training).

### ML_models.py
- **Purpose**: Training and fine-tuning of machine learning models.
- **Output**: Models are saved in the `model_data/[model name]` folder.

### ML_plots.py
- **Purpose**: Contains code for creating various plots.
- **Output**: Plot data saved in the `output_images` folder.


                                          
