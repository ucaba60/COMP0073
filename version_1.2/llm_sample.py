# Imports
import pandas as pd
import openai
import csv
import os
from datasets import preprocess_and_save
import time

# Constants
BATCH_SIZE = 10  # Define the batch size
openai.api_key = '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'


def generate_gpt3_responses(prompt_csv_path, response_folder_path, model="gpt-3.5-turbo", temperature=1):
    """
    Generate GPT-3 responses for a list of prompts saved in a csv file.

    Args:
        prompt_csv_path (str): Path to the csv file containing the prompts.
        response_folder_path (str): Path to the folder where the responses will be saved.
        model (str, optional): The ID of the model to use. Defaults to "gpt-3.5-turbo".
        temperature (float, optional): Determines the randomness of the AI's output. Defaults to 1, as per OpenAI docs.

    Returns:
        None, generates a csv file with the responses.
    """

    # Load the prompts
    df = pd.read_csv(prompt_csv_path)
    prompts = df['Prompt'].tolist()

    # Initialize the starting point
    start = 0

    # Construct the response file path
    response_csv_path = os.path.join(response_folder_path, f"{model}_responses.csv")

    # Check if the response file already exists
    if os.path.exists(response_csv_path):
        # If so, get the number of completed prompts from the file
        with open(response_csv_path, "r", newline="", encoding='utf-8') as file:
            start = sum(1 for row in csv.reader(file)) - 1  # Subtract 1 for the header

    while start < len(prompts):
        try:
            # Process the remaining prompts in batches
            for i in range(start, len(prompts), BATCH_SIZE):
                batch = prompts[i:i + BATCH_SIZE]
                responses = []

                for prompt in batch:
                    # Generate the response
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature
                    )

                    # Append the response to the list
                    responses.append('<<RESP>> ' + response['choices'][0]['message']['content'].strip())

                # Save the responses to a new DataFrame
                response_df = pd.DataFrame({
                    'Prompt': batch,
                    'Response': responses
                })

                # Write the DataFrame to the CSV file, appending if it already exists
                if os.path.exists(response_csv_path):
                    response_df.to_csv(response_csv_path, mode='a', header=False, index=False)
                else:
                    response_df.to_csv(response_csv_path, mode='w', index=False)

                print(f"Batch {i // BATCH_SIZE + 1} completed")
                start = i + BATCH_SIZE
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Sleeping for 10 seconds before retrying...")
            time.sleep(10)  # wait for 10 seconds before retrying


# generate_gpt3_responses('extracted_data/prompts.csv', 'extracted_data', temperature=1) #temeprature is arbirtary this is the default value as per OpenAI docs.


def extract_and_combine(response_csv_path):
    """
    Load 'Prompt' and 'Response' from the generated responses csv file, remove the '<<RESP>>' string,
    adjust the format to match the original datasets, add a label 1 to every instance,
    and save to a new csv file.

    Args:
        response_csv_path (str): Path to the csv file containing the generated responses.

    Returns:
        None, generates a csv file with the combined text and labels.
    """
    # Load the responses
    df = pd.read_csv(response_csv_path)

    # Remove the '<<RESP>>' string from each response
    df['Response'] = df['Response'].str.replace('<<RESP>> ', '')

    # Replace the specific string in the prompt
    df['Prompt'] = df['Prompt'].str.replace(
        'Write an abstract for a scientific paper that answers the Question:', 'Answer:')

    # Combine the prompt and the response in a new column 'Text' with adjustments for specific prompts
    df['Text'] = df.apply(
        lambda row: (
            'Prompt: ' + row['Prompt'].replace(' Continue the story:', '') + ' Story: ' + row['Response']
            if row['Prompt'].endswith('Continue the story:')
            else (
                'Summary: ' + row['Prompt'].replace('Write a news article based on the following summary: ',
                                                    '') + ' Article: ' + row['Response']
                if row['Prompt'].startswith('Write a news article based on the following summary:')
                else row['Prompt'] + ' ' + row['Response']
            )
        ), axis=1
    )

    # Remove 'Title:' and/or 'Abstract:' if they appear after 'Answer:'
    df['Text'] = df['Text'].str.replace(r'Answer: (Title:|Abstract:)', 'Answer:', regex=True)

    # Remove 'Abstract:' if it appears after 'Answer:'
    df['Text'] = df['Text'].str.replace(r'Answer:.*Abstract:', 'Answer:', regex=True)

    # Remove 'Abstract:' if it appears in the text
    df['Text'] = df['Text'].str.replace('Abstract:', '', regex=False)

    # Add a new column 'Label' with value 1 to each instance
    df['Label'] = 1

    # Keep only the 'Text' and 'Label' columns
    df = df[['Text', 'Label']]

    # Print the number of entries pre-processed
    num_entries = len(df)
    print(f"Number of entries pre-processed: {num_entries}")

    # Construct the output file path based on the response file path
    base_path, extension = os.path.splitext(response_csv_path)
    output_csv_path = f"{base_path}_preprocessed{extension}"

    # Check if the output file already exists
    if os.path.isfile(output_csv_path):
        overwrite = input(f"{output_csv_path} already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.lower() != 'y':
            print("Operation cancelled.")
            return

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False)

# extract_and_combine("extracted_data/gpt-3.5-turbo_responses.csv")


# preprocess_and_save(gpt_dataset='gpt-3.5-turbo_responses_preprocessed.csv', gpt_dataset_path='extracted_data',
# output_folder='extracted_data')
