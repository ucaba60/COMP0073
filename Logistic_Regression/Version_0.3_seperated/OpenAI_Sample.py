import pandas as pd
import openai
import time
import csv
import os

BATCH_SIZE = 5  # Define the batch size
openai.api_key = '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'


def generate_gpt_responses(prompt_csv_path, response_csv_path, model="gpt-3.5-turbo", temperature=1):
    """
    Generate GPT-3 responses for a list of prompts saved in a csv file.

    Args:
        prompt_csv_path (str): Path to the csv file containing the prompts.
        response_csv_path (str): Path to the csv file where the responses will be saved.
        model (str, optional): The ID of the model to use. Defaults to "gpt-3.5-turbo".
        temperature (float, optional): Determines the randomness of the AI's output. Defaults to 1, as per OpenAI docs.

    Returns:
        None, generates a csv file with the responses.
    """

    # Load the prompts
    df = pd.read_csv(prompt_csv_path)
    prompts = df['text'].tolist()

    # Initialize the starting point
    start = 0

    # Check if the response file already exists
    if os.path.exists(response_csv_path):
        # If so, get the number of completed prompts from the file
        with open(response_csv_path, "r", newline="", encoding='utf-8') as file:
            start = sum(1 for row in csv.reader(file)) - 1  # Subtract 1 for the header

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
                temperature=temperature,
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


# generate_gpt_responses("Labelled_Data/prompts.csv", "Labelled_Data/t1_responses.csv")
def check_lengths(prompt_csv_path, response_csv_path):
    prompts_df = pd.read_csv(prompt_csv_path)
    responses_df = pd.read_csv(response_csv_path)

    if len(prompts_df) == len(responses_df):
        print("Both files have the same length!")
    else:
        print(f"Length of prompts file: {len(prompts_df)}")
        print(f"Length of responses file: {len(responses_df)}")


def extract_and_combine(response_csv_path, output_csv_path):
    """
    Load 'Prompt' and 'Response' from the generated responses csv file, remove the '<<RESP>>' string,
    combine them in a new column 'Text', add a label 1 to every instance, and save to a new csv file.

    Args:
        response_csv_path (str): Path to the csv file containing the generated responses.
        output_csv_path (str): Path to the csv file where the combined text and labels will be saved.

    Returns:
        None, generates a csv file with the combined text and labels.
    """
    # Load the responses
    df = pd.read_csv(response_csv_path)

    # Remove the '<<RESP>>' string from each response
    df['Response'] = df['Response'].str.replace('<<RESP>> ', '')

    # Combine the prompt and the response in a new column 'Text'
    df['Text'] = df['Prompt'] + ' ' + df['Response']

    # Add a new column 'Label' with value 1 to each instance
    df['Label'] = 1

    # Keep only the 'Text' and 'Label' columns
    df = df[['Text', 'Label']]

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False)



