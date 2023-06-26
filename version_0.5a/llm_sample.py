import pandas as pd
import openai
import time
import csv
import os

# Constants
BATCH_SIZE = 5  # Define the batch size
openai.api_key = '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'


def generate_gpt_responses(prompt_csv_path, response_csv_path, model="gpt-3.5-turbo", temperature=1):
    """
    Generate GPT-3 responses for a list of prompts saved in a csv file.

    Args:
        prompt_csv_path (str): Path to the csv file containing the prompts.
        response_csv_path (str): Base path to the csv file where the responses will be saved.
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
            # Define max tokens based on the current index
            # The first 150 entries will have 100 max tokens, the rest will have 1000
            # This is because the first 150 entries are short prompts/responses, while the rest are longer
            # max_tokens = 100 if i < 150 else 1000

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

        # Append temperature value to the base filename
        response_csv_path = f"{response_csv_path}_t{temperature}_responses.csv"

        # Write the DataFrame to the CSV file, appending if it already exists
        if os.path.exists(response_csv_path):
            response_df.to_csv(response_csv_path, mode='a', header=False, index=False)
        else:
            response_df.to_csv(response_csv_path, mode='w', index=False)

        print(f"Batch {i // BATCH_SIZE + 1} completed")


# ------------------------------------------------------------------------------------------#

# generate_gpt_responses("Labelled_Data/prompts.csv", "Labelled_Data/t1_responses.csv")

# ------------------------------------------------------------------------------------------#


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

    # Combine the prompt and the response in a new column 'Text' with adjustments for specific prompts
    df['Text'] = df.apply(
        lambda row: (
            'Prompt: ' + row['Prompt'].replace(' Continue the story:', '') + ' Story: ' + row['Response']
            if row['Prompt'].endswith('Continue the story:')
            else (
                'Summary: ' + row['Prompt'].replace('Write a news article based on the following summary: ', '') + ' Article: ' + row['Response']
                if row['Prompt'].startswith('Write a news article based on the following summary:')
                else row['Prompt'] + ' ' + row['Response']
            )
        ), axis=1
    )

    # Add a new column 'Label' with value 1 to each instance
    df['Label'] = 1

    # Keep only the 'Text' and 'Label' columns
    df = df[['Text', 'Label']]

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



# ------------------------------------------------------------------------------------------#

extract_and_combine('t1_responses.csv')

def extract_prompts_and_save():
    """
    Extracts prompts from the combined dataset and saves them to a .csv file.
    """
    # Load the combined dataset
    df = pd.read_csv('combined_source_data.csv')
    combined_data = list(zip(df['Text'], df['Label']))

    # Extract prompts from the combined data
    prompts = []
    for full_text, _ in combined_data:
        if 'Question:' in full_text and 'Answer:' in full_text:
            prompts.append(full_text.split('Answer:')[0] + 'Answer:')
        elif 'Summary:' in full_text and 'Article:' in full_text:
            prompts.append('Write a news article based on the following summary: ' +
                           full_text.split('Summary:')[1].split('Article:')[0].strip())
        elif 'Prompt:' in full_text and 'Story:' in full_text:
            prompts.append(full_text.replace('Prompt:', '').split('Story:')[0].strip() + ' Continue the story:')
        else:
            print(f"Could not determine dataset for the entry: {full_text}")

    # Save the prompts to a new CSV file
    df_prompts = pd.DataFrame(prompts, columns=['Prompt'])
    df_prompts.to_csv('prompts.csv', index=False)
    print(f"Prompts extracted and saved to 'prompts.csv' with {len(df_prompts)} entries.")


def append_to_source_and_save(preprocessed_csv_path, source_csv_path='combined_source_data.csv',
                              output_csv_path='llm_and_human_data.csv'):
    """
    Appends a preprocessed CSV file to a source CSV file and saves the result in a new CSV file.

    Args:
        preprocessed_csv_path (str): Path to the preprocessed CSV file to be appended.
        source_csv_path (str, optional): Path to the source CSV file. Defaults to 'combined_source_data.csv'.
        output_csv_path (str, optional): Path to the CSV file where the result will be saved.
                                         Defaults to 'llm_and_human_data.csv'.

    Returns:
        None, generates a csv file with the combined data.
    """
    # Load the source data
    df_source = pd.read_csv(source_csv_path)

    # Load the preprocessed data
    df_preprocessed = pd.read_csv(preprocessed_csv_path)

    # Append the preprocessed data to the source data
    df_combined = pd.concat([df_source, df_preprocessed], ignore_index=True)

    # Save the combined data to a CSV file
    df_combined.to_csv(output_csv_path, index=False)


# append_to_source_and_save("t1_preprocessed.csv")