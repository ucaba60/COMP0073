import pandas as pd
import openai
import time
import csv
import os

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Constants
BATCH_SIZE = 10  # Define the batch size
openai.api_key = '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'


def generate_gpt2_responses(prompt_csv_path, response_folder_path, model_name):
    """
    Generate responses for a list of prompts saved in a csv file using a GPT-2 model.

    Args:
        prompt_csv_path (str): Path to the csv file containing the prompts.
        response_folder_path (str): Path to the folder where the responses will be saved.
        model_name (str): Name of the GPT-2 model to use (for example, "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl").

    Returns:
        None, generates a csv file with the responses.
    """
    # Define acceptable models
    acceptable_models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

    if model_name not in acceptable_models:
        raise ValueError(f"Invalid model name. Acceptable models are: {', '.join(acceptable_models)}")

    # Load the GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Load the prompts
    df = pd.read_csv(prompt_csv_path)
    prompts = df['Prompt'].tolist()

    # Construct the response file path
    response_csv_path = os.path.join(response_folder_path, f"{model_name}_responses.csv")

    # Check if the response file already exists
    if os.path.exists(response_csv_path):
        # Load the existing responses
        existing_responses_df = pd.read_csv(response_csv_path)

        # Determine the starting point based on the number of existing responses
        start = len(existing_responses_df)
    else:
        start = 0

    for i in range(start, len(prompts)):
        # Encode the prompt
        input_ids = tokenizer.encode(prompts[i], return_tensors="pt")

        # Generate a response
        output = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),  # Set all positions to 1 (i.e., no padding)
            pad_token_id=tokenizer.eos_token_id,  # Use the EOS token as the PAD token
            do_sample=True,
            max_length=1024,  # Use GPT-2's maximum sequence length
        )

        # Calculate the number of tokens in the prompt
        prompt_length = input_ids.shape[-1]

        # Decode only the response, excluding the prompt
        response = tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)

        # Save the prompt and response to a DataFrame
        response_df = pd.DataFrame({
            'Prompt': [prompts[i]],
            'Response': [response]
        })

        # Append the DataFrame to the CSV file
        if os.path.exists(response_csv_path):
            response_df.to_csv(response_csv_path, mode='a', header=False, index=False)
        else:
            response_df.to_csv(response_csv_path, mode='w', index=False)

        print(f"Prompt {i + 1} of {len(prompts)} processed")

    print(f"All prompts processed. Responses saved to {response_csv_path}.")


# generate_gpt2_responses("extracted_data/prompts.csv", "extracted_data")


def generate_gpt_responses(prompt_csv_path, response_folder_path, model="gpt-3.5-turbo", temperature=1):
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
                temperature=temperature,  # as per OpenAI Documentation default temperature value is 1,
                # which is the case here as well, this line is intended for future developments.
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


# ------------------------------------------------------------------------------------------#

# generate_gpt_responses('extracted_data/prompts.csv', 'extracted_data', temperature=1)

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
                'Summary: ' + row['Prompt'].replace('Write a news article based on the following summary: ',
                                                    '') + ' Article: ' + row['Response']
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

# extract_and_combine('extracted_data/gpt2_t1_responses.csv')

# ------------------------------------------------------------------------------------------#

def regenerate_responses(response_csv_path):
    """
    Check the csv file containing generated responses for any NaN values.
    If any are found, regenerate the responses using the provided model.

    Args:
        response_csv_path (str): Path to the csv file containing the generated responses.

    Returns:
        None, updates the csv file with the regenerated responses.
    """
    # Extract the model name from the filename
    model_name = os.path.basename(response_csv_path).split('_')[0]

    # Load the model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    print(f"Loaded model {model_name}")

    # Load the responses
    df = pd.read_csv(response_csv_path)

    # Iterate over the DataFrame
    for i, row in df.iterrows():
        if pd.isnull(row['Response']):
            # Encode the prompt
            input_ids = tokenizer.encode(row['Prompt'], return_tensors="pt")

            # Generate a response
            output = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),  # Set all positions to 1 (i.e., no padding)
                pad_token_id=tokenizer.eos_token_id,  # Use the EOS token as the PAD token
                do_sample=True,
                max_length=1024,  # Use GPT-2's maximum sequence length
            )

            # Calculate the number of tokens in the prompt
            prompt_length = input_ids.shape[-1]

            # Decode only the response, excluding the prompt
            response = tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)

            # Replace the NaN response with the new one
            df.at[i, 'Response'] = response

            # Save the DataFrame back to the CSV file
            df.to_csv(response_csv_path, index=False)

            print(f"Regenerated response for prompt {i + 1} of {len(df)}. Updated responses saved to {response_csv_path}.")

    print(f"All NaN responses regenerated. Updated responses saved to {response_csv_path}.")


regenerate_responses('extracted_data/gpt2-large_responses.csv')
#
# df = pd.read_csv("extracted_data/gpt2_t1_responses.csv")
# nan_rows = df[df.isna().any(axis=1)]
# print(nan_rows)


def extract_prompts_and_save(file_folder_path):
    """
    Extracts prompts from the combined dataset and saves them to a .csv file.

    Args:
        file_folder_path (str): The path to the folder where the combined_source_data.csv file is located.

    Returns:
        None, saves the prompts to a .csv file.
    """
    # Load the combined dataset
    combined_data_file = os.path.join(file_folder_path, 'combined_source_data.csv')
    df = pd.read_csv(combined_data_file)
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
    df_prompts.to_csv(os.path.join(file_folder_path, 'prompts.csv'), index=False)
    print(f"Prompts extracted and saved to '{os.path.join(file_folder_path, 'prompts.csv')}' with {len(df_prompts)}"
          f" entries.")

# extract_prompts_and_save("extracted_data")
