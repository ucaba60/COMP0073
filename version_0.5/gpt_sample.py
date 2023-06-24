import pandas as pd
import openai
import time
import csv
import os
import argparse

#Constants
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

        # Write the DataFrame to the CSV file, appending if it already exists
        if os.path.exists(response_csv_path):
            response_df.to_csv(response_csv_path, mode='a', header=False, index=False)
        else:
            response_df.to_csv(response_csv_path, mode='w', index=False)

        print(f"Batch {i // BATCH_SIZE + 1} completed")


# ------------------------------------------------------------------------------------------#

# generate_gpt_responses("Labelled_Data/prompts.csv", "Labelled_Data/t1_responses.csv")

# ------------------------------------------------------------------------------------------#


def check_lengths(prompt_csv_path, response_csv_path):
    prompts_df = pd.read_csv(prompt_csv_path)
    responses_df = pd.read_csv(response_csv_path)

    if len(prompts_df) == len(responses_df):
        print("Both files have the same length!")
    else:
        print(f"Length of prompts file: {len(prompts_df)}")
        print(f"Length of responses file: {len(responses_df)}")


# check_lengths("Labelled_Data/prompts.csv", "Labelled_Data/t1_responses.csv")


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


def main():
    parser = argparse.ArgumentParser(description='Script for generating GPT responses and preprocessing them.')

    parser.add_argument('--generate', action='store_true', help='Generate GPT responses.')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the generated responses.')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='The temperature hyperparameter for GPT generation.')

    args = parser.parse_args()

    if args.generate:
        output_filename = f"Labelled_Data/t{args.temperature}_responses.csv"
        if os.path.exists(output_filename):
            overwrite = input(f"File {output_filename} already exists. Do you want to overwrite it? (y/n): ")
            if overwrite.lower() != 'y':
                print("Aborting generation process.")
                return
        generate_gpt_responses("Labelled_Data/prompts_combined.csv", output_filename, temperature=args.temperature)
        print(f"GPT response generation complete. The data has been saved in {output_filename}")

    if args.preprocess:
        input_filename = f"Labelled_Data/t{args.temperature}_responses.csv"
        output_filename = f"Labelled_Data/t{args.temperature}_preprocessed.csv"
        if not os.path.exists(input_filename):
            print(f"File {input_filename} does not exist. Please generate responses first.")
            return
        if os.path.exists(output_filename):
            overwrite = input(f"File {output_filename} already exists. Do you want to overwrite it? (y/n): ")
            if overwrite.lower() != 'y':
                print("Aborting preprocessing process.")
                return
        extract_and_combine(input_filename, output_filename)
        print(f"Data preprocessing complete. The preprocessed data has been saved in {output_filename}")


if __name__ == '__main__':
    main()