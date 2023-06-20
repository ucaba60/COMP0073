import openai
import csv
import pandas as pd

# OpenAI API key
openai.api_key = 'YOUR_OPEN_AI_KEY'


def gpt3_chat_prompts():
    # Load prompts from CSV file
    df = pd.read_csv('prompts.csv')
    prompts = df['text'].tolist()

    # Prepare data for new CSV file
    data = []

    # For each prompt, send it to the GPT-3 model and get a response
    for prompt in prompts:
        message = {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=message)
        data.append((prompt, response['choices'][0]['message']['content']))

    # Save responses to a new CSV file
    with open('responses.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Prompt", "Response"])
        writer.writerows(data)


gpt3_chat_prompts()