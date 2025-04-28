import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import chainlit as cl

load_dotenv()

subscription_key = os.getenv("AZURE_API_KEY")
endpoint = os.getenv("AZURE_ENDPOINT")
model_name = "gpt-4o-mini"
deployment = "gpt-4o-mini"
api_version = "2025-01-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)
cl.config.title = "AI Assistant"
cl.config.description = "Welcome to your AI assistant powered by Azure OpenAI!"

@cl.on_message
async def main(message: cl.Message):
    try:
      
        prompt = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": message.content,
            },
            {
                "role": "assistant",
                "content": "Sure! How can I assist you today?",
            }
        ]


        response = client.chat.completions.create(
            messages=prompt,
            max_tokens=4096,
            temperature=1.0,
            top_p=1.0,
            model=model_name
        )


        reply = response.choices[0].message.content
        await cl.Message(content=reply).send()

    except Exception as ex:
        # Handle the error and send a message to the user
        await cl.Message(content=f"Error: {str(ex)}").send()