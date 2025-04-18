import chainlit as cl
from azure.ai.projects import AIProjectClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

project_connection_string = os.getenv("AZURE_API_KEY")
model_deployment = os.getenv("AZURE_MODEL_NAME")

# Initialize project client outside the handler
if not project_connection_string:
    raise ValueError("Missing AZURE_API_KEY in .env")
if not model_deployment:
    raise ValueError("Missing AZURE_MODEL_NAME in .env")

project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=project_connection_string,
)

openai_client = project_client.inference.get_chat_completions_client()

@cl.on_message
async def main(message: cl.Message):
    try:
        prompt = [
            SystemMessage("You are a helpful AI assistant that answers questions."),
            UserMessage(message.content)
        ]

        response = openai_client.complete(
            model=model_deployment,
            messages=prompt
        )

        reply = response.choices[0].message.content
        await cl.Message(content=reply).send()

    except Exception as ex:
        await cl.Message(content=f"Error: {str(ex)}").send()