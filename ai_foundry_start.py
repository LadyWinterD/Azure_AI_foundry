from azure.ai.projects import AIProjectClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

project_connection_string = os.getenv("AZURE_API_KEY")
model_deployment = os.getenv("AZURE_MODEL_NAME")

try:
    if not project_connection_string:
        raise ValueError("Missing AZURE_API_KEY in .env")
    if not model_deployment:
        raise ValueError("Missing AZURE_MODEL_NAME in .env")

    # Create project client
    project_client = AIProjectClient.from_connection_string(
        credential=DefaultAzureCredential(),
        conn_str=project_connection_string,
    )

    # Get OpenAI chat client
    openai_client = project_client.inference.get_chat_completions_client()

    # Get user input
    input_text = input("Enter a question: ")

    # Create chat history
    prompt = [
        SystemMessage("You are a helpful AI assistant that answers questions."),
        UserMessage(input_text)
    ]

    # Call chat model
    response = openai_client.complete(
        model=model_deployment,
        messages=prompt
    )

    # Get and display result
    completion = response.choices[0].message.content
    print(completion)

    # Optionally track the assistant message
    prompt.append(AssistantMessage(completion))

except Exception as ex:
    print("Error:", ex)