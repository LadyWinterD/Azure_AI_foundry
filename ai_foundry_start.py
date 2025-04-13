from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key (connection string) from the environment variable
project_connection_string = os.getenv("AZURE_API_KEY")

# Check if the API key is loaded correctly
if project_connection_string is None:
    raise ValueError("API key not found. Please ensure it is set in the .env file.")

# Initialize the project client with the API key
project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=project_connection_string,
)
