import asyncio
import base64
import io
import json
import requests
from collections import deque
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from PIL import Image
import chainlit as cl

# Global variable to store loaded tools from the MCP server. Initialized as empty.
global_tools = []

# Parameters for connecting to the MCP server via standard input/output.
server_params = StdioServerParameters(
    command="python",  # The command to execute the MCP server.
    args=["server.py"]  # Arguments passed to the server command.
)

# Your Azure Blob Storage SAS URL for storing chat history as a JSON file.
# Replace "YOUR_SAS_URL_HERE" with your actual SAS URL.
sas_url = "YOUR_SAS_URL_HERE"

async def encode_chainlit_file_to_base64(file: cl.File) -> str:
    """
    Asynchronously reads the content of a Chainlit file and encodes it into a base64 string.
    This is useful for sending image data over network requests.
    """
    content = await file.read()
    return base64.b64encode(content).decode("utf-8")

async def update_history(usecase: str, result):
    """
    Updates the chat history stored as a JSON file in Azure Blob Storage.
    It fetches the existing history, adds the new interaction, and uploads the updated history.
    Handles different types of results (text, image URLs).
    Uses a deque to maintain a fixed maximum length for the history.
    """
    new_record = {"usecase": usecase, "result": result}
    try:
        # Fetch existing history from Azure Blob Storage.
        response = requests.get(sas_url)
        response.raise_for_status()  # Raise an exception for HTTP errors.

        try:
            # Attempt to parse the existing content as JSON.
            existing_data = response.json()
            # Ensure the existing data is a list for consistent handling.
            if not isinstance(existing_data, list):
                print("Existing JSON is not a list. Fixing...")
                existing_data = [existing_data]
        except json.JSONDecodeError:
            # If the JSON is empty or invalid, start with an empty list.
            print("JSON is empty or invalid. Starting fresh...")
            existing_data = []

        # Use a deque with a maximum length of 10 to manage history size.
        history = deque(existing_data, maxlen=10)
        history.appendleft(new_record)  # Add the new record to the beginning of the deque.

        # Convert the deque back to a list and serialize it to JSON with indentation for readability.
        updated_json_data = json.dumps(list(history), indent=4)

        # Upload the updated JSON data to Azure Blob Storage.
        headers = {"x-ms-blob-type": "BlockBlob", "Content-Type": "application/json"}
        upload_response = requests.put(sas_url, headers=headers, data=updated_json_data)
        upload_response.raise_for_status()

        print("Updated JSON successfully uploaded!")

    except requests.exceptions.RequestException as e:
        print(f"Error updating history: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding existing JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


@cl.on_chat_start
async def on_chat_start():
    """
    Event handler that is called when a new chat session begins.
    It sends a welcome message to the user with instructions and capabilities.
    """
    await cl.Message(
        content="""

# 🏠 Pic2Plot - Smart Floor Plan & Room Analyzer

Welcome! Upload room photos or provide a text description — I'll generate floor plans, real estate descriptions, or even health tips based on your rooms!

## What you can do:
- 🖼️ **Images to Floor Plan**: Upload room images to automatically create a detailed floor plan.
- ✍️ **Text to Floor Plan**: Provide a text description of a space, and I'll turn it into a floor plan.
- 🏡 **Images to Real Estate Description**: Get professional real estate listing descriptions from your uploaded room images.
- 💡 **Health Recommendations from Room Images**: Receive personalized suggestions to improve your room's health, lighting, or ergonomics.

## How to use:
1. Upload room images **or** type a description.
2. Wait a moment while the AI processes your input.
3. Get your generated floor plan, description, or health tips!

Let's get started!


        """
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Event handler that is called when the user sends a new message.
    It processes the user's input (text or images) and interacts with the MCP server
    to perform the requested action.
    """
    global global_tools

    # Extract user's text input, removing leading/trailing whitespace.
    user_text = message.content.strip() if message.content else ""
    # Filter out image files from the message elements.
    image_files = [file for file in message.elements if "image" in file.mime] if message.elements else []

    # Prepare a list to store base64 encoded image data.
    encoded_images = []
    # Encode each uploaded image file to base64.
    if image_files:
        for img in image_files:
            encoded = await encode_chainlit_file_to_base64(img)
            encoded_images.append({
                "data": encoded,
                "mime": img.mime
            })

    # Initialize the tools from the MCP server if they haven't been loaded yet.
    if not global_tools:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                # Load tools exposed by the MCP server using the Langchain MCP adapter.
                global_tools = await load_mcp_tools(session)
                print(f"✅ Loaded Tools: {[tool.name for tool in global_tools]}")

    # Handle the case where tools could not be loaded.
    if not global_tools:
        await cl.Message(content="❌ Error: Tools could not be loaded.").send()
        return

    # Find the tool responsible for determining the next function to call based on user input.
    user_func = next((tool for tool in global_tools if tool.name == "get_function_name_from_user_input"), None)

    # Handle the case where the function name retrieval tool is not found.
    if not user_func:
        await cl.Message(content="❌ Error: Tool 'get_function_name_from_user_input' not found.").send()
        return

    try:
        # Invoke the function to get the action (tool name) to perform based on user input.
        action = await user_func.ainvoke({"user_input": user_text})
        await cl.Message(content="Loading...⏳").send()  # Send a loading message to the user.

        # Execute the appropriate tool based on the determined action.
        if action == "generate_health_tips_from_image":
            tool = next(tool for tool in global_tools if tool.name == "generate_health_tips_from_image")
            result = await tool.ainvoke({"image_paths": encoded_images})
            await cl.Message(content=result).send()
            await update_history("generate_health_tips_from_image", result)

        elif action == "generate_real_estate_description_from_images":
            tool = next(tool for tool in global_tools if tool.name == "generate_real_estate_description_from_images")
            result = await tool.ainvoke({"image_paths": encoded_images})
            await cl.Message(content=result).send()
            await update_history("generate_real_estate_description_from_images", result)

        elif action == "analyze_images_and_generate_floorplan":
            tool = next(tool for tool in global_tools if tool.name == "analyze_images_and_generate_floorplan")
            floorplan_image = await tool.ainvoke({"image_paths": encoded_images})
            await cl.Message(
                content=f"🛠️ Generated floorplan based on your request: '{user_text}'",
                elements=[cl.Image(name="Floorplan", display="inline", url=floorplan_image)]
            ).send()
            await update_history("analyze_images_and_generate_floorplan", floorplan_image)

        else:
            # Default case: process the user query to generate floor plans from text.
            tool = next(tool for tool in global_tools if tool.name == "process_query")
            image_urls = await tool.ainvoke({"user_query": user_text})
            elements = [cl.Image(url=url, display="inline") for url in image_urls]
            await cl.Message(
                content="📋 Here are some floor plans based on your description!",
                elements=elements
            ).send()
            await update_history("text_to_floorplan", image_urls)  # Pass the list of URLs directly

    except Exception as e:
        # Handle any exceptions that occur during the processing of the user message.
        print(f"Error: {e}")
        await cl.Message(content="❗ Sorry, something went wrong processing your request.").send()import asyncio
import base64
import io
import json
import requests
from collections import deque
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from PIL import Image
import chainlit as cl

# Global variable to store loaded tools from the MCP server. Initialized as empty.
global_tools = []

# Parameters for connecting to the MCP server via standard input/output.
server_params = StdioServerParameters(
    command="python",  # The command to execute the MCP server.
    args=["server.py"]  # Arguments passed to the server command.
)

# Your Azure Blob Storage SAS URL for storing chat history as a JSON file.
# Replace "YOUR_SAS_URL_HERE" with your actual SAS URL.
sas_url = "YOUR_SAS_URL_HERE"

async def encode_chainlit_file_to_base64(file: cl.File) -> str:
    """
    Asynchronously reads the content of a Chainlit file and encodes it into a base64 string.
    This is useful for sending image data over network requests.
    """
    content = await file.read()
    return base64.b64encode(content).decode("utf-8")

async def update_history(usecase: str, result):
    """
    Updates the chat history stored as a JSON file in Azure Blob Storage.
    It fetches the existing history, adds the new interaction, and uploads the updated history.
    Handles different types of results (text, image URLs).
    Uses a deque to maintain a fixed maximum length for the history.
    """
    new_record = {"usecase": usecase, "result": result}
    try:
        # Fetch existing history from Azure Blob Storage.
        response = requests.get(sas_url)
        response.raise_for_status()  # Raise an exception for HTTP errors.

        try:
            # Attempt to parse the existing content as JSON.
            existing_data = response.json()
            # Ensure the existing data is a list for consistent handling.
            if not isinstance(existing_data, list):
                print("Existing JSON is not a list. Fixing...")
                existing_data = [existing_data]
        except json.JSONDecodeError:
            # If the JSON is empty or invalid, start with an empty list.
            print("JSON is empty or invalid. Starting fresh...")
            existing_data = []

        # Use a deque with a maximum length of 10 to manage history size.
        history = deque(existing_data, maxlen=10)
        history.appendleft(new_record)  # Add the new record to the beginning of the deque.

        # Convert the deque back to a list and serialize it to JSON with indentation for readability.
        updated_json_data = json.dumps(list(history), indent=4)

        # Upload the updated JSON data to Azure Blob Storage.
        headers = {"x-ms-blob-type": "BlockBlob", "Content-Type": "application/json"}
        upload_response = requests.put(sas_url, headers=headers, data=updated_json_data)
        upload_response.raise_for_status()

        print("Updated JSON successfully uploaded!")

    except requests.exceptions.RequestException as e:
        print(f"Error updating history: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding existing JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


@cl.on_chat_start
async def on_chat_start():
    """
    Event handler that is called when a new chat session begins.
    It sends a welcome message to the user with instructions and capabilities.
    """
    await cl.Message(
        content="""

# 🏠 Pic2Plot - Smart Floor Plan & Room Analyzer

Welcome! Upload room photos or provide a text description — I'll generate floor plans, real estate descriptions, or even health tips based on your rooms!

## What you can do:
- 🖼️ **Images to Floor Plan**: Upload room images to automatically create a detailed floor plan.
- ✍️ **Text to Floor Plan**: Provide a text description of a space, and I'll turn it into a floor plan.
- 🏡 **Images to Real Estate Description**: Get professional real estate listing descriptions from your uploaded room images.
- 💡 **Health Recommendations from Room Images**: Receive personalized suggestions to improve your room's health, lighting, or ergonomics.

## How to use:
1. Upload room images **or** type a description.
2. Wait a moment while the AI processes your input.
3. Get your generated floor plan, description, or health tips!

Let's get started!


        """
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Event handler that is called when the user sends a new message.
    It processes the user's input (text or images) and interacts with the MCP server
    to perform the requested action.
    """
    global global_tools

    # Extract user's text input, removing leading/trailing whitespace.
    user_text = message.content.strip() if message.content else ""
    # Filter out image files from the message elements.
    image_files = [file for file in message.elements if "image" in file.mime] if message.elements else []

    # Prepare a list to store base64 encoded image data.
    encoded_images = []
    # Encode each uploaded image file to base64.
    if image_files:
        for img in image_files:
            encoded = await encode_chainlit_file_to_base64(img)
            encoded_images.append({
                "data": encoded,
                "mime": img.mime
            })

    # Initialize the tools from the MCP server if they haven't been loaded yet.
    if not global_tools:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                # Load tools exposed by the MCP server using the Langchain MCP adapter.
                global_tools = await load_mcp_tools(session)
                print(f"✅ Loaded Tools: {[tool.name for tool in global_tools]}")

    # Handle the case where tools could not be loaded.
    if not global_tools:
        await cl.Message(content="❌ Error: Tools could not be loaded.").send()
        return

    # Find the tool responsible for determining the next function to call based on user input.
    user_func = next((tool for tool in global_tools if tool.name == "get_function_name_from_user_input"), None)

    # Handle the case where the function name retrieval tool is not found.
    if not user_func:
        await cl.Message(content="❌ Error: Tool 'get_function_name_from_user_input' not found.").send()
        return

    try:
        # Invoke the function to get the action (tool name) to perform based on user input.
        action = await user_func.ainvoke({"user_input": user_text})
        await cl.Message(content="Loading...⏳").send()  # Send a loading message to the user.

        # Execute the appropriate tool based on the determined action.
        if action == "generate_health_tips_from_image":
            tool = next(tool for tool in global_tools if tool.name == "generate_health_tips_from_image")
            result = await tool.ainvoke({"image_paths": encoded_images})
            await cl.Message(content=result).send()
            await update_history("generate_health_tips_from_image", result)

        elif action == "generate_real_estate_description_from_images":
            tool = next(tool for tool in global_tools if tool.name == "generate_real_estate_description_from_images")
            result = await tool.ainvoke({"image_paths": encoded_images})
            await cl.Message(content=result).send()
            await update_history("generate_real_estate_description_from_images", result)

        elif action == "analyze_images_and_generate_floorplan":
            tool = next(tool for tool in global_tools if tool.name == "analyze_images_and_generate_floorplan")
            floorplan_image = await tool.ainvoke({"image_paths": encoded_images})
            await cl.Message(
                content=f"🛠️ Generated floorplan based on your request: '{user_text}'",
                elements=[cl.Image(name="Floorplan", display="inline", url=floorplan_image)]
            ).send()
            await update_history("analyze_images_and_generate_floorplan", floorplan_image)

        else:
            # Default case: process the user query to generate floor plans from text.
            tool = next(tool for tool in global_tools if tool.name == "process_query")
            image_urls = await tool.ainvoke({"user_query": user_text})
            elements = [cl.Image(url=url, display="inline") for url in image_urls]
            await cl.Message(
                content="📋 Here are some floor plans based on your description!",
                elements=elements
            ).send()
            await update_history("text_to_floorplan", image_urls)  # Pass the list of URLs directly

    except Exception as e:
        # Handle any exceptions that occur during the processing of the user message.
        print(f"Error: {e}")
        await cl.Message(content="❗ Sorry, something went wrong processing your request.").send()
