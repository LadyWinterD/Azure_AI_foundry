import math
from mcp.server.fastmcp import FastMCP
from PIL import Image
import io
import base64
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, TextContentItem, ImageContentItem, ImageUrl, ImageDetailLevel
from azure.core.credentials import AzureKeyCredential
import chainlit as cl
import mimetypes
import os
import requests
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging


mcp = FastMCP("Pic2plot")

# Function to analyze and summarize the room descriptions
def analyze_room_layout(descriptions: str, endpoint: str, model_name: str, token: str):
    # Initialize the Azure client
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )

    # Structured prompt for generating the summary
    prompt = f"""
You are an expert architectural assistant.

Your task is to analyze the room descriptions provided and generate a single, clean paragraph summarizing the apartment or house layout.

Follow these rules:
- Mention the number of bedrooms and bathrooms using the BHK format (e.g., 1BHK, 2BHK).
- Mention the bathroom in the first line along with the kitchen ,living room and bedroom.
- Identify and briefly describe each room (e.g., Kitchen, Living Room, Bedroom, Bathroom, etc.).
- Explain how the rooms are connected (e.g., "The kitchen opens into the living room").
- Use a professional, architectural tone.
- Output should be a single, well-structured paragraph.

Room Descriptions:
{descriptions}

### Examples of output format

Example 1:
This is a 1BHK apartment with an attached bathroom. The entry opens into a well-lit living room that connects directly to a balcony. To the side is a compact L-shaped kitchen with bar seating. The bedroom, accessible via a short hallway off the living room, includes a sliding wardrobe and large window. The en-suite bathroom features a modern shower cabin and vanity. The layout prioritizes light and privacy, with all spaces efficiently interconnected.

Example 2:
This is a 2BHK layout featuring two bedrooms, two bathrooms, a study, and an open kitchen-living concept. The entry leads into the living area, which flows into the kitchen and includes a side room that functions as a compact study. The master bedroom includes an attached bath and walk-in closet, while the second bedroom is located near the common bathroom. The layout is designed for comfort and functionality, ideal for small families or remote workers.

Example 3:
This 2BHK layout features a master bedroom with an en-suite bathroom and walk-in closet, and a guest bedroom with access to a separate guest bathroom. The spacious living room includes a fireplace and bookshelves, and is adjacent to the modern kitchen, which features granite countertops and a breakfast bar. The layout is ideal for those who value space and functionality, with a small home office for remote work or study.

Example 4:
This is a 2BHK apartment with a spacious open living area and adjacent dining area, ideal for entertaining. The kitchen is designed for functionality with an island and high-end appliances. The master bedroom is comfortably furnished with a queen-sized bed and large closet, and the bathrooms are well-equipped with modern fixtures. The layout offers a good balance of open space and privacy, making it suitable for couples or small families.
"""

    # Send the request to the model
    response = client.complete(
        messages=[
            SystemMessage("You are an expert architect."),
            UserMessage(prompt),
        ],
        temperature=1.0,
        top_p=1.0,
        max_tokens=1000,
        model=model_name
    )

    # Return the response message content (summary of the layout)
    return response.choices[0].message.content


# Function to generate a floor plan image using Azure OpenAI DALL·E 3 based on a text description.
def generate_floor_plan(user_description: str, api_key: str) -> Image:
    # Azure OpenAI DALL·E 3 API Configuration
    api_base_url = "https://gchil-m9nglnci-swedencentral.cognitiveservices.azure.com"
    api_endpoint = "/openai/deployments/dall-e-3/images/generations?api-version=2024-02-01"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Construct prompt from user input
    prompt = f"""Create a floor plan sketch of a house based on the following description: {user_description}.

"""

    # Prepare the request payload
    data = {
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
        "response_format": "url"
    }

    # Make the request
    response = requests.post(api_base_url + api_endpoint, headers=headers, json=data)

    # Handle response
    if response.status_code == 200:
        response_data = response.json()
        image_url = response_data['data'][0]['url']

        # Download and display image
        image = Image.open(BytesIO(requests.get(image_url).content))

        return image
    else:
        print("Error:", response.status_code, response.text)
        return None

# Function to find top 3 similar labels based on query
def find_top_similar_labels(query, embeddings_url="https://pic2plotrg2465003651.blob.core.windows.net/rag-data/embeddings.npy?your_auth_token"",
                            csv_url="https://pic2plotrg2465003651.blob.core.windows.net/rag-data/updated_file.csv?your_auth_token"):
    # Load the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load embeddings
    response = requests.get(embeddings_url)
    embeddings = np.load(BytesIO(response.content))

    # Load dataset
    df = pd.read_csv(csv_url)

    # Function to find top N similar labels
    def find_top_similar_labels(query, df, embeddings, top_n=3):
        labels = df["Property Label"].tolist()
        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, embeddings)[0]  # shape: (num_labels,)

        # Get indices of top N most similar
        top_indices = similarities.argsort()[-top_n:][::-1]

        top_results = []
        for idx in top_indices:
            label = labels[idx]
            image_url = df.loc[idx, "Image Path"]
            similarity_score = similarities[idx]
            top_results.append((label, image_url, similarity_score))

        return top_results

    # Find top similar labels
    top_results = find_top_similar_labels(query, df, embeddings)

    # Return top results
    return top_results

# Function to extract structured floor plan information (number of beds, baths, etc.) from a user query.
def get_floor_plan_info(user_query, model_name="microsoft/Phi-3-mini-4k-instruct", endpoint="https://models.github.ai/inference", token='your_githubtoken'):
    # Initialize the client
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )

    # Prompt for extracting structured floor plan details
    prompt = f"""
    Extract and return the structured floor plan format from the given text.
    Only include the fields Bed, Bath, Garage, and SQFT if they are explicitly mentioned.
    Return the result in this exact format:

    n_Bed_n_Bath_n_Garage_n_SQFT

    If any field is missing, do not include it in the output.
    Use capital letters as shown and separate each field with an underscore.

    Input: "3 beds and 2 baths"
    Output: 3_Bed_2_Bath

    Input: "4 bed, 3 bath, 2-car garage, 2100 sqft"
    Output: 4_Bed_3_Bath_2_Garage_2100_SQFT

    Input: "Only 2-car garage and 1800 square feet"
    Output: 2_Garage_1800_SQFT

    Now process this input:
    "{user_query}"
    """

    # Query the model
    response = client.complete(
        messages=[
            SystemMessage("You are a helpful assistant."),
            UserMessage(prompt),
        ],
        temperature=0.7,
        top_p=1.0,
        max_tokens=100,
        model=model_name
    )

    query = response.choices[0].message.content.strip()
    return query
    
# Function to encode an image to a base64 string, attempting to keep the size under a target limit.
def encode_image(image_path, target_kb=10):
    with Image.open(image_path) as img:
        img = img.convert("RGB")

        # Resize down aggressively (start at 128x128)
        img.thumbnail((128, 128))

        buffer = io.BytesIO()
        quality = 10  # Very low quality for maximum compression

        # Save as JPEG to buffer
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)

        # Check size
        size_kb = buffer.getbuffer().nbytes / 1024
        print(f"Compressed size: {size_kb:.2f} KB")

        # Encode to base64
        encoded = base64.b64encode(buffer.read()).decode("utf-8")
        return encoded  # Just return the base64 string

# Function to save a base64 string as an image file.
def save_base64_as_image(base64_string, output_path):
    # Decode the base64 string and save as an image
    img_data = base64.b64decode(base64_string)
    with open(output_path, "wb") as img_file:
        img_file.write(img_data)
    return output_path

# MCP tool to generate a real estate description from provided room images.
@mcp.tool()
def generate_real_estate_description_from_images(image_paths, TOKEN="your_githubtoken",prompt_text=None)->str:
    ENDPOINT = "https://models.github.ai/inference"
    MODEL = "openai/gpt-4.1"


    client = ChatCompletionsClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(TOKEN),
    )

    if prompt_text is None:
        prompt_text = (
        """Using the provided images of different rooms in the house, identify and describe each room (e.g., kitchen, living room, bedroom) highlighting their key features, style, and amenities. Then, craft a professional and engaging real estate listing that paints a complete picture of the home's layout, flow, and atmosphere to attract potential buyers. """
        )

    # Convert paths to image content items
    image_items = [
        ImageContentItem(image_url=ImageUrl(url=encode_image(image.path)))
        for image in image_paths
    ]

    messages = [
        SystemMessage("You are a helpful assistant that can identify rooms from images and generate a basic floorplan."),
        UserMessage(
            content=[
                TextContentItem(text=prompt_text),
                *image_items
            ]
        )
    ]

    response = client.complete(
        messages=messages,
        temperature=0.8,
        top_p=1.0,
        model=MODEL,
        max_tokens=2048
    )

    return response.choices[0].message.content


@mcp.tool()
# Main function to analyze images and generate a floorplan
def analyze_images_and_generate_floorplan(image_paths, TOKEN='your_githubtoken', prompt_text=None):
    ENDPOINT = "https://models.github.ai/inference"
    MODEL = "openai/gpt-4.1"

    client = ChatCompletionsClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(TOKEN),
    )

    if prompt_text is None:
        prompt_text = (
            "These are images of various rooms from a house. Identify each room "
            "(e.g., kitchen, living room, bedroom) and describe its features. "
            "Then, infer a possible layout and generate a textual floorplan of the house."
        )

    # Convert paths to image content items using proper base64 encoding
    image_items = [
        ImageContentItem(image_url=ImageUrl(url=encode_image(image.path)))
        for image in image_paths
    ]

    messages = [
        SystemMessage("You are a helpful assistant that can identify rooms from images and generate a basic floorplan."),
        UserMessage(
            content=[
                TextContentItem(text=prompt_text),
                *image_items
            ]
        )
    ]

    response = client.complete(
        messages=messages,
        temperature=0.8,
        top_p=1.0,
        model=MODEL,
        max_tokens=2048
    )

    r1=response.choices[0].message.content
    endpoint = "https://models.github.ai/inference"
    model_name = "openai/gpt-4.1-mini"
    token = "your_githubtoken"
    r2=analyze_room_layout(r1, endpoint, model_name,token)
    r3=image = generate_floor_plan(r2, 'your_token')
    return r3
        
@mcp.tool()
def get_function_name_from_user_input(user_input: str) -> str:
    # Azure model config
    endpoint1 = "https://models.github.ai/inference"
    model_name1 = "microsoft/Phi-3.5-mini-instruct"
    token1 = 'your_githubtoken'

    # Setup client inside the function
    client1 = ChatCompletionsClient(
        endpoint=endpoint1,
        credential=AzureKeyCredential(token1),
    )

    # Make request
    response1 = client1.complete(
        messages=[
            SystemMessage("""You are an intelligent assistant. Based on the user's question, choose the correct function to call from the following options:

- analyze_images_and_generate_floorplan — If the user asks to convert their images into a floor plan.
- generate_real_estate_description_from_images — If the user asks to write a real estate listing based on their pictures.
- generate_health_tips_from_image — If the user asks for health or wellness tips based on an image of a room.
- process_query — If the user gives a text description and asks you to generate a floor plan.

Reply ONLY with the function name (no explanation, no extra text).

Examples:

User: "Can you turn my house photos into a floor plan?" → analyze_images_and_generate_floorplan

User: "Suggest how to make my bedroom healthier." → generate_health_tips_from_image
"""),
            UserMessage(user_input),
        ],
        temperature=1.0,
        top_p=1.0,
        max_tokens=1000,
        model=model_name1
    )
    return response1.choices[0].message.content.strip()

@mcp.tool()
def generate_health_tips_from_image(image_paths) -> str:
    token = "your_githubtoken"
    endpoint = "https://models.github.ai/inference"
    model_name = "meta/Llama-3.2-90B-Vision-Instruct"

    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )

    # Decode and save the base64 image
    encoded_image = encode_image(image_paths[0])
    output_image_path = "/tmp/temp_image.jpg"
    with open(output_image_path, "wb") as f:
        f.write(decoded_bytes)
    saved_image_path = output_image_path

    response = client.complete(
        messages=[
            SystemMessage("You are a helpful assistant that describes images in detail."),
            UserMessage(
                content=[
                    TextContentItem(
                        text="Using the images of the room, provide personalized health and wellness tips in points. For example, suggest improvements like, 'You can add a window here for better natural light' or 'Consider placing your desk near a source of natural ventilation for a healthier work environment."
                    ),
                    ImageContentItem(
                        image_url=ImageUrl.load(
                            image_file=saved_image_path,
                            image_format="jpg",
                            detail=ImageDetailLevel.LOW
                        ),
                    ),
                ],
            ),
        ],
        model=model_name,
    )

    return response.choices[0].message.content



@mcp.tool()
def process_query(user_query:str)->list:
    # Step 1: Get the floor plan description from GPT model
    floor_plan_info = get_floor_plan_info(user_query)

    # Step 2: Find the top 3 similar floor plan labels
    top_results = find_top_similar_labels(floor_plan_info)
    image_urls=[]
    # Step 3: Print and display top 3 results
    for i, (label, image_path, score) in enumerate(top_results, 1):

        print(f"Match #{i}:")
        print(f"  Label: {label}")
        print(f"  Score: {score:.4f}")

        # Construct full image URL
        base_url = "https://pic2plotrg2465003651.blob.core.windows.net/imgaes-rag/"
        full_image_url = base_url + image_path + '?authtoken'
        full_image_url = full_image_url.replace('/images/', '/')
        print(f"  Image URL: {full_image_url}")
        image_urls.append(full_image_url)
        try:
            img_response = requests.get(full_image_url)
            img = Image.open(BytesIO(img_response.content))
            display(img)
        except Exception as e:
            print(f"  Could not display image #{i}: {e}")
    return image_urls


if __name__ == "__main__":
    print("🚀 Starting MCP Server...")
    mcp.run(transport="stdio")
