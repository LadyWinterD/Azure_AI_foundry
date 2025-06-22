# 🏠 Pic2Plot - Smart Floor Plan & Room Analyzer

## Overview

Pic2Plot is a Python application that uses AI to generate floor plans, real estate descriptions, and health recommendations from room photos or text descriptions. It's built using Chainlit for the user interface, MCP for handling AI tasks, and various libraries for image processing and AI model interaction.

## Features

-   **Images to Floor Plan**: Upload room images to automatically create a detailed floor plan.
-   **Text to Floor Plan**: Provide a text description of a space, and the AI will turn it into a floor plan.
-   **Images to Real Estate Description**: Get professional real estate listing descriptions from uploaded room images.
-   **Health Recommendations from Room Images**: Receive personalized suggestions to improve your room's health, lighting, or ergonomics.
    

## Problem Statement

Current floorplan generation solutions often suffer from:

* High Latency
* High Cost

## How This Implementation Addresses the Problem

Pic2Plot addresses these problems through:

* **Efficiency**:
    * FastMCP framework for optimized communication.
    * `asyncio` for concurrent operations.
* **Cost Optimization & Adaptive LLM Usage**: 

This project strategically employs various language and vision models, focusing on optimizing both performance and cost-efficiency. Our key optimization strategies include:

* **Cost-Effective Models for Routine Tasks:** Utilizing models like Phi-3 for simple text parsing ensures speed and low cost for high-frequency operations.
* **Modular Activation:** Expensive models like GPT-4.1 are invoked only when necessary, such as for real estate description and floorplan generation.
* **Embedding-Based Matching:** Employing cosine similarity for tasks like floorplan from text by matching improves accuracy while reducing reliance on costly live API calls.
* **Balanced Vision-Language Models:** Choosing models like LLaMA 3 Vision provides strong vision and language capabilities at a more favorable cost and latency than alternatives like GPT-4.

Here's a summary of the model usage for each feature:

| Use Case                       | Model(s)                      | Frequency | Cost Level |
|--------------------------------|-------------------------------|-----------|------------|
| Simple text parsing            | Phi-3                         | High      | 💲         |
| Floorplan description/image gen.| GPT-4.1 + GPT-4.1-Mini + Dalle     | Medium    | 💲💲💲        |
| Health tips from room image    | LLaMA 3 Vision                | Medium    | 💲         |
| Real estate description        | GPT-4.1                       | Low       | 💲💲       |
| Layout matching                | Phi-3 + CosSim               | High      | 💲💲         |


* **Unique Feature**: Health recommendations from room images.


## Architecture

![Description](https://github.com/GayathriChilukala/Final_Ai_Agent/blob/main/Final_arch.gif)


## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    
3.  **Set up Azure credentials:**
    * You'll need an Azure account and API keys for the Azure AI services used in `server.py`.  Replace the placeholder values (e.g., `your_githubtoken`, `your_token`) in `server.py` with your actual credentials.
    * If using ngrok, replace `your_auth_token` in `run.py` with your ngrok auth token.

4.  **Run the application:**
    **Important:** Start the server first, then the client.
    ```bash
    python server.py
    python run.py
    ```
    This will start the MCP server, and then the Chainlit application.  If configured, the application will be accessible via the ngrok URL.
    

## Project Tech Stack

### Core Technologies

* **Programming Language:** Python
* **Web Framework:** Chainlit (for the user interface)
* **MCP:** A custom framework (FastMCP) for defining and running tools
  

### Key Components & Libraries

* **Azure OpenAI:** Used for AI model interactions, specifically the Chat Completions API.
* **Models:**
    * GPT-4 (for image analysis, description, and floorplan generation)
    * Phi-3 (for processing user input and extracting floorplan information)
    * Llama 3 (for generating health tips from images)
    * DALL-E 3 (for generating floor plan images)
* **Image Processing:** PIL (Python Imaging Library)
* **Data Handling:** Pandas
* **Sentence Transformers:** For calculating embedding similarities.
* **Networking:** `requests`
* **Other:** `asyncio`, `base64`


### Infrastructure

* **Azure Machine Learning (AML):** The project is developed and run on Azure Machine Learning.
  

## Configuration

* `run.py`:  Configures the ngrok tunnel (port, auth token) and starts the Chainlit application.
* `server.py`:  Contains the server-side logic, including API keys and model endpoints for Azure AI.
* `client.py`:  Sets up the Chainlit client and defines how user input is processed and sent to the server.

## UI Output Screenshots

### Front Page Image
![Front Page Image](https://github.com/GayathriChilukala/Final_Ai_Agent/blob/main/UI_Frontpage.jpg)

### UI Images to Floor Plan

![UI Use Case 1 Image](https://github.com/GayathriChilukala/Final_Ai_Agent/blob/main/usecase1op.jpg)


### UI Text to Floor Plan

![UI Use Case 2 Image](https://github.com/GayathriChilukala/Final_Ai_Agent/blob/main/usecase2op.jpg)


### UI Health Recommendations from Room Images

![UI Use Case 3 Image](https://github.com/GayathriChilukala/Final_Ai_Agent/blob/main/usecase3op.jpg)


### UI Images to Real Estate Description 

![UI Use Case 4 Image](https://github.com/GayathriChilukala/Final_Ai_Agent/blob/main/usecase4op.jpg)


## Team Members

- **Danny Favela** — dfavela@alumni.usc.edu
- **Dongdong Li** — dongdong@outlook.co.nz
- **Gayathri Chilukala** — gchilukala2023@fau.edu
- **Tiffany Siman** — tiffanysiman@gmail.com

## 🚀 Future Plans

This section outlines our upcoming development efforts aimed at enhancing our platform's capabilities.

**Azure AI Search Integration:**

To further enhance our search capabilities, we plan to integrate [Azure AI Search](https://azure.microsoft.com/en-us/products/cognitive-services/azure-ai-search/) for managing and retrieving floor plans and embeddings. This powerful tool will enable us to efficiently index room images and textual data, ensuring faster and more relevant search results.

**Key benefits of this integration include:**

* **Efficient Indexing:** Seamlessly index and manage a large volume of floor plan images and associated textual information.
* **Faster Search Results:** Leverage Azure AI Search's advanced indexing and querying capabilities for quicker retrieval of relevant data.
* **Improved Relevance:** Utilize vector search and semantic ranking to provide more contextually accurate search results based on user queries.

**Spatial Language Models (Spatial LM):**

We aim to integrate **Spatial Language Models (Spatial LM)** for smarter, more context-aware spatial understanding and more advanced floor plan generation, as demonstrated in [Here](spatiallm_demo/README.md). By leveraging the power of these models, our platform will gain a deeper understanding of spatial relationships and user intent.

**This integration will enable:**

* **Smarter Spatial Understanding:** Enhanced ability to interpret and reason about spatial configurations within floor plans.
* **Context-Aware Generation:** Generation of more accurate and contextually relevant floor plans based on user input and spatial understanding.
* **Advanced Features:** Potential for new features leveraging spatial reasoning, such as intelligent layout suggestions and spatial query understanding.

We are excited about these future developments and believe they will significantly enhance the user experience and capabilities of our platform. Stay tuned for updates!


**🧠 Responsible AI Practices**

* **Transparency and Fair Usage:** We utilize publicly available datasets sourced from Kaggle and open-source models primarily from platforms like Hugging Face and GitHub. This ensures transparency in our data origins and model provenance, and we are committed to fair usage in accordance with their respective terms.



**Real-time Use Cases:** Imagine websites like Zillow, Redfin, and assisted living platforms integrating Pic2Plot. Users could upload photos of rooms in a listed property or their living space, and our AI could **instantly generate a basic floor plan sketch and a compelling real estate or design description, alongside initial health and safety insights (e.g., identifying potential fall hazards, accessibility concerns, or suggesting better lighting for visual comfort).** This rapid analysis would provide potential buyers, renters, or healthcare providers with a quicker and more comprehensive understanding of the property's layout and its implications for well-being, directly from the images. This capability could significantly enhance user engagement, streamline property exploration and assessment, and offer valuable preliminary health-related information across numerous web platforms.



**🙏 Acknowledgements**

We would like to extend our sincere gratitude to **Microsoft** for organizing and conducting this insightful hackathon. Your support and platform have been invaluable in fostering innovation and allowing us to explore these exciting possibilities.

A special thank you to **Pamela Fox** for her engaging and informative lectures. Her expertise and guidance have been truly inspiring and have significantly contributed to our understanding and progress.

Finally, we would also like to thank all the other organizers, mentors, and participants who contributed to making this hackathon a valuable and enriching experience. Your collective efforts and enthusiasm are greatly appreciated.

