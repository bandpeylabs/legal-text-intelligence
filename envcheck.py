import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("Loaded Endpoint:", os.getenv("ENDPOINT"))
print("Loaded Subscription Key:", os.getenv("SUBSCRIPTION_KEY"))

# Get the values from environment variables
endpoint = os.getenv("ENDPOINT")
model_name = os.getenv("MODEL_NAME")
deployment = os.getenv("DEPLOYMENT")
subscription_key = os.getenv("SUBSCRIPTION_KEY")
api_version = os.getenv("API_VERSION")

# Check if the environment variables are being loaded
if not endpoint or not subscription_key:
    raise ValueError("Endpoint or Subscription Key is missing from the .env file")

# Initialize the client with correct parameters
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,  # Ensure api_key is passed correctly
)

# Call the model
response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    model=deployment
)

# Print the response
print(response.choices[0].message.content)
