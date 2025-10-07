import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")

tavily_client = TavilyClient(api_key=api_key)
response = tavily_client.search("Who is Leo Messi?")
print("response: ", response)