import os
from dotenv import load_dotenv

load_dotenv()


def web_search(query: str) -> str:
    """Search the internet using Tavily for research-ready content."""
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(query=query, search_depth="advanced", max_results=5)
    results = response.get("results", [])
    return "\n\n".join(
        f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content'][:900]}"
        for r in results
    )
