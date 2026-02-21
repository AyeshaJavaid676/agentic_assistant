from tavily import TavilyClient
from config.settings import TAVILY_API_KEY

class WebSearch:
    def __init__(self):
        """Initialize Tavily using the key from central config"""
        if not TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY is missing in your .env or config!")
            
        self.client = TavilyClient(api_key=TAVILY_API_KEY)
        print("🌐 Tavily Web Search ready (Connected via Config)")

    def search(self, query, max_results=3):
        try:
            print(f"🔍 AI Researching: {query}")
            response = self.client.search(query=query, max_results=max_results)
            
            results = []
            for r in response.get('results', []):
                # Standardizing the labels for the Agent to read
                title = r.get('title', 'No Title')
                content = r.get('content', 'No content available')
                url = r.get('url', '#')
                
                results.append(f"Title: {title}\nSUMMARY: {content}\nURL: {url}")
            
            return "\n\n---\n\n".join(results) if results else "No results found."
        except Exception as e:
            return f"Tavily Search failed: {str(e)}"