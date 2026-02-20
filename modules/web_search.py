from duckduckgo_search import DDGS

class WebSearch:
    def __init__(self):
        """Initialize web search"""
        self.ddgs = DDGS()
        print("🌐 Web search ready")
    
    def search(self, query, max_results=3):
        """Search the web for information"""
        try:
            print(f"🔍 Searching web for: {query}")
            results = []
            for r in self.ddgs.text(query, max_results=max_results):
                results.append(f"Title: {r['title']}\nSnippet: {r['body']}\nLink: {r['href']}")
            
            if results:
                return "\n\n".join(results)
            else:
                return "No web results found."
        except Exception as e:
            return f"Web search failed: {str(e)}"