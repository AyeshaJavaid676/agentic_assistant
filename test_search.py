import os
from modules.web_search import WebSearch
# If you have your .env loading in your config, this should work automatically.
# Otherwise, uncomment the next two lines:
# from dotenv import load_dotenv
# load_dotenv()

def test_search():
    print("🚀 Initializing WebSearch module...")
    try:
        # 1. Initialize the class (this will trigger your __init__ print)
        web_tool = WebSearch()
        
        # 2. Define a specific test query
        test_query = "What is n8n workflow automation and what are its key features?"
        print(f"\n📡 Sending Query: '{test_query}'")
        
        # 3. Run the search
        results = web_tool.search(test_query, max_results=2)
        
        # 4. Display and Validate Results
        print("\n" + "="*50)
        print("🔎 TAVILY SEARCH RESULTS:")
        print("="*50)
        print(results)
        print("="*50)
        
        if "Title:" in results and "SUMMARY:" in results:
            print("\n✅ SUCCESS: Tavily returned structured data successfully!")
        elif "No results found" in results:
            print("\n⚠️ WARNING: Search worked but found no matches. Check query.")
        else:
            print("\n❌ FAILED: The output format is not as expected.")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR during test: {str(e)}")

if __name__ == "__main__":
    test_search()