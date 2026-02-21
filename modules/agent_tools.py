from langchain_core.tools import tool
from langchain.tools import Tool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from modules.web_search import WebSearch
from config.settings import GROQ_API_KEY, LLM_MODEL

class AgentAssistant:
    def __init__(self, vectorstore):
        """Initialize the agent with tools"""
        print("🤖 Setting up AI agent...")
        
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_MODEL,
            temperature=0.1,
            max_tokens=1024
        )
        self.vectorstore = vectorstore
        self.web_search_engine = WebSearch()
        self.setup_agent()
        print("✅ Agent ready!")
    
    def setup_agent(self):
        """Create the agent with tools using @tool decorator"""
        
        # 1. Define tools with clear descriptions for the LLM
        @tool
        def pdf_search(query: str) -> str:
            """Search the uploaded PDF document for information. Use this for questions 
            about the content of the document, images described in the text, or specific data from the PDF."""
            # If query is about images, search for the image markers
            if any(word in query.lower() for word in ['image', 'picture', 'figure', 'chart', 'graph', 'photo']):
                # Search specifically for image descriptions
                docs = self.vectorstore.search("Image on page", k=5)
                if docs:
                    return "\n\n".join([doc.page_content for doc in docs])
            # Otherwise, search normally
            docs = self.vectorstore.search(query, k=3)
            return "\n\n".join([doc.page_content for doc in docs])
            
        
        @tool
        def web_search(query: str) -> str:
            """Search the internet for real-time information. Use this for current events, 
            verifying facts outside the PDF, or gathering broader context."""
            
            return self.web_search_engine.search(query)
        
        tools = [pdf_search, web_search]
        
        # 2. Define the ReAct Prompt Template
        # IMPORTANT: The variables {tools} and {tool_names} are REQUIRED by create_react_agent
        template = """You are a professional AI assistant. Answer the following questions as accurately as possible. 
        You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        
        # 3. Create the agent and executor
        self.agent = create_react_agent(self.llm, tools, prompt)
        self.executor = AgentExecutor(
            agent=self.agent, 
            tools=tools, 
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def run(self, question):
        """Run the agent with a question"""
        try:
            # We pass only 'input' because create_react_agent handles 
            # the tools and tool_names injection automatically
            response = self.executor.invoke({"input": question})
            return response
        except Exception as e:
            print(f"Agent Error: {e}")
            return {"output": f"I encountered an error while processing your request: {str(e)}"}