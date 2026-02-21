from langchain_core.tools import tool, Tool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from config.settings import GROQ_API_KEY, LLM_MODEL
from modules.web_search import WebSearch 
import os

class BaseSpecialistAgent:
    """Base class for all specialist agents"""
    def __init__(self, name, description, tools, system_prompt):
        self.name = name
        self.description = description
        self.llm = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, temperature=0.1)
        self.tools = tools
        
        template = f"""{system_prompt}

You have access to the following tools:
{{tools}}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
Question: {{input}}
Thought: {{agent_scratchpad}}"""

        prompt = PromptTemplate.from_template(template)
        
        agent = create_react_agent(self.llm, tools, prompt)
        self.executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            max_iterations=3,
            handle_parsing_errors=True
        )
    
    def run(self, query):
        try:
            result = self.executor.invoke({"input": query})
            return result["output"]
        except Exception as e:
            return f"[{self.name} Error: {str(e)}]"
    
    def as_tool(self):
        return Tool(
            name=self.name,
            func=lambda q: self.run(q),
            description=self.description
        )

class PDFResearcherAgent(BaseSpecialistAgent):
    def __init__(self, vectorstore):
        @tool
        def search_pdf(query: str) -> str:
            """Search the PDF vector database for relevant information"""
            try:
                docs = vectorstore.search(query, k=4)
                if not docs: 
                    return "No relevant information found."
                
                results = []
                for doc in docs:
                    page = doc.metadata.get('page', '?')
                    results.append(f"[Page {page}]: {doc.page_content}")
                return "\n\n".join(results)
            except Exception as e:
                return f"Error: {str(e)}"
        
        super().__init__(
            name="PDF_Researcher",
            description="Searches the uploaded PDF for text and facts.",
            tools=[search_pdf],
            system_prompt="You are a PDF expert. Use search_pdf to find facts."
        )

class WebResearcherAgent(BaseSpecialistAgent):
    def __init__(self):
        self.web_search = WebSearch() 
        
        @tool
        def search_web(query: str) -> str:
            """Search the internet for real-time information"""
            return self.web_search.search(query, max_results=3)
        
        super().__init__(
            name="Web_Researcher",
            description="Finds real-time info on the internet.",
            tools=[search_web],
            system_prompt="You are a web expert. Use search_web for current news."
        )

class SynthesizerAgent(BaseSpecialistAgent):
    def __init__(self):
        super().__init__(
            name="Synthesizer",
            description="Combines information from multiple sources into clear answers.",
            tools=[],
            system_prompt="You are a synthesis expert. Combine information from different sources into one clear, comprehensive answer."
        )

class SupervisorAgent:
    """The Manager Agent"""
    def __init__(self, vectorstore):
        self.pdf_agent = PDFResearcherAgent(vectorstore)
        self.web_agent = WebResearcherAgent()
        self.synthesizer = SynthesizerAgent()
        
        self.specialist_tools = [
            self.pdf_agent.as_tool(),
            self.web_agent.as_tool()
        ]
        
        self.llm = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, temperature=0.1)
        
        # FIXED: Proper template with all required variables
        supervisor_template = """You are a Supervisor Agent managing a team of specialists.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer - I should call Synthesizer
Final Answer: the final answer to the original input question

AVAILABLE SPECIALISTS:
- PDF_Researcher: For questions about document content, text, or data in PDFs
- Web_Researcher: For current events, facts not in PDFs, external knowledge

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate.from_template(supervisor_template)
        agent = create_react_agent(self.llm, self.specialist_tools, prompt)
        self.executor = AgentExecutor(
            agent=agent, 
            tools=self.specialist_tools, 
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )

    def run(self, question):
        try:
            # Let supervisor decide which specialists to call
            supervisor_response = self.executor.invoke({"input": question})
            intermediate_info = supervisor_response['output']
            
            # Synthesize into final answer
            print("\n🔄 Synthesizer combining information...")
            final = self.synthesizer.run(
                f"Information gathered: {intermediate_info}\n\n"
                f"Original question: {question}\n\n"
                f"Please provide a comprehensive answer."
            )
            
            return {"output": final}
            
        except Exception as e:
            print(f"❌ Multi-agent error: {e}")
            import traceback
            traceback.print_exc()
            return {"output": f"Error processing your question: {str(e)}"}