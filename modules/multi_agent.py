from langchain_core.tools import tool, Tool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser  # Add this
from config.settings import GROQ_API_KEY, LLM_MODEL
from modules.web_search import WebSearch 
import os

class BaseSpecialistAgent:
    """Base class for all specialist agents"""
    def __init__(self, name, description, tools, system_prompt, use_react=True):
        self.name = name
        self.description = description
        self.llm = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, temperature=0.1)
        self.tools = tools
        self.use_react = use_react
        
        if use_react:
            # ReAct agent for tools that need decision making
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
        else:
            # MODERN: Use pipe operator instead of LLMChain
            self.prompt = PromptTemplate.from_template(f"{system_prompt}\n\nQuestion: {{input}}\n\nAnswer:")
            self.chain = self.prompt | self.llm | StrOutputParser()
    
    def run(self, query):
        try:
            if self.use_react:
                result = self.executor.invoke({"input": query})
                return result["output"]
            else:
                # MODERN: Use invoke instead of run
                return self.chain.invoke({"input": query})
        except Exception as e:
            return f"[{self.name} Error: {str(e)}]"
    
    def as_tool(self):
        return Tool(
            name=self.name,
            func=lambda q: self.run(q),
            description=self.description
        )


class CodeExplanationAgent(BaseSpecialistAgent):
    """Simple chain-based agent for explaining code concepts"""
    def __init__(self):
        super().__init__(
            name="Code_Explainer",
            description="Explains programming concepts, syntax, and best practices in simple terms.",
            tools=[],  # No tools needed
            system_prompt="""You are a friendly programming teacher. Explain coding concepts clearly and simply.

Guidelines:
- Break down complex topics into simple steps
- Use analogies when helpful
- Provide small code examples to illustrate points
- Assume the learner is at an intermediate level
- Focus on practical understanding, not just theory

Explain the concept thoroughly but concisely.""",
            use_react=False  # Simple chain - no tools needed
        )


class CodeDebuggerAgent(BaseSpecialistAgent):
    """Agent for debugging code issues"""
    def __init__(self):
        super().__init__(
            name="Code_Debugger",
            description="Helps identify and fix bugs in code. Analyzes error messages and suggests solutions.",
            tools=[],  # No tools needed - just reasoning
            system_prompt="""You are an expert debugger. Help identify and fix issues in code.

When debugging:
1. Analyze the error message carefully
2. Identify the most likely cause
3. Suggest specific fixes
4. Explain why the fix works
5. Provide corrected code examples

Be methodical and thorough in your analysis.""",
            use_react=False  # Simple chain for reasoning
        )


class CodeSynthesizerAgent(BaseSpecialistAgent):
    """Synthesizer that combines all information - using modern chain syntax"""
    def __init__(self):
        # We'll override run method to use custom prompt
        self.llm = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, temperature=0.1)
        self.prompt = PromptTemplate.from_template(
            """You are a technical documentation expert. Combine information from multiple sources into clear, helpful answers.

Information gathered: {info}

Original question: {question}

Provide a comprehensive answer with:
- Brief overview
- Step-by-step explanation
- Code examples (with proper formatting)
- Best practices and tips
- Common pitfalls to avoid

Make your answer practical and immediately useful for developers."""
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        # Call parent but skip its initialization
        self.name = "Code_Synthesizer"
        self.description = "Combines code examples, explanations, and web research into comprehensive answers."
    
    def run(self, info, question=None):
        """Custom run method that takes both info and question"""
        try:
            if question is None:
                # If called as a tool, parse the input
                import json
                try:
                    data = json.loads(info)
                    info = data.get('info', info)
                    question = data.get('question', '')
                except:
                    question = ""
            
            return self.chain.invoke({
                "info": info,
                "question": question
            })
        except Exception as e:
            return f"[Synthesizer Error: {str(e)}]"
    
    def as_tool(self):
        """Convert to tool for supervisor"""
        return Tool(
            name=self.name,
            func=lambda q: self.run(q),
            description=self.description
        )


# Keep all your other agent classes (CodeDocumentExpertAgent, WebResearcherAgent, etc.)
# and CodingSupervisorAgent - they remain the same
class CodeDocumentExpertAgent(BaseSpecialistAgent):
    """Specialist agent for searching and explaining code in documents"""
    def __init__(self, vectorstore):
        @tool
        def search_code(query: str) -> str:
            """Search code documents for functions, classes, and examples"""
            try:
                # Search for code-related terms
                docs = vectorstore.search(query, k=5)
                if not docs: 
                    return "No relevant code found."
                
                results = []
                for doc in docs:
                    page = doc.metadata.get('page', '?')
                    content = doc.page_content
                    
                    # Detect if content contains code
                    if any(code_word in content for code_word in ['def ', 'class ', 'import ', 'function', '```', '=>', 'var ']):
                        results.append(f"[CODE on Page {page}]:\n{content}")
                    else:
                        results.append(f"[EXPLANATION on Page {page}]:\n{content}")
                
                return "\n\n---\n\n".join(results)
            except Exception as e:
                return f"Error searching code: {str(e)}"
        
        super().__init__(
            name="Code_Expert",
            description="Searches code documents for functions, classes, APIs, and programming examples. Use for questions about code in your documents.",
            tools=[search_code],
            system_prompt="""You are a senior software engineer specializing in code analysis and documentation.

Your expertise:
- Understanding code snippets in Python, JavaScript, Java, and other languages
- Explaining what functions and classes do
- Identifying code patterns and best practices
- Finding API usage examples
- Debugging common coding issues

When you find code:
1. Identify the programming language
2. Explain what the code does
3. Point out important patterns or potential issues
4. Suggest improvements if relevant

Always provide clear, practical explanations that help developers understand and use the code.""",
            use_react=True  # Needs ReAct for searching
        )





class WebResearcherAgent(BaseSpecialistAgent):
    """Specialist agent for web searches - keeps ReAct for search decisions"""
    def __init__(self):
        self.web_search = WebSearch() 
        
        @tool
        def search_web(query: str) -> str:
            """Search the internet for programming resources, documentation, and tutorials"""
            try:
                print(f"🌐 Searching web for: {query}")
                results = self.web_search.search(query, max_results=4)
                
                # If no results, try programming-specific terms
                if not results or results == "No web results found.":
                    prog_queries = [
                        f"{query} documentation",
                        f"{query} stack overflow",
                        f"{query} tutorial",
                        f"{query} example code"
                    ]
                    for prog_query in prog_queries[:2]:
                        results = self.web_search.search(prog_query, max_results=3)
                        if results and results != "No web results found.":
                            break
                
                return results if results else "No web results found."
            except Exception as e:
                return f"Web search failed: {str(e)}"
        
        super().__init__(
            name="Web_Researcher",
            description="Searches the web for programming tutorials, documentation, Stack Overflow answers, and latest tech news.",
            tools=[search_web],
            system_prompt="""You are a technical research specialist.

Focus on finding:
- Official documentation
- Stack Overflow solutions
- Tutorials and guides
- GitHub examples
- Latest programming trends

Prioritize authoritative sources and recent information.""",
            use_react=True  # Needs ReAct for search decisions
        )





class SupervisorAgent:
    """Supervisor that coordinates all coding specialist agents"""
    def __init__(self, vectorstore):
        print("="*60)
        print("🤖 CODING ASSISTANT SUPERVISOR")
        print("="*60)
        
        # Create all coding specialist agents
        self.code_expert = CodeDocumentExpertAgent(vectorstore)
        self.code_explainer = CodeExplanationAgent()
        self.code_debugger = CodeDebuggerAgent()
        self.web_researcher = WebResearcherAgent()
        self.synthesizer = CodeSynthesizerAgent()
        
        # Tools for supervisor (only agents that need to be called)
        self.specialist_tools = [
            self.code_expert.as_tool(),
            self.code_explainer.as_tool(),
            self.code_debugger.as_tool(),
            self.web_researcher.as_tool(),
            self.synthesizer.as_tool()
        ]
        
        self.llm = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, temperature=0.1)
        
        # Supervisor prompt for coding questions
        supervisor_template = """You are a Senior Technical Lead managing a team of coding specialists.

AVAILABLE SPECIALISTS:

1. Code_Expert - Use for:
   - Finding specific functions/classes in your code documents
   - Understanding what code in your documents does
   - Locating API usage examples
   - Analyzing code structure

2. Code_Explainer - Use for:
   - Understanding programming concepts
   - Learning syntax and language features
   - Getting simplified explanations
   - Understanding best practices

3. Code_Debugger - Use for:
   - Fixing errors in code
   - Understanding error messages
   - Debugging logic issues
   - Code review and improvement

4. Web_Researcher - Use for:
   - Latest programming trends
   - External documentation
   - Stack Overflow solutions
   - Tutorials and guides

DECISION RULES:
- If question asks about code IN documents → Code_Expert
- If question asks about concepts or learning → Code_Explainer
- If question involves errors or bugs → Code_Debugger
- If question needs external info → Web_Researcher
- For complex questions, use MULTIPLE specialists then Synthesizer

You have access to these tools:
{tools}

Use this format:

Question: {input}
Thought: Consider which specialists to use
Action: one of [{tool_names}]
Action Input: specific question for that specialist
Observation: result
... (repeat as needed)
Thought: I have all information - call Synthesizer
Final Answer: final response

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate.from_template(supervisor_template)
        agent = create_react_agent(self.llm, self.specialist_tools, prompt)
        self.executor = AgentExecutor(
            agent=agent, 
            tools=self.specialist_tools, 
            verbose=True,
            max_iterations=8,
            handle_parsing_errors=True
        )
        
        print("✅ Coding Assistant Ready!")
        print(f"   Specialists: Code_Expert, Code_Explainer, Code_Debugger, Web_Researcher, Synthesizer")
        print("="*60)

    def run(self, question):
        try:
            # Let supervisor coordinate
            supervisor_response = self.executor.invoke({"input": question})
            intermediate_info = supervisor_response['output']
            
            # MODERN: Use the updated synthesizer with both info and question
            print("\n🔄 Synthesizing final answer...")
            final = self.synthesizer.run(
                info=intermediate_info,
                question=question
            )
            
            return {"output": final}
            
        except Exception as e:
            print(f"❌ Coding Assistant error: {e}")
            import traceback
            traceback.print_exc()
            return {"output": f"Error: {str(e)}. Please try rephrasing your question."}

