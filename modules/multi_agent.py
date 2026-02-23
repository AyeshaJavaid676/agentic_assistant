from langchain_core.tools import tool, Tool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser  
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
    """Enhanced Synthesizer that prioritizes user documents and incorporates best coding practices"""
    
    def __init__(self):
        self.llm = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, temperature=0.1)
        
        # Main synthesis prompt with clear prioritization
        self.main_prompt = PromptTemplate.from_template(
            """You are a Senior Technical Lead and Code Reviewer. Your role is to synthesize information with a clear hierarchy:

            ===== PRIORITY HIERARCHY (MOST TO LEAST IMPORTANT) =====
            1️⃣ **USER'S DOCUMENTS** (highest priority - their actual code/project)
            2️⃣ **WEB RESEARCH** (external information when needed)
            3️⃣ **GENERAL KNOWLEDGE** (fallback only)

            ===== CURRENT INPUT =====
            USER'S DOCUMENTS:
            {document_info}
            
            WEB RESEARCH (if any):
            {web_info}
            
            ORIGINAL QUESTION: {question}

            ===== YOUR TASK =====
            Create a comprehensive answer following this structure:

            ## 📊 WHAT YOUR DOCUMENT SAYS
            - Start with findings from the user's own documents
            - Quote specific page numbers, tables, and code snippets
            - Use exact numbers and findings from their project
            
            ## 🔍 DETAILED ANALYSIS
            - Explain the context using document information
            - Add web research if it complements (not replaces) document info
            - Only use general knowledge if documents lack information

            ## 💻 CODE EXAMPLES
            - If relevant, show code that matches their project context
            - Use proper language specification in code blocks
            - Include comments explaining key parts

            ## ✨ BEST PRACTICES
            - Suggest improvements based on their specific code
            - Reference industry standards
            - Include optimization tips

            ## ⚠️ COMMON PITFALLS
            - Warn about issues relevant to their specific code
            - Suggest how to avoid them

            ===== FORMATTING RULES =====
            - Use **bold** for key numbers and findings
            - Use `code` for inline references
            - Use ```language for code blocks
            - Add emojis for visual hierarchy (📊, 🔍, 💻, ✨, ⚠️)
            - ALWAYS cite sources: "According to your document (Page X)..."

            Remember: The user's own documents contain the MOST RELEVANT information. Make them feel heard and valued by referencing their actual work!"""
        )
        
        # Code review specific prompt
        self.code_review_prompt = PromptTemplate.from_template(
            """You are conducting a code review on the user's project.

            USER'S CODE/DOCUMENTS:
            {document_info}
            
            REVIEW FOCUS: {question}

            Provide a thorough code review covering:
            
            ## 📁 CODE QUALITY
            - Readability and maintainability
            - Naming conventions
            - Comment quality
            
            ## 🐛 POTENTIAL BUGS
            - Edge cases not handled
            - Error handling gaps
            - Logic issues
            
            ## ⚡ PERFORMANCE
            - Inefficient operations
            - Memory usage concerns
            - Optimization opportunities
            
            ## 🔒 SECURITY
            - Input validation
            - Data exposure risks
            - Best practice violations
            
            ## ✅ RECOMMENDATIONS
            - Specific fixes with code examples
            - Priority level (High/Medium/Low)
            - Implementation suggestions
            
            Format with clear sections and actionable feedback."""
        )
        
        # Learning path prompt
        self.learning_prompt = PromptTemplate.from_template(
            """Create a personalized learning path based on the user's project.

            USER'S PROJECT CONTEXT:
            {document_info}
            
            THEIR QUESTION: {question}

            Provide:
            
            ## 🎯 CURRENT PROJECT ANALYSIS
            - What their code reveals about their skill level
            - Strengths shown in their work
            - Areas for growth
            
            ## 📚 PERSONALIZED LEARNING PATH
            - 3-5 specific topics to learn next
            - Resources tailored to their project
            - Practice exercises using their code
            
            ## 🛠️ TOOLS & EXTENSIONS
            - VS Code extensions that would help
            - Libraries to explore
            - Debugging tools
            
            ## 🏆 NEXT PROJECT IDEAS
            - 2-3 project suggestions building on their skills
            - What they'd learn from each
            - Difficulty level estimate"""
        )
        
        # Main chain
        self.main_chain = self.main_prompt | self.llm | StrOutputParser()
        self.code_review_chain = self.code_review_prompt | self.llm | StrOutputParser()
        self.learning_chain = self.learning_prompt | self.llm | StrOutputParser()
        
        self.name = "Code_Synthesizer"
        self.description = "Combines document findings with best practices - prioritizes user's actual code and projects"
    
    def _detect_query_type(self, question):
        """Detect if this is a code review or learning path question"""
        question_lower = question.lower()
        
        review_keywords = ['review', 'feedback', 'improve', 'better', 'quality', 'bug', 'fix', 'error']
        learning_keywords = ['learn', 'study', 'course', 'tutorial', 'beginner', 'advanced', 'practice']
        
        review_score = sum(1 for word in review_keywords if word in question_lower)
        learning_score = sum(1 for word in learning_keywords if word in question_lower)
        
        if review_score > learning_score:
            return 'review'
        elif learning_score > review_score:
            return 'learning'
        else:
            return 'general'
    
    def _extract_document_info(self, info):
        """Extract and structure document information"""
        if not info or info == "No document search results available.":
            return "No specific document information found."
        
        # Try to parse if it's already structured
        try:
            if isinstance(info, str) and 'DOCUMENT INFORMATION' in info:
                return info
        except:
            pass
        
        return info
    
    def run(self, info, question=None):
        """Enhanced run method with query type detection"""
        try:
            # Parse inputs
            if question is None:
                import json
                try:
                    data = json.loads(info) if isinstance(info, str) else {}
                    info = data.get('info', info)
                    question = data.get('question', '')
                except:
                    question = ""
            
            # Extract document and web info
            document_info = "No document information provided."
            web_info = "No web research provided."
            
            if isinstance(info, str):
                if 'DOCUMENT INFORMATION' in info and 'WEB INFORMATION' in info:
                    parts = info.split('WEB INFORMATION')
                    document_info = parts[0].replace('DOCUMENT INFORMATION', '').strip()
                    web_info = parts[1].strip() if len(parts) > 1 else "No web information provided."
                else:
                    document_info = info
            
            # Detect query type
            query_type = self._detect_query_type(question)
            
            # Route to appropriate chain
            if query_type == 'review':
                result = self.code_review_chain.invoke({
                    "document_info": document_info,
                    "question": question
                })
            elif query_type == 'learning':
                result = self.learning_chain.invoke({
                    "document_info": document_info,
                    "question": question
                })
            else:
                result = self.main_chain.invoke({
                    "document_info": document_info,
                    "web_info": web_info,
                    "question": question
                })
            
            return result
            
        except Exception as e:
            return f"""
## ❌ Synthesis Error

**Error:** {str(e)}

**What happened:** The synthesizer encountered an issue while processing your request.

**Suggested fix:** Please try rephrasing your question or check if your documents were processed correctly.

*If this persists, check the logs for more details.*
"""
    
    def as_tool(self):
        """Convert to tool for supervisor"""
        return Tool(
            name=self.name,
            func=lambda q: self.run(q),
            description=self.description
        )

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
        
        # Store vectorstore for direct access
        self.vectorstore = vectorstore
        
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

5. Synthesizer - Use for:
   - Combining information from multiple specialists
   - Creating comprehensive final answers
   - ALWAYS call this at the end

DECISION RULES:
- If question asks about code IN documents → Code_Expert FIRST
- If question asks about concepts or learning → Code_Explainer FIRST
- If question involves errors or bugs → Code_Debugger FIRST
- If question needs external info → Web_Researcher FIRST
- For complex questions, use MULTIPLE specialists
- ALWAYS end by calling Synthesizer to combine everything

You have access to these tools:
{tools}

Use this format:

Question: {input}
Thought: Consider which specialists to use based on the question
Action: one of [{tool_names}]
Action Input: specific question for that specialist
Observation: result
... (repeat as needed)
Thought: I have gathered all necessary information
Action: Synthesizer
Action Input: Combine all findings into a comprehensive answer
Observation: synthesized result
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
        """Run the supervisor agent with document-first prioritization"""
        try:
            print(f"\n🔍 Processing question: '{question}'")
            print("-"*60)
            
            # STEP 1: ALWAYS search documents directly FIRST (bypass agent for reliability)
            doc_context = "No document information found."
            doc_results = []
            
            if self.vectorstore:
                try:
                    print("📚 Searching documents directly...")
                    doc_results = self.vectorstore.search(question, k=15)  # Get more results
                    
                    if doc_results:
                        print(f"✅ Found {len(doc_results)} relevant document chunks")
                        
                        # Build detailed document context
                        doc_parts = []
                        for i, doc in enumerate(doc_results, 1):
                            doc_name = doc.metadata.get('document', doc.metadata.get('source', 'your document'))
                            page = doc.metadata.get('page', '?')
                            doc_type = doc.metadata.get('type', 'text')
                            content = doc.page_content
                            
                            # Truncate very long content but keep the important parts
                            if len(content) > 500:
                                content = content[:500] + "..."
                            
                            type_icon = "🖼️" if doc_type == 'image' else "📄"
                            doc_parts.append(f"{type_icon} [DOC {i}] From: {doc_name}, Page {page}\n{content}")
                        
                        doc_context = "\n\n---\n\n".join(doc_parts)
                    else:
                        print("⚠️ No documents found matching the query")
                        doc_context = "No matching documents found in your PDFs."
                except Exception as e:
                    print(f"⚠️ Error searching documents: {e}")
                    doc_context = f"Error searching documents: {str(e)}"
            else:
                print("⚠️ No vectorstore available")
                doc_context = "No PDF documents have been loaded yet."
            
            # STEP 2: Now let the supervisor agent handle any web research needed
            print("\n🤖 Consulting specialist agents...")
            supervisor_response = self.executor.invoke({"input": question})
            agent_output = supervisor_response['output']
            
            # STEP 3: Combine document info (HIGH PRIORITY) with agent output
            combined_info = f"""
============================================================
📚 DOCUMENT INFORMATION (HIGHEST PRIORITY - MUST USE THIS FIRST)
============================================================
{doc_context}

============================================================
🤖 AGENT RESEARCH INFORMATION (SECONDARY - USE ONLY IF DOCUMENTS LACK INFO)
============================================================
{agent_output}

============================================================
INSTRUCTIONS FOR SYNTHESIZER:
1. FIRST analyze the DOCUMENT INFORMATION above - this is the user's actual project data
2. QUOTE specific findings with page numbers (e.g., "On Page 8, your document shows...")
3. Use the actual numbers from their documents (like "19 outliers in Income")
4. ONLY use agent research if documents don't have relevant information
5. NEVER ignore document information when it exists
============================================================
"""
            
            # STEP 4: Synthesize with clear prioritization
            print("\n🔄 Synthesizing answer - prioritizing document findings...")
            final = self.synthesizer.run(
                info=combined_info,
                question=question
            )
            
            return {"output": final}
            
        except Exception as e:
            print(f"❌ Coding Assistant error: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback response
            return {
                "output": f"""I encountered an error while processing your question: {str(e)}

However, I can see you have PDF documents loaded. Here's what I found in them directly:

{doc_context if 'doc_context' in locals() else 'No documents found.'}

Please try rephrasing your question or check if your PDFs were processed correctly."""
            }
