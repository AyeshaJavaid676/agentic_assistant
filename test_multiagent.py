# test_question.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.vector_store import VectorStore
from modules.multi_agent import SupervisorAgent

# Load vector store
vectorstore = VectorStore()
try:
    vectorstore.load_existing()
    print("✅ Vector store loaded")
except:
    print("⚠️ No existing vector store, creating new one")
    vectorstore = VectorStore()

# Create supervisor
supervisor = SupervisorAgent(vectorstore)

# Ask a question
question = input("\n❓ Tell about my skills and n8n: ")
if question:
    print(f"\n🔄 Processing: '{question}'")
    response = supervisor.run(question)
    print(f"\n✅ Answer: {response['output']}")