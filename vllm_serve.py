from llama_index.llms.vllm import VllmServer
from llama_index.core.llms import ChatMessage
import json

llm = VllmServer(
    api_url="http://localhost:5000/generate", max_new_tokens=10, temperature=0
)

def generate_response(prompt):
    payload = {
        "prompt": prompt,
        "stream": False
    }
    response = llm.complete(payload)
    return response

print("Welcome to this simple chat bot interface. Type your messages below. Type 'EXIT' to end the chat.")
while True:
    print()
    user_input = input("You: ")
    if user_input.strip().upper() == "EXIT":
        print("Bye!")
        break
    response = generate_response(user_input)
    print("\nAssistant:", response)
