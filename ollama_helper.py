import os
import requests
import json

def find_ollama_port():
    ollama_host = os.getenv('OLLAMA_HOST')
    if ollama_host:
        _, port_str = ollama_host.split(':')
        return int(port_str)
    else:
        return 11434

OLLAMA_PORT = find_ollama_port()

def query_model_ollama(query):
    url = f"http://localhost:{OLLAMA_PORT}/api/chat"
    response = requests.post(url, json=query)
    response.raise_for_status()
    response_json = json.loads(response.text)
    return response_json["message"]["content"]
