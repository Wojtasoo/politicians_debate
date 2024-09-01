from flask import Flask, request, jsonify
import os
import requests
import time
from dotenv import load_dotenv
import json
load_dotenv()
from agentsdebate import start_debate

app = Flask(__name__)

@app.route('/debate-created', methods=['POST'])
def debate_created_webhook():
    data = request.json
    print(data)
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Extract the debate information
    debate_id = data.get('debateId')
    prompt = data.get('prompt')
    politician1 = data.get('politician1')
    politician2 = data.get('politician2')
    
    # Validate the required fields
    if not all([debate_id, prompt, politician1, politician2]):
        return jsonify({"error": "Missing required fields"}), 400
    
    # Process the debate information
    # This is where you'd add your logic to handle the new debate
    # For example, you might want to:
    #  - Store the debate in a database
    #  - Trigger an AI to generate the debate content
    #  - Send notifications
    #  - etc.
    
    print(f"New debate created:")
    print(f"ID: {debate_id}")
    print(f"Prompt: {prompt}")
    print(f"Politicians: {politician1} vs {politician2}")
    start_debate(debate_id, prompt, politician1,politician2)
    # Here's where you'd call your debate processing function
    # process_debate(debate_id, prompt, politician1, politician2)
    
    return jsonify({"message": "Debate received and processing initiated"}), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 6000))
    app.run(host='0.0.0.0', port=port)
