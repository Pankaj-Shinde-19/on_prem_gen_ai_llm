import requests
import json

# Define the URL and headers
url = "http://localhost:11434/api/generate"
headers = {
    "Content-Type": "application/json"
}

# Define the payload
payload = {
    "model": "phi3:mini",
    "prompt": "What is the capital of France?"
}

try:
    # Make the POST request
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Split the response into individual JSON objects
    raw_data = response.text.strip().split("\n")

    # Parse each JSON object and extract the 'response' field
    output = []
    for item in raw_data:
        try:
            data = json.loads(item)
            if 'response' in data:
                output.append(data['response'])
        except json.JSONDecodeError:
            print(f"Failed to parse: {item}")

    # Combine the extracted responses into a complete output
    final_output = "".join(output)
    print("Final Output:", final_output)

except Exception as e:
    print(f"An error occurred: {e}")
