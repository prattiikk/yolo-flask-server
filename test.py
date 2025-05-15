import base64
import requests

# Load and encode image
with open("test.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# Send request
response = requests.post("http://127.0.0.1:6000/predict", json={"image": encoded_string})
data = response.json()

# Print results
print(data["summary"])