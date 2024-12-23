from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
from torchvision import models, transforms

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Backend is running!"



@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    images = data.get('images', [])
    symptoms = data.get('symptoms', [])

    # Process images and symptoms
    processed_images = []
    for img_str in images:
        image = Image.open(io.BytesIO(base64.b64decode(img_str)))
        processed_images.append(image)

    # Call the AI model for diagnosis
    diagnosis_results = get_diagnosis(processed_images, symptoms)

    return jsonify(diagnosis_results)

@app.route('/register', methods=['POST'])
def register():
    # Handle user registration
    pass

@app.route('/login', methods=['POST'])
def login():
    # Handle user login
    pass
@app.route('/logout', methods=['POST'])
def login():
    # Handle user login
    pass


# Load the tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained('path_to_tokenizer')
model = LlamaForCausalLM.from_pretrained('path_to_model')

# Ensure you're using the correct device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def get_diagnosis(images, symptoms):
    # Preprocess images (e.g., extract features using a CNN)
    # For simplicity, let's assume we extract textual descriptions
    image_descriptions = []

    for image in images:
        # Implement your image processing logic here
        # For example, use a pre-trained image classifier
        image_description = 'skin lesion with redness and swelling'
        image_descriptions.append(image_description)

    # Combine symptoms and image descriptions
    input_text = generate_input_text(image_descriptions, symptoms)

    # Tokenize and encode the input
    inputs = tokenizer(input_text, return_tensors='pt').to(device)

    # Generate output
    outputs = model.generate(**inputs, max_length=512)
    diagnosis_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Parse the output to extract diagnoses and probabilities
    diagnosis_results = parse_diagnosis_output(diagnosis_text)

    return diagnosis_results

def generate_input_text(image_descriptions, symptoms):
    # Combine image descriptions and symptoms into a prompt
    prompt = "Patient presents with the following symptoms: "
    prompt += ', '.join(symptoms) + '. '
    prompt += "Images indicate: " + ', '.join(image_descriptions) + '. '
    prompt += "Provide the most probable diagnoses with probabilities and suggested treatments."
    return prompt

def parse_diagnosis_output(diagnosis_text):
    # Parse the model's output to extract structured data
    # This will depend on how the model formats its output
    # For simplicity, let's assume it returns JSON-like text
    # You may need to use regex or NLP techniques to parse it
    diagnosis_results = {
        'diagnoses': [
            {'condition': 'Eczema', 'probability': '70%', 'treatment': 'Apply moisturizer'},
            {'condition': 'Psoriasis', 'probability': '20%', 'treatment': 'Use topical steroids'},
            {'condition': 'Allergic Reaction', 'probability': '10%', 'treatment': 'Take antihistamines'},
        ]
    }
    return diagnosis_results


# Load a pre-trained CNN
cnn_model = models.resnet18(pretrained=True)
cnn_model.eval()
cnn_model.to(device)

# Define image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def extract_image_features(image):
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = cnn_model(image_tensor)
    return features.cpu().numpy()
    
if __name__ == '__main__':
    app.run(debug=True)