from flask import Flask, request, jsonify, render_template, send_file
import torch
import pickle
import base64
import requests
from Model import ScoliosisCNN  # Your model class
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import io
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from data_preprocessing import get_data_loaders, Preprocessor
import os
from blockchain_logger import log_client_update, log_global_model_update, get_client_reputation
import hashlib

global_model_updated = False

app = Flask(__name__)


# Define a fixed path to your evaluation dataset folder
EVALUATION_DATASET_PATH = r"C:\Users\zboon\Desktop\BC-FLCNN_for_AIS\FL\Data"  # Replace with the actual path to your dataset folder

# Configuration
CLIENTS = []  # List to hold registered clients
ROUNDS = 8
client_accuracies = {}  # Dictionary to store client accuracies
global_model_accuracies = []  # List to store global model accuracies
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global model (to be shared among clients)
global_model = ScoliosisCNN()

# A dictionary to store client metrics and model updates
client_updates = []  # List to store model updates from clients
client_sample_sizes = []  # List to store sample sizes from each client

# Image transformation for model input
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize if required
])

client_metrics = {}
# Helper function to serialize model
def serialize_model(model):
    weights = pickle.dumps(model.state_dict())  # Convert model weights to bytes
    return base64.b64encode(weights).decode()   # Return base64-encoded string

# Helper function to deserialize model
def deserialize_model(encoded_model):
    decoded = base64.b64decode(encoded_model.encode())  # Decode the base64 string
    return pickle.loads(decoded)  # Load the state_dict back into the model

def get_model_hash(state_dict):
    model_bytes = pickle.dumps(state_dict)
    return hashlib.sha256(model_bytes).hexdigest()


def federated_round(round_num):
    print(f"\n🌐 Starting Federated Round {round_num}")

    # Serialize global model
    global_serialized = serialize_model(global_model)
    headers = {"Content-Type": "application/json"}
    data = {"model": global_serialized}

    client_updates = []
    client_sample_sizes = []

    # Send the global model to each client and collect updates
    for client in CLIENTS:
        client_url = client["client_url"]
        try:
            print(f"📤 Sending model to {client_url}")
            res = requests.post(client_url, json=data, timeout=120)
            res_data = res.json()  # Parse the response data (client's updated model)
            updated_weights = deserialize_model(res_data["updated_model"])  # Get the updated weights
            client_updates.append(updated_weights)

            # Get client accuracy and sample size from the response and track it
            accuracy = res_data.get("accuracy")
            num_samples = res_data.get("num_samples", 0)  # Number of samples the client used during training

            if client["client_id"] not in client_accuracies:
                client_accuracies[client["client_id"]] = []
            client_accuracies[client["client_id"]].append(accuracy)

            client_sample_sizes.append(num_samples)  # Store the sample size for weighted averaging

            # Update client status
            client["status"] = "Model received and trained"
            print(f"✅ Received update from {res_data['client_id']} with accuracy {accuracy} and {num_samples} samples")
        except Exception as e:
            print(f"❌ Failed to connect to {client_url}: {e}")
            client["status"] = "Failed to connect"

    # Check if we received any updates from clients
    if len(client_updates) == 0:
        print("❌ No updates received from clients. Skipping aggregation.")
        return

    # Aggregate the updates from all clients using FedAvg
    print("🔄 Aggregating weights using FedAvg...")

    # Compute the total number of samples across all clients
    total_samples = sum(client_sample_sizes)

    # Initialize the aggregated weights with zeros
    aggregated_weights = {key: torch.zeros_like(val) for key, val in client_updates[0].items()}

    # Weighted average of the model weights
    for i, update in enumerate(client_updates):
        weight_factor = client_sample_sizes[i] / total_samples  # Weight by number of samples
        for key in aggregated_weights:
            aggregated_weights[key] += update[key] * weight_factor

    # Update the global model with aggregated weights
    global_model.load_state_dict(aggregated_weights)


    return

def aggregate_weights(weight_list):
    """
    This function aggregates the model weights from multiple clients using Federated Averaging (FedAvg).
    :param weight_list: A list of model weights (state_dict) from different clients.
    :return: The averaged model weights.
    """
    # Start with the first client's weights
    avg_state_dict = weight_list[0]
    
    # Loop through each weight tensor and average them
    for key in avg_state_dict.keys():
        for i in range(1, len(weight_list)):  # Loop through other clients
            avg_state_dict[key] += weight_list[i][key]
        avg_state_dict[key] = avg_state_dict[key] / len(weight_list)  # Average the weights

    return avg_state_dict

def save_global_model():
    # Save the global model's state_dict (weights) to a .pth file
    model_file_path = "global_model.pth"
    torch.save(global_model.state_dict(), model_file_path)
    return model_file_path


@app.route('/')
def index():
    # Pass the client metrics along with clients and plot_image (if needed)
    return render_template('index.html', clients=CLIENTS, client_metrics=client_metrics)


@app.route('/register_client', methods=['POST'])
def register_client():
    client_info = request.json
    client_id = client_info['client_id']
    
    # Check if client is already registered
    if client_id in [client['client_id'] for client in CLIENTS]:
        return jsonify({"message": "Client already registered."}), 400
    
    # Add the client to the list (simulating the registration)
    client_url = f"http://127.0.0.1:{client_info['port']}/train"  # Example URL for the client
    CLIENTS.append({"client_id": client_id, "client_url": client_url, "status": "Registered"})

    # Send a notification to the frontend to inform about the successful registration
    return jsonify({"message": f"Client {client_id} registered successfully!"}), 200

@app.route('/get_model_architecture', methods=['GET'])
def get_model_architecture():
    # Save the model architecture as a Python file (ScoliosisCNN.py)
    model_code = '''
import torch
import torch.nn as nn

class ScoliosisCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(ScoliosisCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    '''
    model_file_path = "ScoliosisCNN.py"
    with open(model_file_path, "w") as f:
        f.write(model_code)
    return send_file(model_file_path, as_attachment=True)

@app.route('/test_global_model', methods=['POST'])
def test_global_model():
    file = request.files['image']  # Get the uploaded image file
    if not file:
        return jsonify({"message": "No image file provided."}), 400

    # Process the image
    image = Image.open(file.stream)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Run the image through the model
    with torch.no_grad():
        output = global_model(image)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    # Convert to percentage for confidence
    confidence_percentage = confidence.item() * 100
    predicted_class = predicted_class.item()

    # Map the predicted class to label (adjust this to your use case)
    class_labels = ['Normal', 'Scoliosis']  # Example labels
    predicted_label = class_labels[predicted_class]

    return jsonify({
        "predicted_label": predicted_label,
        "confidence": confidence_percentage
    })


@app.route('/get_global_model_updates', methods=['GET'])
def get_global_model_updates():
    # Save the global model as a .pth file
    model_file_path = save_global_model()

    # Serve the model weights for download
    return send_file(model_file_path, as_attachment=True, download_name="global_model.pth")

@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    # Serialize the global model and send it to the client
    serialized_model = pickle.dumps(global_model.state_dict())  # Serialize model weights
    encoded_model = base64.b64encode(serialized_model).decode()  # Base64 encode the serialized model
    return jsonify({"model": encoded_model})  # Send as JSON

@app.route('/update_model', methods=['POST'])
def update_model():
    global global_model_updated

    client_data = request.json
    client_id = client_data.get("client_id")
    updated_model = client_data.get("updated_model")
    train_accuracy = client_data.get("train_accuracy")
    val_accuracy = client_data.get("val_accuracy")
    test_accuracy = client_data.get("test_accuracy")
    precision = client_data.get("precision")
    recall = client_data.get("recall")
    f1_score = client_data.get("f1_score")
    num_samples = client_data.get("num_samples")

    if not updated_model or not client_id:
        return jsonify({"message": "Invalid data received."}), 400

    # Deserialize the updated model
    updated_weights = base64.b64decode(updated_model.encode())  # Decode the base64 string
    updated_state_dict = pickle.loads(updated_weights)  # Deserialize the model state

    # Update the global model with the new weights from the client
    global_model.load_state_dict(updated_state_dict)
    # Compute hash of the model update
    model_hash = get_model_hash(updated_state_dict)

    # Convert floats to integers (×100) for on-chain logging (Solidity does not handle floats)
    int_train_acc = int(train_accuracy * 100)
    int_val_acc = int(val_accuracy * 100)
    int_test_acc = int(test_accuracy * 100)
    int_precision = int(precision * 100)
    int_recall = int(recall * 100)
    int_f1 = int(f1_score * 100)

    # Log update to blockchain
    log_client_update(
        client_id, model_hash, 0,  # replace 0 with actual round if available
        int_train_acc, int_val_acc, int_test_acc,
        int_precision, int_recall, int_f1
        )

    # Mark the global model as updated
    global_model_updated = True

    # Store the client metrics in the client_metrics dictionary
    client_metrics[client_id] = {
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "num_samples": num_samples
    }

    # Update the client's status to 'Parameters Received'
    client = next(client for client in CLIENTS if client['client_id'] == client_id)
    client["status"] = "Parameters Received"

    # Respond to the client
    return jsonify({"message": f"Model updated successfully from client {client_id}!"}), 200
@app.route('/display_metrics', methods=['GET'])
def display_metrics():
    return render_template('metrics.html', client_metrics=client_metrics)
# Flask route to start a Federated round
@app.route('/start_round', methods=['POST'])
def start_round():
    round_num = request.json.get("round", 1)  # Get round number from the client
    print(f"Starting round {round_num}...")
    federated_round(round_num)
    # Compute and log global model hash
    global_model_hash = get_model_hash(global_model.state_dict())
    log_global_model_update(global_model_hash, round_num)

    return jsonify({"message": f"Round {round_num} completed successfully!"})


# Check if the global model has been updated before evaluating
@app.route('/evaluate_global_model', methods=['POST'])
def evaluate_global_model():
    if not global_model_updated:
        return jsonify({"message": "Global model not updated yet. Evaluation unavailable."}), 400

    # Evaluation process (if the global model is updated)
    if not os.path.exists(EVALUATION_DATASET_PATH):
        return jsonify({"message": f"Dataset path {EVALUATION_DATASET_PATH} not found."}), 400

    preprocessor = Preprocessor(client_id="global_evaluation", data_dir=EVALUATION_DATASET_PATH)
    preprocessor.run()

    train_loader, val_loader, test_loader = get_data_loaders(client_id="global_evaluation")

    model = global_model
    model.eval()

    total_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return jsonify({
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })


# Start Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8282)
