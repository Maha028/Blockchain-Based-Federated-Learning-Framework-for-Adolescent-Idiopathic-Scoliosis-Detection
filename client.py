import torch
import requests
import base64
import pickle
from Model import ScoliosisCNN
from trainer import train_model 
from data_preprocessing import get_data_loaders, Preprocessor   # Importing the data loaders
import argparse
import torch.nn.functional as F
from sklearn.metrics import accuracy_score  # For accuracy calculation
from sklearn.metrics import precision_score, recall_score, f1_score

client_id = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
port = None

def register_client():
    client_info = {
        "client_id": client_id,
        "port": port
    }
    # Register the client with the server
    response = requests.post("http://127.0.0.1:8282/register_client", json=client_info)
    if response.status_code == 200:
        print(f"Client {client_id} registered successfully!")
    else:
        print(f"Error: {response.json()['message']}")

def preprocess_data():
    # First step: Run preprocessing to ensure data is ready
    print("Starting data preprocessing...")
    preprocessor = Preprocessor(client_id)
    preprocessor.run()  # This will handle preprocessing and save the processed data
    print("Data preprocessing completed.")

def calculate_accuracy(model, val_loader):
    model.eval()  # Set model to evaluation mode
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy

def save_local_model(model):
    # Save the model locally before sending to the server
    model_path = f"local_model_{client_id}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Local model saved as {model_path}")
    return model_path


def train_on_client():
    # Send a request to the server to get the global model
    server_url = "http://127.0.0.1:8282/get_global_model"  # Ensure this endpoint exists on the server
    try:
        response = requests.get(server_url)  # Get the global model from the server
        response.raise_for_status()  # Raise an exception for any HTTP error codes

        # Get the model data from the response
        data = response.json()  # Correctly access the JSON response
        model_bytes = base64.b64decode(data["model"])  # Decode the base64 model
        model_state = pickle.loads(model_bytes)  # Deserialize the model state

        # Initialize the model and load the state
        model = ScoliosisCNN().to(device)
        model.load_state_dict(model_state)

        # Load the data
        train_loader, val_loader, test_loader = get_data_loaders(client_id)

        # Train the model locally
        trained_model = train_model(model, train_loader, val_loader, device)

        # Calculate accuracy after local training
        train_accuracy = calculate_accuracy(trained_model, train_loader)
        val_accuracy = calculate_accuracy(trained_model, val_loader)
        test_accuracy = calculate_accuracy(trained_model, test_loader)

        # Calculate evaluation metrics (Precision, Recall, F1-Score)
        true_labels = []
        predicted_labels = []
        
        # Get predictions for validation set
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(preds.cpu().numpy())

        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)

        # Get number of samples used for training
        num_samples = len(train_loader.dataset)

        # Save the local model update
        local_model_path = save_local_model(trained_model)

        # Serialize the updated model and send it back to the server
        updated_weights = pickle.dumps(trained_model.state_dict())
        encoded = base64.b64encode(updated_weights).decode()

        # Send the updated model, accuracy, and evaluation metrics back to the server
        update_url = "http://127.0.0.1:8282/update_model"  # Endpoint to send the updated model
        update_data = {
            "client_id": client_id,
            "updated_model": encoded,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "num_samples": num_samples  # Include the number of samples used for training
        }
        update_response = requests.post(update_url, json=update_data)

        if update_response.status_code == 200:
            print(f"Successfully sent updated model and accuracy from {client_id} to the server.")
        else:
            print(f"Failed to send updated model. Status code: {update_response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while communicating with the server: {e}")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="Client ID (e.g., client1, client2)")
    parser.add_argument("--port", type=int, required=True, help="Port for the client")
    args = parser.parse_args()

    client_id = args.id
    port = args.port
    print(f"🚀 Starting client '{client_id}' on port {port}")

    # Register the client with the server
    #register_client()

    # Perform preprocessing first
    preprocess_data()

    # Train the model locally (without Flask)
    train_on_client()
