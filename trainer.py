import torch
import torch.optim as optim
import torch.nn as nn

# Simple function to train the model
def train_model(model, train_loader, val_loader, device, epochs=13, learning_rate=0.002):
    """
    Train the given model using the provided data.
    Args:
    - model (nn.Module): The model to train.
    - train_loader (DataLoader): DataLoader for training data.
    - val_loader (DataLoader): DataLoader for validation data.
    - device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu').
    - epochs (int): Number of training epochs.
    - learning_rate (float): Learning rate for the optimizer.
    
    Returns:
    - trained_model (nn.Module): The trained model after the specified epochs.
    """

    model.to(device)  # Move model to device (GPU or CPU)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Since this is a classification problem
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # Train on the training data
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print training statistics
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

        # Validation step (optional)
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():  # No gradient calculation needed for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return model  # Return the trained model
