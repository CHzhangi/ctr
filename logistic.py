import torch
import torch.nn as nn
import torch.optim as optim

# Generate some random data
torch.manual_seed(42)
X = torch.randn(100, 2)
y = torch.randint(0, 2, (100,), dtype=torch.float32)

# Define the logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out

# Initialize the model
input_size = 2
model = LogisticRegression(input_size)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    outputs = model(X)

    # Compute the loss
    loss = criterion(outputs.view(-1), y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the trained model
with torch.no_grad():
    # Generate some test data
    X_test = torch.randn(10, 2)

    # Get predictions
    predictions = model(X_test)
    predictions = (predictions > 0.5).float()  # Convert to binary predictions (0 or 1)

    print("Test Predictions:")
    print(predictions)
