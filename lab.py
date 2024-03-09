import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Part 1: Data Preparation
x = np.linspace(-100, 100, 1000)
y = x * np.sin(x**2 / 300)
x = x.reshape(-1, 1).astype(np.float32)
y = y.reshape(-1, 1).astype(np.float32)

x_tensor = torch.from_numpy(x)
y_tensor = torch.from_numpy(y)

# Part 2: Build Models
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=42)

x_train_tensor = torch.from_numpy(x_train)
y_train_tensor = torch.from_numpy(y_train)
x_test_tensor = torch.from_numpy(x_test)
y_test_tensor = torch.from_numpy(y_test)

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(model, x_train, y_train, x_test, y_test, epochs=100, batch_size=16):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # To train it
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # To test it
        model.eval()
        with torch.no_grad():
            outputs = model(x_test)
            loss = criterion(outputs, y_test)
            test_losses.append(loss.item())

        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

    return train_losses, test_losses

# Model 1
model_1 = Model1()
train_losses_1, test_losses_1 = train_model(model_1, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor)

# Model 2
model_2 = Model2()
train_losses_2, test_losses_2 = train_model(model_2, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor)

# Model 3
model_3 = Model3()
train_losses_3, test_losses_3 = train_model(model_3, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor)


# Part 3: Model Evaluation
def plot_results(model, x, y, title):
    with torch.no_grad():
        model.eval()
        plt.scatter(x, y, color='blue', label='Original data')
        plt.scatter(x, model(x_tensor).detach().numpy(), color='red', label='Predicted data')
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

plot_results(model_1, x, y, 'Model 1')

# Part 4: Get Model Output and Feedforward by Yourself
best_model = model_1

weights = [param.data.numpy() for param in best_model.parameters()]

sample_data = x_tensor[:5]

manual_outputs = torch.mm(sample_data, torch.from_numpy(weights[0]).T) + torch.from_numpy(weights[1])
for i in range(2, len(weights), 2):
    manual_outputs = torch.mm(manual_outputs, torch.from_numpy(weights[i]).T) + torch.from_numpy(weights[i + 1])

model_predictions = best_model(sample_data)
for i in range(len(sample_data)):
    print("Sample Data:", sample_data[i])
    print("Model Prediction:", model_predictions[i].item())
    print("Manual Output:", manual_outputs[i].item())
