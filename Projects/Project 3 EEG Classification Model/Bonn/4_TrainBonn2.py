import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Define the EEG Dataset class
class EEGDataset(Dataset):
    """ Custom Dataset class for EEG data with data augmentation. """
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        self.scaler = StandardScaler()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.loadtxt(self.file_paths[idx])
        # Data augmentation: Add random noise
        noise = np.random.normal(0, 0.1, data.shape)
        data += noise
        data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        data = torch.from_numpy(data).float()
        data = data.unsqueeze(0)
        label = self.labels[idx]
        return data, label

# Define the CNN model
class EnhancedCNN(nn.Module):
    """ More complex CNN model for EEG data classification with regularization. """
    def __init__(self, num_classes):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 1024, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

folders = ['F', 'N', 'O', 'S', 'Z']
file_paths = []
labels = []

for i, folder in enumerate(folders):
    for j in range(1, 101):  # Assuming there are 100 files per folder
        file_paths.append(f'C:/Users/surpriseX/Desktop/HW/6400/project3/preBonn/{folder}/{folder}{j:03d}.txt')
        labels.append(i)

# Split the dataset into training and testing sets
file_paths_train, file_paths_test, labels_train, labels_test = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42)

# Create the dataset objects for training and testing
train_dataset = EEGDataset(file_paths_train, labels_train)
test_dataset = EEGDataset(file_paths_test, labels_test)

# Create DataLoaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize the network
num_classes = 5  # Number of classes in the dataset
model = EnhancedCNN(num_classes=num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

loss_history = []  # To record the average loss for each epoch

# Train the model
def train(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        loss_history.append(average_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

# Test the model
def test(model, test_loader):
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            all_targets.extend(target.numpy())
            all_predictions.extend(predicted.numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')

    return accuracy, precision, recall, f1


# Run the training and testing
train(model, train_loader, criterion, optimizer, num_epochs=10)

# Plot the loss curve
plt.plot(loss_history)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

accuracy, precision, recall, f1 = test(model, test_loader)
print(f'Test Accuracy: {accuracy:.2f}')
print(f'Test Precision: {precision:.2f}')
print(f'Test Recall: {recall:.2f}')
print(f'Test F1 Score: {f1:.2f}')
