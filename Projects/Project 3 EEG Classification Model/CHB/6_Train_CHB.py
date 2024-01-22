import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Data loading
def load_data(directory):
    data = []
    labels = []
    for folder in ['nonseizure', 'seizure']:
        folder_path = os.path.join(directory, folder)
        for file in os.listdir(folder_path):
            if file.endswith('.npy') and not file.endswith('_label.npy'):
                file_path = os.path.join(folder_path, file)
                signal_data = np.load(file_path).astype(np.float32)
                label_data = np.load(file_path.replace('.npy', '_label.npy'))
                data.append(signal_data)

                # Use any() to determine whether there is 1
                label = 1 if np.any(label_data == 1) else 0
                labels.append(label)

    return data, labels

data_directory = "Data_CHB"
signals, labels = load_data(data_directory)
# print(f"First 10 labels: {labels[:10]}")

lengths = [len(signal) for signal in signals]

# Data preprocessing: padding or truncation to unify signal length
def pad_sequences(sequences, max_len, padding_value=0):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            padded_seq = np.pad(seq, (0, max_len - len(seq)), 'constant', constant_values=(padding_value,))
        else:
            padded_seq = seq[:max_len]
        padded_sequences.append(padded_seq)
    return np.array(padded_sequences)

max_len = max([len(signal) for signal in signals])  # Select the length of the longest signal
signals = pad_sequences(signals, max_len)

# print(max_len)

# Feature extraction
def extract_features(signals):
    features = []
    for signal in signals:
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        fft_vals = rfft(signal)
        fft_freq = rfftfreq(len(signal), 1 / 256)  # 采样率为256Hz
        fft_amp = np.abs(fft_vals)
        dominant_freq_amp = np.max(fft_amp)
        features.append([mean_val, std_val, dominant_freq_amp])
    return np.array(features)

features = extract_features(signals)

# Convert to PyTorch tensors
# Use raw EEG signal
signals_tensor = torch.tensor(signals).float()
labels_tensor = torch.tensor(labels).long()

# Use extracted features
# features_tensor = torch.tensor(features).float()
# labels_tensor = torch.tensor(labels).long()

# Dataset split
dataset = TensorDataset(signals_tensor, labels_tensor)
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
# Reduce batch size
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 从64减少到32
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def check_data_loading(directory):
    for folder in ['nonseizure', 'seizure']:
        folder_path = os.path.join(directory, folder)
        if not os.path.exists(folder_path):
            print(f"Directory does not exist: {folder_path}")
        else:
            file_count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
            print(f"{folder} Number of files: {file_count}")

# check_data_loading(data_directory)

def check_labels(directory):
    labels_check = []
    for folder in ['nonseizure', 'seizure']:
        folder_path = os.path.join(directory, folder)
        for file in os.listdir(folder_path):
            if file.endswith('_label.npy'):
                label_data = np.load(os.path.join(folder_path, file))
                labels_check.append(np.mean(label_data))
    # print(f"Label sample: {labels_check[:10]}")
    # print(f"different label values: {set(labels_check)}")

# check_labels(data_directory)


# print(f"First 10,000 labels: {labels[:10000]}")

# def check_sequence_lengths(signals):
    # lengths = [len(signal) for signal in signals]
    # print(f"Maximum signal length: {max(lengths)}")
    # print(f"Minimum signal length: {min(lengths)}")

# check_sequence_lengths(signals)

# sample_features = extract_features(signals[:5])
# print(f"Feature sample: {sample_features}")

# Neural network definition
class CNN1D(nn.Module):
    def __init__(self, input_size):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(3715072, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# GPU settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_size = max_len  # Use padded or truncated length
model = CNN1D(input_size).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# print(model)

# input size
input_size = 464384

# The first convolutional layer
conv1_output_size = (input_size - 3 + 2 * 1) // 1 + 1
# The first pooling layer
pool1_output_size = conv1_output_size // 2

# The second convolutional layer
conv2_output_size = (pool1_output_size - 3 + 2 * 1) // 1 + 1
# The second pooling layer
pool2_output_size = conv2_output_size // 2

# output size
final_output_size = pool2_output_size * 32  # 32 is the number of output channels of the last convolutional layer

# print(f'Output size after convolutional and pooling layers: {final_output_size}')

# Make sure this is consistent with the fc1 input size in the model

# print(f"Number of training set samples: {len(train_dataset)}")
# print(f"Number of validation set samples: {len(val_dataset)}")
# print(f"Number of test set samples: {len(test_dataset)}")

# training loop
num_epochs = 20
loss_history = []  # Used to store the loss of each epoch

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    loss_history.append(average_loss)  # Record average loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}')

    # Save the model of the current epoch
    torch.save(model.state_dict(), f'CHBmodel/model_epoch_{epoch+1}.pth')

# Draw loss curve
plt.plot(loss_history)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Model evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Validation Accuracy: {100 * correct / total}%')

# Model Evaluation
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')

# Print metrics
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')