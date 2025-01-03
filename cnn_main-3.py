import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam



# Load the test data
test_df = pd.read_csv('fashion_mnist_test.csv')

x_test = test_df.drop(columns=['label']).values
y_test = test_df['label'].values

# Normalize and convert to PyTorch tensors
x_test = x_test / 255.0
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).reshape(-1, 1, 28, 28)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the CNN model class (same as above)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
model.load_state_dict(torch.load('cnn_model.pth'))
model.eval()

# Evaluate the model on the test data
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Convert lists to arrays
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

# Compute confusion matrix
conf_matrix_CNN = confusion_matrix(all_labels, all_predictions)

# Plot the confusion matrix
# plt.figure(figsize=(9, 6))
# plt.title("Confusion Matrix for CNN")
# sns.heatmap(conf_matrix_CNN, cmap="Blues", annot=True, fmt='g', xticklabels=True, yticklabels=True)
# plt.xlabel("Predicted")
# plt.yticks(rotation=360)
# plt.ylabel("Actual")
# plt.show()
print("conf_matrix is ",conf_matrix_CNN)


# Calculate metrics
test_accuracy = accuracy_score(all_labels, all_predictions)
test_precision = precision_score(all_labels, all_predictions, average="weighted")
test_recall = recall_score(all_labels, all_predictions, average="weighted")
test_f1 = f1_score(all_labels, all_predictions, average="weighted")

# Print metrics
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
