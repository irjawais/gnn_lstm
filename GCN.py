import numpy as np
from imblearn.over_sampling import SMOTE
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
import networkx as nx
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def create_adj_matrix(graph):
    num_nodes = len(graph)
    adj_matrix = [[0 for i in range(num_nodes)] for j in range(num_nodes)]

    for node in graph:
        for neighbor in graph[node]:
            adj_matrix[node][neighbor] = 1
    return adj_matrix


''' # Define the GCN model
class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        self.conv3 = GCNConv(out_channels, out_channels)
        self.conv4 = GCNConv(out_channels, out_channels)
        self.conv5 = GCNConv(out_channels, out_channels)
        self.fc = nn.Linear(out_channels, 2)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv5(x, edge_index)
        return self.fc(x) '''
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.conv4(x, edge_index)
        return x

class LSTM(nn.Module):
    def __init__(self, in_features, hidden_size, num_classes):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=1)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        x = self.fc(h_n[-1, :, :])
        return x

class GCN_LSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hidden_size, num_classes):
        super(GCN_LSTM, self).__init__()
        self.gcn = GCN(in_channels, hidden_channels, out_channels)
        self.lstm = LSTM(out_channels, hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        x = torch.unsqueeze(x, 0)
        x = self.lstm(x)
        return x
        
def create_graph(X, corr_threshold):
  corr = np.corrcoef(X)

  # Create an empty graph
  G = nx.Graph()
  # Add each row as a node in the graph
  corr_threshold = 0.5
  for i in range(X.shape[0]):
      G.add_node(i)
  # Iterate over the rows and add edges between nodes with correlation above the threshold
  for i in range(X.shape[0]):
      for j in range(X.shape[0]):
          if i == j:
              continue
          if corr[i, j] > corr_threshold:
              G.add_edge(i, j)
  return  G
  

each_channel_features = np.load('content/each_channel_features.npy', allow_pickle=True)
each_channel_features = np.nan_to_num(each_channel_features, nan=0, posinf=0, neginf=0)

print(each_channel_features.shape)

selected_1_rows = each_channel_features[each_channel_features[:, -1] == 1]
selected_0_rows = each_channel_features[each_channel_features[:, -1] == 0]
selected_0_rows = selected_0_rows[:100]
merged_rows = np.concatenate((selected_1_rows, selected_0_rows), axis=0)

#each_channel_features = each_channel_features[10000:20000]

print(merged_rows.shape)

X = merged_rows[:, :-1]
y = merged_rows[:, -1]


smote = SMOTE(sampling_strategy='minority')
X, y = smote.fit_resample(X, y)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

print(X.shape,y.shape)


G = create_graph(X, 0.5)




print(G.number_of_nodes())
adj = create_adj_matrix(G)
adj = np.array(adj)
labels = y
labels = labels.astype(int)
labels = torch.from_numpy(labels).long()
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(adj, labels, test_size=0.01, random_state=42)

# Create the GCN model
model = GCN(in_channels=G.number_of_nodes(), out_channels=16)

num_nodes = X_train.shape[0]
valid_indices = np.where((X_train.nonzero()[0] < num_nodes) & (X_train.nonzero()[1] < num_nodes))
edge_index = torch.from_numpy(np.concatenate((X_train.nonzero()[0][valid_indices][np.newaxis, :],
                                                X_train.nonzero()[1][valid_indices][np.newaxis, :]), axis=0))
                                                
adj = torch.from_numpy(adj)
X_train = torch.from_numpy(X_train)

print(X_train.shape,G.number_of_nodes(),edge_index.shape)
accuracies = []
optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
y_train_pred = []
for epoch in range(1000):
    optimizer.zero_grad()
    out = model(X_train.float(), edge_index)
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()
    _, predicted = torch.max(out.data, 1)
    correct = (predicted == y_train).sum().item()
    accuracy = correct / len(y_train)
    accuracies.append(accuracy)
    y_train_pred.append(predicted)
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

# Convert the predictions to numpy array
y_train_pred = np.array(y_train_pred[-1])

# Calculate the confusion matrix
cm = confusion_matrix(y_train, y_train_pred)

# Plot the confusion matrix
plt.imshow(cm, cmap='binary')
plt.colorbar()
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.show()
