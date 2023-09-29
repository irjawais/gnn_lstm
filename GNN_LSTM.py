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
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sns
from scipy.spatial.distance import euclidean
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score, roc_auc_score, auc


def create_adj_matrix(graph):
    num_nodes = len(graph)
    adj_matrix = [[0 for i in range(num_nodes)] for j in range(num_nodes)]

    for node in graph:
        for neighbor in graph[node]:
            adj_matrix[node][neighbor] = 1
    return adj_matrix

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = nn.Linear(in_channels, hidden_channels)
        self.conv2 = nn.Linear(hidden_channels, hidden_channels)
        self.conv3 = nn.Linear(hidden_channels, hidden_channels)
        self.conv4 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)
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

class GNN_LSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hidden_size, num_classes):
        super(GNN_LSTM, self).__init__()
        self.gnn = GNN(in_channels, hidden_channels, out_channels)
        self.lstm = LSTM(out_channels, hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = self.gnn(x, edge_index)
        x = torch.unsqueeze(x, 0)
        x = self.lstm(x)
        return x
        
def create_graph(X, y):

    # Create an empty graph
    G = nx.Graph()
    # Add each row as a node in the graph
    for i in range(X.shape[0]):
        G.add_node(i)
    # Loop through all nodes to create edges
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            if i == j:
                continue
            dist = euclidean(X[i], X[j])

            if dist < 0.2 and  y[i] == y[j] : #
                G.add_edge(i, j)
    ''' # Iterate over all node pairs
    for i in range(X.shape[0]):
        n1 = tuple(X[i])
        for j in range(i+1, X.shape[0]):
            n2 = tuple(X[j])
            avg_diff = np.abs(np.mean(n1) - np.mean(n2))
            #print(avg_diff)
            if avg_diff < 0.15:
                G.add_edge(n1, n2) '''
    return  G
  

each_channel_features = np.load('content/each_channel_features.npy', allow_pickle=True)
each_channel_features = np.nan_to_num(each_channel_features, nan=0, posinf=0, neginf=0)

print(each_channel_features.shape)

selected_1_rows = each_channel_features[each_channel_features[:, -1] == 1]
selected_0_rows = each_channel_features[each_channel_features[:, -1] == 0]
selected_0_rows = selected_0_rows[:500]
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


G = create_graph(X,y)

adj = nx.adjacency_matrix(G)
adj = adj.todense()
adj = np.array(adj)

#print(G.number_of_nodes(),"adj matrix here",adj)
labels = y

labels = labels.astype(int)
labels = torch.from_numpy(labels).long()
print(len(adj),len(labels))
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(adj, labels, test_size=0.01, random_state=42)

# Create the GCN + LSTM model
model = GNN_LSTM(in_channels=X.shape[1], hidden_channels=120, out_channels=16, hidden_size=64, num_classes=2)

                                               
adj = torch.from_numpy(adj)
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)

accuracies = []
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
print(X_train.shape,adj.shape)
y_train_pred = []
for epoch in range(1500):
    optimizer.zero_grad()
    out = model(X_train.float(), adj)
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()
    _, predicted = torch.max(out.data, 1)
    correct = (predicted == y_train).sum().item()
    accuracy = correct / len(y_train)
    recall = recall_score(y_train, predicted)
    precision = precision_score(y_train, predicted)
    f1 = f1_score(y_train, predicted)
    sensitivity = recall

    k = cohen_kappa_score(y_train, predicted)
    auc_value = roc_auc_score(y_train, predicted)
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1-measure: {f1:.4f}, Sensitivity: {sensitivity:.4f},  k-cohen: {k:.4f}, AUC: {auc_value:.4f}')



accuracies = np.array(accuracies)
x=np.array(range(1500))
# plotting
plt.title("Line graph")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(x, accuracies, linestyle='dashed', color='mediumvioletred')
plt.show()

num_nodes = X_test.shape[0]
valid_indices = np.where((X_test.nonzero()[0] < num_nodes) & (X_test.nonzero()[1] < num_nodes))
edge_index = torch.from_numpy(np.concatenate((X_test.nonzero()[0][valid_indices][np.newaxis, :],
                                                X_test.nonzero()[1][valid_indices][np.newaxis, :]), axis=0))

# Evaluate the model on the test data
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    correct = 0
    total = 0
    outputs = model(X_test.float(), edge_index)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == y_test).sum().item()
        
# Print the test accuracy
print('Accuracy of the model on the test data: {}'.format(correct / total))



# Convert the predictions to numpy array
''' y_train_pred = np.array(y_train_pred[-1])

# Calculate the confusion matrix
cm = confusion_matrix(y_train, y_train_pred)

# Plot the confusion matrix
plt.imshow(cm, cmap='binary')
plt.colorbar()
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.show() '''
