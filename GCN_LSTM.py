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
from torchviz import make_dot
import netron
import scikitplot as skplt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import plotly.graph_objects as go

def create_adj_matrix(graph):
    num_nodes = len(graph)
    adj_matrix = [[0 for i in range(num_nodes)] for j in range(num_nodes)]

    for node in graph:
        for neighbor in graph[node]:
            adj_matrix[node][neighbor] = 1
    return adj_matrix

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.conv4(x, edge_index)
        x = torch.relu(x)
        x = self.conv5(x, edge_index)
        x = torch.relu(x)
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
        
''' def create_graph(X, y):

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

            if dist < 0.5 : #
                G.add_edge(i, j)
    return  G '''
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

            if dist < 0.2 : #
                G.add_edge(i, j)
    return  G

each_channel_features = np.load('content/each_channel_features (siena).npy', allow_pickle=True)
each_channel_features = np.nan_to_num(each_channel_features, nan=0, posinf=0, neginf=0)

''' each_channel_features_2 = np.load('content/each_channel_features_2.npy', allow_pickle=True)
each_channel_features_2 = np.nan_to_num(each_channel_features_2, nan=0, posinf=0, neginf=0)
each_channel_features = np.concatenate((each_channel_features, each_channel_features_2), axis=0) '''

print(each_channel_features.shape)

selected_1_rows = each_channel_features[each_channel_features[:, -1] == 1]
selected_0_rows = each_channel_features[each_channel_features[:, -1] == 0]
#selected_0_rows = selected_0_rows[:100]
merged_rows = np.concatenate((selected_1_rows, selected_0_rows), axis=0)

#each_channel_features = each_channel_features[10000:20000]

print(merged_rows.shape)

X = merged_rows[:, :-1]
y = merged_rows[:, -1]

from knnor import data_augment
knnor = data_augment.KNNOR()
X, y, X_aug_min, y_aug_min = knnor.fit_resample(X,y)

''' smote = SMOTE(sampling_strategy='minority')
X, y = smote.fit_resample(X, y) '''

''' print("befiore ",merged_rows.shape)
new_data = np.loadtxt('new_data.csv', delimiter=',')
zeros_col = np.zeros((new_data.shape[0], 1))
new_data = np.hstack((new_data, zeros_col))
merged_rows = np.vstack((each_channel_features, new_data))
print("after ",merged_rows.shape) '''



scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
selection = SelectFromModel(rf_model, threshold='median', prefit=True)
X_train_selected = selection.transform(X_train)
X_test_selected = selection.transform(X_test)


''' X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=100)
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)
#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy----->:",metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show() '''


print(X.shape,y.shape)


#G = create_graph(X,y)
G = create_graph(X_train_selected,y_train)
labels = y_train
''' nx.draw(G, with_labels=True)
plt.show() '''
''' nx.draw(G, with_labels=True, node_color=y_train)
plt.show() '''





adj = nx.adjacency_matrix(G)
adj = adj.todense()
adj = np.array(adj)

#print(G.number_of_nodes(),"adj matrix here",adj)


labels = labels.astype(int)
labels = torch.from_numpy(labels).long()
print(len(adj),len(labels))

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(adj, labels, test_size=0.01, random_state=42)
print("===>",G.number_of_nodes())
# Create the GCN + LSTM model
model = GCN_LSTM(in_channels=G.number_of_nodes(), hidden_channels=120, out_channels=16, hidden_size=120, num_classes=2)
print(model)
from torchsummary import summary
summary(model)
num_nodes = X_train.shape[0]
valid_indices = np.where((X_train.nonzero()[0] < num_nodes) & (X_train.nonzero()[1] < num_nodes))
edge_index = torch.from_numpy(np.concatenate((X_train.nonzero()[0][valid_indices][np.newaxis, :],
                                                X_train.nonzero()[1][valid_indices][np.newaxis, :]), axis=0))
                                                
adj = torch.from_numpy(adj)
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)

''' dot = make_dot(model(X_train.float(), edge_index), params=dict(model.named_parameters()))
dot.render(filename='GCN_LSTM', engine='/usr/share/zsh/5.3/help/dot') '''
''' inputs = (X_train.float(), edge_index)
torch.onnx.export(model, inputs, "GCN_LSTM.onnx")
netron.start("GCN_LSTM.onnx") '''

print(X_train.shape,G.number_of_nodes(),edge_index.shape)
accuracies = []
recalls = []
precisions = []
losses = []

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

y_train_pred = []
for epoch in range(2000):
    optimizer.zero_grad()
    out = model(X_train.float(), edge_index)
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    _, predicted = torch.max(out.data, 1)
    correct = (predicted == y_train).sum().item()
    accuracy = correct / len(y_train)
    accuracies.append(accuracy)
    recall = recall_score(y_train, predicted)
    recalls.append(recall)
    precision = precision_score(y_train, predicted)
    f1 = f1_score(y_train, predicted)
    sensitivity = recall



    k = cohen_kappa_score(y_train, predicted)
    auc_value = roc_auc_score(y_train, predicted)
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1-measure: {f1:.4f}, Sensitivity: {sensitivity:.4f},  k-cohen: {k:.4f}, AUC: {auc_value:.4f}')

out = out.detach().numpy()
y_train = y_train.detach().numpy()

pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(out)
plt.scatter(embeddings_pca[:,0], embeddings_pca[:,1], c=y_train)
plt.show()
pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(out)
plt.scatter(embeddings_pca[:,0], embeddings_pca[:,1], c=y_train)
plt.show()

tsne = TSNE(n_components=2)
embeddings_tsne = tsne.fit_transform(out)
plt.scatter(embeddings_tsne[:,0], embeddings_tsne[:,1], c=y_train)
plt.show()





# Save the trained model
torch.save(model.state_dict(), "trained_gcn_lstm_model.pth")
torch.save(edge_index, "edge_index.pth")


accuracies = np.array(accuracies)
x=np.array(range(5000))
# plotting
plt.title("GCN-LSTM Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(x, accuracies, color='mediumvioletred')
plt.show()

losses = np.array(losses)
x=np.array(range(5000))
# plotting
plt.title("GCN-LSTM Model Loss graph")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(x, losses, color='mediumvioletred')
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
y_test_pred = predicted.numpy()
y_test = np.array(y_test)
y_test_pred = np.array(predicted)
print(y_test_pred.shape,y_test.shape)
matrix_confusion = confusion_matrix(y_test,y_test_pred, normalize='true')
print(matrix_confusion)
skplt.metrics.plot_confusion_matrix(y_test,y_test_pred, normalize=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.show()


''' each_channel_features_2 = np.load('content/each_channel_features_2.npy', allow_pickle=True)
each_channel_features_2 = np.nan_to_num(each_channel_features_2, nan=0, posinf=0, neginf=0)
X = each_channel_features_2[:, :-1]
labels = y = each_channel_features_2[:, -1]
X = torch.from_numpy(X)

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    outputs = model(X.float())
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == y).sum().item()
# Print the test accuracy
print('Accuracy of the model on the test data: {}'.format(correct / total)) '''
