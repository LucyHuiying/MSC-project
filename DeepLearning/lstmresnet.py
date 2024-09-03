import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import seaborn as sns


df = pd.read_csv('dff.csv')


print("处理后的 DataFrame:")
print(df)

train_df = df[df['set'] == 1]
test_df = df[df['set'] == 2]

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed(2025)


map_sex = {'Female': 0, 'Male': 1}
map_yes_no = {'No': 0, 'Yes': 1}
map_yes_no1 = {'Training': 0, 'Validation': 1}
map_yes_no2 = {'I': 0, 'IT': 1}


train_df['SEX'] = train_df['SEX'].map(map_sex)
train_df['NACCDAD'] = train_df['NACCDAD'].map(map_yes_no)
train_df['NACCMOM'] = train_df['NACCMOM'].map(map_yes_no)
train_df['ANYMEDS'] = train_df['ANYMEDS'].map(map_yes_no)
train_df['NACCACEI'] = train_df['NACCACEI'].map(map_yes_no)
train_df['NACCAAAS'] = train_df['NACCAAAS'].map(map_yes_no)
train_df['NACCBETA'] = train_df['NACCBETA'].map(map_yes_no)
train_df['NACCANGI'] = train_df['NACCANGI'].map(map_yes_no)
train_df['NACCLIPL'] = train_df['NACCLIPL'].map(map_yes_no)
train_df['NACCNSD'] = train_df['NACCNSD'].map(map_yes_no)
train_df['NACCAC'] = train_df['NACCAC'].map(map_yes_no)
train_df['NACCADEP'] = train_df['NACCADEP'].map(map_yes_no)
train_df['NACCEMD'] = train_df['NACCEMD'].map(map_yes_no)
train_df['HISPANIC'] = train_df['HISPANIC'].map(map_yes_no)

test_df['SEX'] = test_df['SEX'].map(map_sex)
test_df['NACCDAD'] = test_df['NACCDAD'].map(map_yes_no)
test_df['NACCMOM'] = test_df['NACCMOM'].map(map_yes_no)
test_df['ANYMEDS'] = test_df['ANYMEDS'].map(map_yes_no)
test_df['NACCACEI'] = test_df['NACCACEI'].map(map_yes_no)
test_df['NACCAAAS'] = test_df['NACCAAAS'].map(map_yes_no)
test_df['NACCBETA'] = test_df['NACCBETA'].map(map_yes_no)
test_df['NACCANGI'] = test_df['NACCANGI'].map(map_yes_no)
test_df['NACCLIPL'] = test_df['NACCLIPL'].map(map_yes_no)
test_df['NACCNSD'] = test_df['NACCNSD'].map(map_yes_no)
test_df['NACCAC'] = test_df['NACCAC'].map(map_yes_no)
test_df['NACCADEP'] = test_df['NACCADEP'].map(map_yes_no)
test_df['NACCEMD'] = test_df['NACCEMD'].map(map_yes_no)
test_df['HISPANIC'] = test_df['HISPANIC'].map(map_yes_no)


print("\nTraining set after solo thermal coding:")
print(train_df)

print("\nTest set after solo thermal encoding:")
print(test_df)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 注意这里输入通道数为64
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], 2)
        self.layer3 = self.make_layer(block, 256, layers[2], 2)
        self.layer4 = self.make_layer(block, 512, layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  
        x = x.view(x.size(0), -1)  
  
        assert x.size(1) == 512, "ResNet output shape mismatch"
        x = self.fc(x)
        return x


class LSTMResNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMResNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 64)  
        self.resnet = ResNet(ResidualBlock, [2, 2, 2, 2])
        self.fc = nn.Linear(66, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
    
        out, _ = self.lstm(x, (h0, c0))
        lstm_out = out[:, -1, :]  
    
        
        lstm_out = self.linear(lstm_out) 
        resnet_input = lstm_out.unsqueeze(-1)  
        resnet_out = self.resnet(resnet_input)
    
       
        resnet_out = resnet_out.squeeze(-1) 
    
       
        out = torch.cat((lstm_out, resnet_out), dim=1)
        out = self.fc(out)
        return out


y_train = train_df['dementia'].values
X_train = train_df.drop('dementia', axis=1).values

y_test = test_df['dementia'].values
X_test = test_df.drop('dementia', axis=1).values

input_dim = X_train.shape[1]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).view(-1, 1, input_dim).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).view(-1, 1, input_dim).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)


batch_size = 128
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = input_dim
hidden_size = 128
num_layers = 2
num_classes = 2
model = LSTMResNet(input_size, hidden_size, num_layers, num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")



import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
def plot_cdf(y_scores, y_true, y_preds):
    
    dementia_scores = y_scores[y_true == 1]
    no_dementia_scores = y_scores[y_true == 0]
    misclassified_scores = y_scores[y_true != y_preds]

    def compute_cdf(data):
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(data) + 1) / len(data)
        return sorted_data, cdf

   
    dementia_sorted, dementia_cdf = compute_cdf(dementia_scores)
    no_dementia_sorted, no_dementia_cdf = compute_cdf(no_dementia_scores)
    misclassified_sorted, misclassified_cdf = compute_cdf(misclassified_scores)

    
    plt.figure(figsize=(8, 6))
    plt.plot(dementia_sorted, dementia_cdf, label='Dementia', color='blue')
    plt.plot(no_dementia_sorted, no_dementia_cdf, label='No dementia', color='orange')
    plt.plot(misclassified_sorted, misclassified_cdf, label='Misclassified', color='green')

    plt.xlabel('Score')
    plt.ylabel('CDF')
    plt.title('LSTM+Resnet')
    plt.legend()
    plt.grid(True)
    plt.savefig("cdf_plot_lstmresnet.png")
    plt.close()
def evaluate_model(model, data_loader):
    model.eval()
    y_true = []
    y_scores = []
    y_preds = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            positive_class_probabilities = probabilities[:, 1].cpu().numpy()
            y_scores.extend(positive_class_probabilities)
            
            _, predicted = torch.max(outputs.data, 1)
            y_preds.extend(predicted.cpu().numpy())
            
            y_true.extend(labels.cpu().numpy())

   
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_preds = np.array(y_preds)

  
    accuracy = accuracy_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
    specificity = tn / (tn + fp)
    auc = roc_auc_score(y_true, y_scores)
    # Plot confusion matrix with proportions using 'plasma' colormap
    cm = confusion_matrix(y_true, y_preds)
    cm_proportions = cm / cm.sum()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_proportions.T, annot=True, fmt=".2f", cmap="plasma", cbar=True,
                annot_kws={"size": 16}, xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title("Confusion Matrix - LSTM+Resnet", fontsize=18)
    plt.xlabel("Actual", fontsize=14)  # Changed to "Actual"
    plt.ylabel("Predicted", fontsize=14)  # Changed to "Predicted"
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Save the confusion matrix plot
    plt.savefig("confusion_lstm.png")
    plt.close()

   
    plot_cdf(y_scores, y_true, y_preds)

    return accuracy, f1, recall, specificity, auc

accuracy, f1, sensitivity, specificity, auc = evaluate_model(model, test_loader)
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"AUC: {auc:.4f}")