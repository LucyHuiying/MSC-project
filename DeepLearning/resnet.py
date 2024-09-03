import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dff4.csv')

# 查看处理后的 DataFrame
print("DataFrame:")
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, self.expansion * out_channels),
                nn.BatchNorm1d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = out.view(out.size(0), -1)  
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


y_train = train_df['dementia'].values
X_train = train_df.drop('dementia', axis=1).values

y_test = test_df['dementia'].values
X_test = test_df.drop('dementia', axis=1).values

input_dim = X_train.shape[1]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)


batch_size = 128
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = ResNet18().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


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
def evaluate_model(model, data_loader):
    model.eval()
    y_true = []
    y_scores = []  

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            
           
            probabilities = torch.softmax(outputs, dim=1)
            
            positive_class_probabilities = probabilities[:, 1].cpu().numpy()
            y_scores.extend(positive_class_probabilities)
            
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.cpu().numpy())
    
    
    y_scores_array = np.array(y_scores)
    
    accuracy = accuracy_score(y_true, y_scores_array > 0.5)  
    f1 = f1_score(y_true, y_scores_array > 0.5)
    recall = recall_score(y_true, y_scores_array > 0.5)
    tn, fp, fn, tp = confusion_matrix(y_true, y_scores_array > 0.5).ravel()
    specificity = tn / (tn + fp)
    auc = roc_auc_score(y_true, y_scores_array) 

    # Plot confusion matrix with proportions using 'plasma' colormap
    cm = confusion_matrix(y_true, y_scores_array > 0.5)
    cm_proportions = cm / cm.sum()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_proportions.T, annot=True, fmt=".2f", cmap="plasma", cbar=True,
                annot_kws={"size": 16}, xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title("Confusion Matrix - Resnet", fontsize=18)
    plt.xlabel("Actual", fontsize=14)  # Changed to "Actual"
    plt.ylabel("Predicted", fontsize=14)  # Changed to "Predicted"
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Save the confusion matrix plot
    plt.savefig("confusion_resnet.png")
    plt.close()

    return accuracy, f1, recall, specificity, auc

accuracy, f1, sensitivity, specificity, auc = evaluate_model(model, test_loader)
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"AUC: {auc:.4f}")
