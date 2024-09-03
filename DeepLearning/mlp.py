from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix
from utlis_ import *
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split




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
train_df['ANYMEDS'] =train_df['ANYMEDS'].map(map_yes_no)
train_df['NACCACEI'] = train_df['NACCACEI'].map(map_yes_no)
train_df['NACCAAAS'] = train_df['NACCAAAS'].map(map_yes_no)
train_df['NACCBETA'] = train_df['NACCBETA'].map(map_yes_no)
train_df['NACCANGI'] = train_df['NACCANGI'].map(map_yes_no)
train_df['NACCLIPL'] =train_df['NACCLIPL'].map(map_yes_no)
train_df['NACCNSD'] = train_df['NACCNSD'].map(map_yes_no)
train_df['NACCAC'] = train_df['NACCAC'].map(map_yes_no)
train_df['NACCADEP'] = train_df['NACCADEP'].map(map_yes_no)
train_df['NACCEMD'] = train_df['NACCEMD'].map(map_yes_no)
train_df['HISPANIC'] = train_df['HISPANIC'].map(map_yes_no)

test_df['SEX'] = test_df['SEX'].map(map_sex)
test_df['NACCDAD'] = test_df['NACCDAD'].map(map_yes_no)
test_df['NACCMOM'] = test_df['NACCMOM'].map(map_yes_no)
test_df['ANYMEDS'] =test_df['ANYMEDS'].map(map_yes_no)
test_df['NACCACEI'] = test_df['NACCACEI'].map(map_yes_no)
test_df['NACCAAAS'] =test_df['NACCAAAS'].map(map_yes_no)
test_df['NACCBETA'] = test_df['NACCBETA'].map(map_yes_no)
test_df['NACCANGI'] =test_df['NACCANGI'].map(map_yes_no)
test_df['NACCLIPL'] =test_df['NACCLIPL'].map(map_yes_no)
test_df['NACCNSD'] = test_df['NACCNSD'].map(map_yes_no)
test_df['NACCAC'] = test_df['NACCAC'].map(map_yes_no)
test_df['NACCADEP'] =test_df['NACCADEP'].map(map_yes_no)
test_df['NACCEMD'] = test_df['NACCEMD'].map(map_yes_no)
test_df['HISPANIC'] = test_df['HISPANIC'].map(map_yes_no)

print("\nTraining set after solo thermal coding:")
print(train_df)

print("\nTest set after solo thermal encoding:")
print(test_df)



combined_df = pd.concat([train_df, test_df], ignore_index=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def add_gaussian_noise(data, mean=0, std=0.1):
    noise = np.random.normal(mean, std, data.shape)
    noisy_data = data + noise
    return noisy_data



y_train = train_df['dementia'].values
train_df = train_df.drop('dementia', axis=1)
X_train = train_df.values

y_test = test_df['dementia'].values
test_df = test_df.drop('dementia', axis=1)
X_test = test_df.values

train_df.reset_index(inplace=True, drop=False)
test_df.reset_index(inplace=True, drop=False)


original_index = combined_df.index


train_df['original_index'] = original_index[train_df.index]
test_df['original_index'] = original_index[test_df.index]





def preprocess_data(X, y):
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

   
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long) if y is not None else None

    return X_tensor, y_tensor


X_train_tensor, y_train_tensor = preprocess_data(X_train, y_train)
X_test_tensor, y_test_tensor = preprocess_data(X_test, y_test)


X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)



batch_size = 128

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_dim = X_train_tensor.shape[1]
hidden_dim = 64
output_dim = 2  # Binary classification, so 2 classes (0 and 1)

model = MLP(input_dim, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)



num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


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
    
    return accuracy, f1, recall, specificity, auc

accuracy, f1, sensitivity, specificity, auc = evaluate_model(model, test_loader)
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"AUC: {auc:.4f}")
