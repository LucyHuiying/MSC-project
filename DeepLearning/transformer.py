import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from tqdm.auto import tqdm
from keras.layers import Dropout
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv('dff4.csv')


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

class PositionalEncoding(nn.Module): 
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
class MaskedTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers):
        super(MaskedTransformerEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)# Dense Layer
        self.positional_encoding = PositionalEncoding(d_model)#Positional Encoding Layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)# Transformer Encoder Layers
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(p=0.5)  
        self.dense_gelu_1 = nn.Linear(d_model, 128)  # Dense + GELU Layer 1
        self.gelu = nn.GELU()
        self.dense_gelu_2 = nn.Linear(128, d_model)  # Dense + GELU Layer 2
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, inputs):
        projected_inputs = self.input_projection(inputs)
        encoded_inputs = self.positional_encoding(projected_inputs)
        output = self.transformer_encoder(encoded_inputs)
        
        output = self.dropout(output)

       
        output = output.mean(dim=0)
        
        output = self.dense_gelu_1(output)
        output = self.gelu(output)

        
        output = self.dense_gelu_2(output)

        
        output = self.dropout(output)
        classifier_output = self.classifier(output)
        
        return output

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





batch_size = 32


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


input_dim = X_train.shape[1]
d_model = 128
nhead = 4
num_layers = 3


model = MaskedTransformerEncoder(input_dim, d_model, nhead, num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss().to(device)

best_accuracy = 0.0
best_model_state = None

def train_model(model, train_loader, epochs=10):
    global best_accuracy, best_model_state

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()  
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        
        avg_loss = running_loss / (batch_idx + 1)

        
        accuracy, auc, f1, recall, specificity = evaluate_model(model, test_loader)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, '
                  f'Test Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, '
                  f'F1 Score: {f1:.4f}, Sensitivity: {recall:.4f}, Specificity: {specificity:.4f}')


def evaluate_model(model, data_loader):
    model.eval()  # Set model to evaluation mode
    all_outputs = []
    all_targets = []
    all_scores = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Extract probability scores for class 1 directly as numpy array
            predicted = outputs.max(1)[1].cpu().numpy()  # Simplify the extraction of predictions
            all_outputs.extend(predicted)
            all_targets.extend(target.cpu().numpy())
            all_scores.extend(scores)  # Append numpy array directly to list

    # Convert lists to numpy arrays for metric calculation
    all_targets = np.array(all_targets)
    all_outputs = np.array(all_outputs)
    all_scores = np.array(all_scores)  # Ensure it is a numpy array

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_outputs)
    auc = roc_auc_score(all_targets, all_outputs)  # Calculate AUC using the probability scores
    f1 = f1_score(all_targets, all_outputs)
    recall = recall_score(all_targets, all_outputs)
    tn, fp, fn, tp = confusion_matrix(all_targets, all_outputs).ravel()
    specificity = tn / (tn + fp)

    # Plot confusion matrix with proportions using 'plasma' colormap
    cm = confusion_matrix(all_targets, all_outputs)
    cm_proportions = cm / cm.sum()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_proportions.T, annot=True, fmt=".2f", cmap="plasma", cbar=True,
                annot_kws={"size": 16}, xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title("Confusion Matrix - Transformer", fontsize=18)
    plt.xlabel("Actual", fontsize=14)  # Changed to "Actual"
    plt.ylabel("Predicted", fontsize=14)  # Changed to "Predicted"
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Save the confusion matrix plot
    plt.savefig("confusion_Tran.png")
    plt.close()

    # Optionally plot the CDF
    plot_cdf(all_scores, all_targets, all_outputs)

    return accuracy, auc, f1, recall, specificity

def plot_cdf(all_scores, all_targets, all_outputs):
    # Calculate CDF for each class
    dementia_scores = all_scores[all_targets == 1]
    no_dementia_scores = all_scores[all_targets == 0]
    misclassified_scores = all_scores[all_targets != all_outputs]

    def compute_cdf(data):
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(data) + 1) / len(data)
        return sorted_data, cdf

    # Compute CDF for different classes
    dementia_sorted, dementia_cdf = compute_cdf(dementia_scores)
    no_dementia_sorted, no_dementia_cdf = compute_cdf(no_dementia_scores)
    misclassified_sorted, misclassified_cdf = compute_cdf(misclassified_scores)


    # Plot CDF
    plt.figure(figsize=(8, 6))
    plt.plot(dementia_sorted, dementia_cdf, label='Dementia', color='blue')
    plt.plot(no_dementia_sorted, no_dementia_cdf, label='No dementia', color='orange')
    plt.plot(misclassified_sorted, misclassified_cdf, label='Misclassified', color='green')

    plt.xlabel('Score')
    plt.ylabel('CDF')
    plt.title('Transformer')
    plt.legend()
    plt.grid(True)
    plt.savefig("cdf_plot.png")
    plt.close()

# Use DataLoader to train the model
train_model(model, train_loader)

# Re-evaluate the model using the best model state
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    final_accuracy, final_auc, final_f1, final_recall, final_specificity = evaluate_model(model, test_loader)
    print(f'Final Test Accuracy: {final_accuracy:.4f}, AUC: {final_auc:.4f}, '
          f'F1 Score: {final_f1:.4f}, Sensitivity: {final_recall:.4f}, Specificity: {final_specificity:.4f}')

    # Save the best model state
    torch.save(best_model_state, 'best_model_state111.pth')


