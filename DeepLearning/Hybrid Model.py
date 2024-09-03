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
from sklearn.metrics import cohen_kappa_score
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns


df = pd.read_csv('dff4.csv')


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


print("Checking NaN values in the training set:")
print(np.isnan(X_train).sum())

print("Checking the NaN value in the test set:")
print(np.isnan(X_test).sum())


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


class AdaptiveWeighting(nn.Module):
    def __init__(self, d_model):
        super(AdaptiveWeighting, self).__init__()
        self.weight_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, inputs):
        weights = self.weight_mlp(inputs)
        return weights
def compute_entropy(inputs, bins=100):

    inputs_np = inputs.cpu().detach().numpy()

    entropies = []
    for i in range(inputs.shape[-1]):
        hist, _ = np.histogram(inputs_np[:, i], bins=bins, density=True)
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        entropies.append(entropy)
    
    # Normalize entropies to be between 0 and 1
    max_entropy = max(entropies)
    normalized_entropies = [e / max_entropy for e in entropies]
    
    # Convert back to tensor with the same type as inputs
    normalized_entropies_tensor = torch.tensor(normalized_entropies, device=inputs.device, dtype=inputs.dtype)
    return normalized_entropies_tensor.unsqueeze(0).unsqueeze(0)

class MaskedTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers):
        super(MaskedTransformerEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 2)
        self.adaptive_weighting = AdaptiveWeighting(d_model)

        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 256)
        )

    def forward(self, inputs):
        projected_inputs = self.input_projection(inputs)
        
        # Generate adaptive weights for each input vector
        weights = self.adaptive_weighting(projected_inputs)
        
        # Compute entropies for each input vector
        entropies = compute_entropy(projected_inputs)
        
        # Combine adaptive weights with entropies
        combined_weights = weights * entropies
        
        # Apply the combined weights to the projected inputs
        weighted_inputs = projected_inputs * combined_weights
        
        encoded_inputs = self.positional_encoding(weighted_inputs)
        output = self.transformer_encoder(encoded_inputs)
        output = output.mean(dim=0)
        classifier_output = self.classifier(output)
        projection_output = self.projection_head(output)

        return classifier_output, projection_output




input_dim = X_train.shape[1]
d_model = 768
nhead = 6
num_layers = 7



model = MaskedTransformerEncoder(input_dim, d_model, nhead, num_layers).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001,weight_decay=1e-4)
criterion = nn.CrossEntropyLoss().to(device)
# criterion = SmoothedCrossEntropyLoss(smoothing=0.3).to(device)
contrastive_criterion = ContrastiveLoss().to(device)
best_accuracy = 0.0
best_model_state = None

def train_model(model, train_loader, epochs=10):
    global best_accuracy, best_model_state

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()  
        running_loss = 0.0
        running_contrastive_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            classifier_output, projection_output = model(data)

            classification_loss = criterion(classifier_output, target)
            contrastive_loss = contrastive_criterion(projection_output, target)
            # print(contrastive_loss)
            loss = classification_loss + 0.01 * contrastive_loss

            loss.backward()
            optimizer.step()

            running_loss += classification_loss.item()
            running_contrastive_loss += contrastive_loss.item()

        
        avg_loss = running_loss / (batch_idx + 1)
        avg_contrastive_loss = running_contrastive_loss / (batch_idx + 1)

        
        accuracy, auc, f1, recall, specificity, kappa = evaluate_model(model, test_loader)

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Classification Loss: {avg_loss:.4f}, '
                  f'Test Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, '
                  f'F1 Score: {f1:.4f}, Sensitivity: {recall:.4f}, Specificity: {specificity:.4f},Kappa: {kappa:.4f},'
                  f'contrastive_loss: {contrastive_loss:4f}')




def evaluate_model(model, data_loader):
    model.eval()  # Set model to evaluation mode
    all_outputs = []
    all_targets = []
    all_scores = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs, _ = model(data)
            scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Get positive class scores
            _, predicted = torch.max(outputs.data, 1)

            all_outputs.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_scores.extend(scores)

    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    all_scores = np.array(all_scores)

    accuracy = accuracy_score(all_targets, all_outputs)
    try:
        auc = roc_auc_score(all_targets, all_outputs)  # Using predicted labels, not probabilities
    except ValueError:
        auc = 0.0  # If only one class present, roc_auc_score throws an exception

    # Compute Kappa coefficient
    kappa = cohen_kappa_score(all_targets, all_outputs)

    # Compute best F1 score by trying different thresholds
    best_f1 = 0
    best_threshold = 0
    for threshold in [0.1 * i for i in range(1, 10)]:
        predicted_labels = (np.array(all_outputs) >= threshold).astype(int)
        f1 = f1_score(all_targets, predicted_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    recall = recall_score(all_targets, (np.array(all_outputs) >= best_threshold).astype(int))
    tn, fp, fn, tp = confusion_matrix(all_targets, all_outputs).ravel()
    specificity = tn / (tn + fp)

    print(f"Positive samples: {sum(all_targets)}")
    print(f"Negative samples: {len(all_targets) - sum(all_targets)}")

    # Plot confusion matrix with proportions using 'plasma' colormap
    cm = confusion_matrix(all_targets, all_outputs)
    cm_proportions = cm / cm.sum()

    plt.figure(figsize=(8, 6))
    # Directly plot the confusion matrix to have x=Actual, y=Predicted
    sns.heatmap(cm_proportions.T, annot=True, fmt=".2f", cmap="plasma", cbar=True, annot_kws={"size": 16}, xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title("Confusion Matrix - Hybrid Model", fontsize=18)
    plt.xlabel("Actual", fontsize=14)  # Changed to "Actual"
    plt.ylabel("Predicted", fontsize=14)  # Changed to "Predicted"
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Save the confusion matrix plot
    plt.savefig("confusion_Hy.png")
    plt.close()

    # Plot Cumulative Distribution Function (CDF)
    plot_cdf(all_scores, all_targets, all_outputs)

    return accuracy, auc, best_f1, recall, specificity, kappa

def plot_cdf(all_scores, all_targets, all_outputs):
    
    dementia_scores = all_scores[all_targets == 1]
    no_dementia_scores = all_scores[all_targets == 0]
    misclassified_scores = all_scores[all_targets != all_outputs]

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
    plt.title('Hybrid Model')
    plt.legend()
    plt.grid(True)
    plt.savefig("dy.png")
    plt.close()


train_model(model, train_loader)

print("\nFinal evaluation on test set:")
accuracy, auc, f1, recall, specificity, kappa = evaluate_model(model, test_loader)
print(f'Test Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, '
      f'F1 Score: {f1:.4f}, Sensitivity: {recall:.4f}, Specificity: {specificity:.4f}')





