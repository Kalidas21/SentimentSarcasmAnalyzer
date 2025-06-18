!pip install transformers
!pip install datasets
!pip install torch
!pip install scikit-learn
import pandas as pd

# Load the dataset
df = pd.read_csv("sarcasm_dataset.csv")

# Check the first few rows
df.head()

from sklearn.model_selection import train_test_split

# Split the dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Check the shape of the train and test datasets
print(f"Train dataset size: {train_df.shape}")
print(f"Test dataset size: {test_df.shape}")

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch

# Load the pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class SarcasmDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        sentence = str(self.dataframe.iloc[index]['Sentence'])  # Use correct column name
        label = self.dataframe.iloc[index]['Label']             # Use correct column name

        inputs = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Parameters
MAX_LEN = 128
BATCH_SIZE = 16

# Create the dataset
train_dataset = SarcasmDataset(train_df, tokenizer, MAX_LEN)
test_dataset = SarcasmDataset(test_df, tokenizer, MAX_LEN)

# DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

from torch.optim import AdamW
from sklearn.metrics import accuracy_score

# Optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
def train_model(model, train_loader, optimizer, loss_fn, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Compute loss
            loss = loss_fn(logits, labels)
            epoch_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, preds = torch.max(logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

        # Print the loss and accuracy for each epoch
        epoch_accuracy = correct_predictions / total_predictions * 100
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {epoch_accuracy:.2f}%")

train_model(model, train_loader, optimizer, loss_fn, device, epochs=3)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

evaluate_model(model, test_loader, device)

model.save_pretrained("sarcasm_model")
tokenizer.save_pretrained("sarcasm_model")

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained("sarcasm_model")
tokenizer = BertTokenizer.from_pretrained("sarcasm_model")
model.to(device)
