#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install torch')
get_ipython().system('pip install datasets')
get_ipython().system('pip install scikit-learn')


# In[6]:


import pandas as pd

# Load the dataset (update the path to your file)
file_path = 'Cleaned_RomanianTextData.csv'
data = pd.read_csv(file_path)

# Check the first few rows to understand the structure
data.head()


# In[11]:


import pandas as pd
from transformers import BertTokenizer

# Load your dataset
data = pd.read_csv('Cleaned_RomanianTextData.csv')

# Check the column names
print(data.columns)

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to encode the text data
def encode_text(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Encode the 'Column1' (assuming it contains the text)
encoded_data = encode_text(data['Column1'].tolist())

# Check the tokenized inputs
print(encoded_data)


# In[14]:


from sklearn.preprocessing import LabelEncoder
import torch

# Assuming 'Column2' contains the text labels for classification
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['Column2'].tolist())  # Convert labels to numerical format

# Convert labels to tensor format for PyTorch
labels = torch.tensor(labels)


# In[15]:


from sklearn.model_selection import train_test_split

# Split data into train and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(encoded_data['input_ids'], labels, test_size=0.2)

# Define Dataset class for training and validation
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

# Create train and validation datasets
train_dataset = TextDataset(train_inputs, train_labels)
val_dataset = TextDataset(val_inputs, val_labels)


# In[16]:


from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Initialize BERT model for sequence classification (adjust num_labels as needed)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',              
    num_train_epochs=3,                  
    per_device_train_batch_size=8,       
    per_device_eval_batch_size=16,       
    warmup_steps=500,                    
    weight_decay=0.01,                   
    logging_dir='./logs',                
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,            
)

# Train the model
trainer.train()


# In[17]:


# Evaluate the model
results = trainer.evaluate()
print("Evaluation results:", results)


# In[18]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Define a custom compute_metrics function
def compute_metrics(pred):
    """
    Custom function to compute accuracy, precision, recall, and F1 score.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)  # Convert logits to class predictions
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')  # Adjust average if needed
    recall = recall_score(labels, preds, average='weighted')        # Adjust average if needed
    f1 = f1_score(labels, preds, average='weighted')                # Adjust average if needed
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Redefine the Trainer with the custom metrics
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,            
    compute_metrics=compute_metrics      # Add the custom metrics function
)

# Evaluate the model
eval_results = trainer.evaluate()

# Print evaluation metrics
print("Evaluation Metrics:")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")


# In[ ]:


# Print evaluation results
print("Evaluation results:", results)

# Get the model predictions on the validation dataset
model.eval()  # Set the model to evaluation mode
predictions = []
true_labels = []

# Evaluate on the validation dataset
for batch in val_dataset:
    with torch.no_grad():
        inputs = batch['input_ids'].unsqueeze(0).to(model.device)
        labels = batch['labels'].unsqueeze(0).to(model.device)

        # Forward pass
        outputs = model(inputs)
        logits = outputs.logits

        # Get predictions
        preds = torch.argmax(logits, dim=-1)

        # Append the predictions and true labels
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Print classification report
print(classification_report(true_labels, predictions))


# In[ ]:




