#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load the dataset
file_path = 'Cleaned_RomanianTextData.csv'  # Update with your CSV file path
data = pd.read_csv(file_path)

# Combine the text columns into one
data['text'] = data[['Column1', 'Column2', 'Column3', 'Column4']].fillna('').agg(' '.join, axis=1)

# Add labels (if necessary)
# Example: Add dummy labels (1 and 0) for binary classification
data['label'] = [1 if i % 2 == 0 else 0 for i in range(len(data))]  # Replace with actual logic if labels exist

# Split the dataset
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train an SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_vect, y_train)

# Predict and evaluate
y_pred = svm_model.predict(X_test_vect)
print(classification_report(y_test, y_pred))


# In[ ]:




