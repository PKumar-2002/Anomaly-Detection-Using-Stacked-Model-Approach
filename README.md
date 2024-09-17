Here’s a cool Markdown version of your code with added flair:

# 🚀 Anomaly Detection with Stacked Model Approach(Ensemble models)

This project demonstrates anomaly detection using three models: **Isolation Forest**, **One-Class SVM**, and an **Autoencoder**. We then create an ensemble model to combine the anomaly scores and evaluate the performance using AUC scores.

### 📦 Libraries Used
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
```

### 📊 Dataset

The dataset used is the **KDD Cup 1999** dataset, focusing on HTTP service data.

```python
# Load the data
column_names = [...]
df = pd.read_csv('kddcup.data_10_percent_corrected', header=None, names=column_names)
df_test = pd.read_csv('corrected', header=None, names=column_names)
```

We filter the dataset to include only `HTTP` service data and drop the `service` column:

```python
df = df[df["service"] == "http"].drop("service", axis=1)
df_test = df_test[df_test["service"] == "http"].drop("service", axis=1)
```

### 🧠 Data Preprocessing

- **Label Encoding** is applied to categorical features.
- **Train-Test Split** with 80% training and 20% validation data.
- **Standardization** of features using `StandardScaler`.

```python
# Encode categorical variables
label_encoders = {}
# [Encoding logic here]

# Shuffle and split data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
```

### 🤖 Model 1: Isolation Forest

```python
iso_forest = IsolationForest(
    n_estimators=1000, 
    max_samples=0.8, 
    contamination=0.05, 
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(x_train)
anomaly_scores_iso_val = iso_forest.decision_function(x_val)
anomaly_scores_iso_test = iso_forest.decision_function(x_test)
```

### 🤖 Model 2: One-Class SVM

```python
ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
ocsvm.fit(x_train)
anomaly_scores_ocsvm_val = ocsvm.decision_function(x_val)
anomaly_scores_ocsvm_test = ocsvm.decision_function(x_test)
```

### 🤖 Model 3: Autoencoder

```python
autoencoder = Sequential([
    Dense(encoding_dim, activation="relu", input_shape=(input_dim,)),
    Dropout(0.2),
    Dense(input_dim, activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mse')
history = autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_val, x_val), verbose=1)
```

### 🔀 Ensemble Model: Weighted Average of Anomaly Scores

The final anomaly scores are a weighted average of the individual models' scores:

```python
val_scores_combined = 0.4 * anomaly_scores_iso_val + 0.3 * anomaly_scores_ocsvm_val + 0.3 * anomaly_scores_auto_val
test_scores_combined = 0.4 * anomaly_scores_iso_test + 0.3 * anomaly_scores_ocsvm_test + 0.3 * anomaly_scores_auto_test
```

### 📈 Anomaly Score Distribution

```python
def plot_histogram(anomaly_scores, title):
    plt.figure(figsize=(15, 10))
    plt.hist(anomaly_scores, bins=100)
    plt.xlabel('Anomaly Scores', fontsize=14)
    plt.ylabel('Number of Data Points', fontsize=14)
    plt.title(title)
    plt.show()

plot_histogram(val_scores_combined, 'Validation Set Combined Anomaly Scores')
plot_histogram(test_scores_combined, 'Test Set Combined Anomaly Scores')
```

### 📊 AUC Evaluation

Validation AUC: **93.64%**  
Test AUC: **97.63%**

```python
auc_val_combined = roc_auc_score(y_val == label_encoders["label"].transform(["normal."])[0], val_scores_combined)
print(f"Validation AUC: {auc_val_combined:.2%}")

auc_test_combined = roc_auc_score(y_test == label_encoders["label"].transform(["normal."])[0], test_scores_combined)
print(f"Test AUC: {auc_test_combined:.2%}")
```

### 🎉 Results

The ensemble model shows strong performance with the following AUC scores:

- **Validation AUC:** 93.64%
- **Test AUC:** 97.63%

```

This Markdown version highlights the key steps in your anomaly detection pipeline while keeping it sleek and readable. The use of emoji makes it engaging and professional!
