# Import necessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Dataset 1: IMDB Dataset
data_imdb = pd.read_csv(r"C:\Users\venne\Downloads\sentiment_analysis-master\sentiment_analysis-master\Datasets\imdb_labelled.txt", sep="\t", header=None)
data_imdb.columns = ['text', 'sentiment']

# Dataset 2: Yelp Dataset
data_yelp = pd.read_csv(r"C:\Users\venne\Downloads\sentiment_analysis-master\sentiment_analysis-master\Datasets\yelp_labelled.txt", sep="\t", header=None)
data_yelp.columns = ['text', 'sentiment']

# Dataset 3: Amazon Cells Dataset
data_amazon = pd.read_csv(r"C:\Users\venne\Downloads\sentiment_analysis-master\sentiment_analysis-master\Datasets\amazon_cells_labelled.txt", sep="\t", header=None)
data_amazon.columns = ['text', 'sentiment']

# Combine all datasets into one
data_all = pd.concat([data_imdb, data_yelp, data_amazon], axis=0).reset_index(drop=True)

# Step 1: Preprocess the Data
X = data_all['text']  # Text data
y = data_all['sentiment']  # Sentiment labels

# Convert labels to numeric values (0 for negative, 1 for positive)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Step 2: Split the Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Feature Extraction using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 4: Train Models and Evaluate Performance

# Initialize classifiers
models = {
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine (SVM)": SVC(probability=True),
    "Artificial Neural Network (ANN)": MLPClassifier()
}

metrics_data = []

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['precision']
    recall = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['recall']
    f1_score = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
    
    # ROC-AUC
    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test_tfidf)[:, 1])
    except:
        auc = np.nan  # SVM might not provide probabilities if `probability=True` is not set
    
    # Append results
    metrics_data.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score,
        "ROC-AUC": auc
    })

# Convert to a DataFrame for better visualization
metrics_df = pd.DataFrame(metrics_data)

# Display the metrics table
print("\nModel Performance Metrics:")
print(metrics_df)

# Step 5: Visualize Metrics - All in One Graph

# Set the index for easier plotting
metrics_df.set_index("Model", inplace=True)

# Create a subplot for all metrics (2x3 grid but with an extra blank space)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Metrics to plot
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
colors = ['skyblue', 'orange', 'lightgreen', 'salmon', 'purple']

# Loop through the metrics and plot each on the respective subplot
for i, metric in enumerate(metrics):
    ax = axes[i // 3, i % 3]  # Assign each metric to a subplot
    metrics_df[metric].plot(kind="bar", color=colors[i], ax=ax, title=f"{metric} Comparison Across Models")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)  # Assuming all metrics are between 0 and 1
    ax.set_xticklabels(metrics_df.index, rotation=45)

# Remove the empty subplot (last one in the grid)
fig.delaxes(axes[1, 2])

# Adjust layout for a clean display
plt.tight_layout()
plt.show()

# Step 6: Select the Best Model Based on F1-Score (or any chosen metric)
best_model_name = metrics_df.sort_values(by="F1-Score", ascending=False).index[0]
best_model = models[best_model_name]

print(f"\nThe selected model for prediction is: {best_model_name}")

# Step 7: Save the selected model for later use
import joblib
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
