import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.utils.class_weight import compute_class_weight
from nltk.stem import WordNetLemmatizer
import joblib
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the dataset
file_path = 'Actual_dataset.csv'
data = pd.read_csv(file_path)

# Enhanced preprocessing function with lemmatization
def preprocess_text_lemmatized(text):
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    tokens = text.lower().split()
    filtered_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in ENGLISH_STOP_WORDS
    ]
    return ' '.join(filtered_tokens)

# Apply enhanced preprocessing to the dataset
data['cleaned_text'] = data['text'].apply(preprocess_text_lemmatized)

# Encode the labels
label_encoder = LabelEncoder()
data['encoded_label'] = label_encoder.fit_transform(data['label'])

# Save the label encoder
os.makedirs("saved_models", exist_ok=True)
joblib.dump(label_encoder, "saved_models/label_encoder.joblib")
print("Label encoder saved to saved_models/label_encoder.joblib.\n")

# .
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_tfidf = tfidf_vectorizer.fit_transform(data['cleaned_text'])

# Save the TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, "saved_models/tfidf_vectorizer.joblib")
print("TF-IDF vectorizer saved to saved_models/tfidf_vectorizer.joblib.\n")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, data['encoded_label'], test_size=0.2, random_state=42
)

# Compute class weights for imbalanced datasets
class_weights = compute_class_weight(
    'balanced', classes=data['encoded_label'].unique(), y=data['encoded_label']
)
class_weight_dict = dict(zip(data['encoded_label'].unique(), class_weights))

# Initialize and tune models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced'),
    "Support Vector Machine": SVC(class_weight='balanced', probability=True),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
    "Naive Bayes": MultinomialNB()
}

# Train, evaluate, and save models
results = {}
metrics_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    results[model_name] = {
        "Accuracy": accuracy,
        "Classification Report": classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    }
    # Add metrics to DataFrame
    metrics_row = pd.DataFrame([{
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }])
    metrics_df = pd.concat([metrics_df, metrics_row], ignore_index=True)

    # Save the trained model
    model_file_path = os.path.join("saved_models", f"{model_name.replace(' ', '_')}.joblib")
    joblib.dump(model, model_file_path)
    print(f"{model_name} saved to {model_file_path}.\n")

# Display results for models
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    print(f"Accuracy: {metrics['Accuracy']:.3f}\n")
    print("Classification Report:")
    print(metrics['Classification Report'])
    print("-" * 50)

# Plot metrics
def plot_metrics(metric_name, metrics_df):
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_df['Model'], metrics_df[metric_name], color='skyblue')
    plt.title(f'Model Comparison - {metric_name}', fontsize=14)
    plt.ylabel(metric_name, fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    plt.show()

# Plot each metric
plot_metrics("Accuracy", metrics_df)
plot_metrics("Precision", metrics_df)
plot_metrics("Recall", metrics_df)
plot_metrics("F1-Score", metrics_df)

# EDA: Distribution of classes
plt.figure(figsize=(8, 5))
data['label'].value_counts().plot(kind='bar', color='coral')
plt.title('Class Distribution', fontsize=14)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Class', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
plt.show()
