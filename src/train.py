from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import data_loader
import preprocess

# Load data
data_dir = "data/train"
pos_reviews, neg_reviews = data_loader.load_data(data_dir)

# Preprocess data
preprocessed_pos_reviews = [preprocess.preprocess_text(review) for review in pos_reviews]
preprocessed_neg_reviews = [preprocess.preprocess_text(review) for review in neg_reviews]

# Create labels
pos_labels = [1] * len(preprocessed_pos_reviews)
neg_labels = [0] * len(preprocessed_neg_reviews)

# Combine data and labels
X = preprocessed_pos_reviews + preprocessed_neg_reviews
y = pos_labels + neg_labels

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate the model
y_pred = model.predict(X_val_vec)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation accuracy: {accuracy}")

# Save the trained model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
