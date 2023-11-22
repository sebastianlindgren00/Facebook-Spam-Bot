# Dataset from: https://www.kaggle.com/datasets/khajahussainsk/facebook-spam-dataset/
from sklearn.naive_bayes import MultinomialNB # Naive Bayes
from sklearn.tree import DecisionTreeClassifier # Decision Trees
from sklearn.dummy import DummyClassifier # Dummy Classifier

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA # PCA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer# Imputer, since SVC() can't handle non existing values.
from sklearn.metrics import accuracy_score

# Read the CSV file
spam_data = pd.read_csv('/Users/sebastianlindgren/Documents/Fork_Projects/FacebookSpamBot/Facebook Spam Dataset.csv')

# Check for missing values
if spam_data.isnull().any().any():
    print("Warning: The dataset contains missing values.")

# Extract features and labels
feature_columns = ["#friends", "#following", "#community", "age", "#postshared", "#urlshared",
                    "#photos/videos", "fpurls", "fpphotos/videos", "avgcomment/post", "likes/post",
                    "tags/post", "#tags/post"]
target_column = "Label"

X = spam_data[feature_columns]
y = spam_data[target_column]

# Replace numerical labels with "spam" and "not spam"
y.replace({0: 'not spam', 1: 'spam'}, inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with ColumnTransformer for different feature types
pipeline = Pipeline([
    ('features', ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=10))
        ]), feature_columns)
    ])),
    ('classifier', SVC())
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Predictions on the test set
y_pred = pipeline.predict(X_test)

# Print examples for each feature category
for feature_category in feature_columns:
    print(f"Examples for {feature_category}:")
    category_indices = X_test.index[X_test[feature_category].notnull()]
    
    for i in range(min(1, len(category_indices))): # Adjust number to change amount of examples.
        index = category_indices[i]
        true_label = y_test.loc[index]
        predicted_label = y_pred[index]
        feature_value = X_test.loc[index, feature_category]
        
        print(f"Example {i + 1}:")
        print(f"  {feature_category}: {feature_value}")
        print(f"  True Label: {true_label}")
        print(f"  Predicted Label: {predicted_label}")
        print("---")

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# PCA and heatmap

# Predictions for all examples in the dataset
all_predictions = pipeline.predict(X)

# Add the predicted labels to the original DataFrame
spam_data['Predicted_Label'] = all_predictions

# Ensure there are no missing values in the dataset
spam_data_no_missing = spam_data.dropna()

# Get the PCA-transformed data for the entire dataset after handling missing values
all_data_pca = pipeline.named_steps['features'].transform(spam_data_no_missing)

# Scree plot to show explained variance for the entire dataset
explained_variance_ratio = pipeline.named_steps['features'].transformers_[0][1].named_steps['pca'].explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

# Biplot to visualize feature contributions to principal components for the entire dataset
pca_components = pipeline.named_steps['features'].transformers_[0][1].named_steps['pca'].components_

pca_loadings_df = pd.DataFrame(pca_components.T, columns=[f'PC{i+1}' for i in range(pca_components.shape[0])], index=feature_columns)

# Plot a heatmap of the PCA loadings
plt.figure(figsize=(12, 8))
sns.heatmap(pca_loadings_df, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
plt.title('PCA Loadings - Heatmap of Feature Contributions to Principal Components')
plt.show()

# Plot to see where the elbow is. Used to decide number of principal components.
# Link for explaining: https://sanchitamangale12.medium.com/scree-plot-733ed72c8608
# Thorugh the graph we can tell that 80% of the variance occurs in the first 4-5 principal components.
# This mean we could regulate the amount of principal components to 4 or 5 instead of the original amount of 10.

explained_variance_ratio = pipeline.named_steps['features'].transformers_[0][1].named_steps['pca'].explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot - Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()