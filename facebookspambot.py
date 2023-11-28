# Dataset from: https://www.kaggle.com/datasets/khajahussainsk/facebook-spam-dataset/
# Importing classifiers from libraries
from sklearn.naive_bayes import MultinomialNB # Naive Bayes
from sklearn.tree import DecisionTreeClassifier # Decision Trees
from sklearn.dummy import DummyClassifier # Dummy Classifier

# Importing libraries
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
from sklearn.impute import SimpleImputer# Imputer, since SVC() can't handle non-existing values.
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------
# Read dataset
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

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline with ColumnTransformer for different feature types
pipeline = Pipeline([
    ('features', ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=10))
        ]), feature_columns)
    ])),
    ('classifier', SVC(kernel='linear'))
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

# Model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# ---------------------------------------------------------------
# Feature importancy

# Use a decision tree-based classifier to extract feature importances
# Tried with SVM before but couldn't handle it since it worked with different array lengths.
tree_classifier = DecisionTreeClassifier(random_state=42)
tree_classifier.fit(X_train, y_train)

# Feature importances and feature names
feature_importances = tree_classifier.feature_importances_
feature_names = X.columns

# DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(feature_importance_df)

# Select the top features
threshold = 0.01  # Adjust the threshold as needed
top_features = feature_importance_df[feature_importance_df['Importance'] > threshold]['Feature'].tolist()

# New pipeline with only the top features
pipeline_top_features = Pipeline([
    ('features', ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), top_features)
    ])),
    ('classifier', SVC(kernel='linear'))
])

pipeline_top_features.fit(X_train, y_train)

y_pred_top_features = pipeline_top_features.predict(X_test)

# Model accuracy with only top features
accuracy_top_features = accuracy_score(y_test, y_pred_top_features)
print("Model Accuracy with Top Features:", accuracy_top_features)
# ---------------------------------------------------------------
# PCA and heatmap

# Predictions for all examples in the dataset
all_predictions = pipeline.predict(X)

# Add the predicted labels to the original DataFrame
spam_data['Predicted_Label'] = all_predictions

# Check for no missing values
spam_data_no_missing = spam_data.dropna()

# Get the PCA-transformed data for the entire dataset
all_data_pca = pipeline.named_steps['features'].transform(spam_data_no_missing)

# Scree plot to show explained variance for the entire dataset
# Scree plot is used to see elbow point
explained_variance_ratio = pipeline.named_steps['features'].transformers_[0][1].named_steps['pca'].explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

# Biplot to visualize feature contributions to principal components for the entire dataset
pca_components = pipeline.named_steps['features'].transformers_[0][1].named_steps['pca'].components_

pca_loadings_df = pd.DataFrame(pca_components.T, columns=[f'PC{i+1}' for i in range(pca_components.shape[0])], index=feature_columns)

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pca_loadings_df, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
plt.title('PCA Loadings - Heatmap of Feature Contributions to Principal Components')
plt.show()

# Plot to see where the elbow is. Used to decide the number of principal components.
# Link for explaining: https://sanchitamangale12.medium.com/scree-plot-733ed72c8608
# Through the graph, we can tell that 80% of the variance occurs in the first 4-5 principal components.
# This means we could regulate the amount of principal components to 4 or 5 instead of the original amount of 10.

explained_variance_ratio = pipeline.named_steps['features'].transformers_[0][1].named_steps['pca'].explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot - Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()
