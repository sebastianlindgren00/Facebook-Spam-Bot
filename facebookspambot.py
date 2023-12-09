# Dataset from: https://www.kaggle.com/datasets/khajahussainsk/facebook-spam-dataset/
# Importing classifiers from libraries
from sklearn.naive_bayes import GaussianNB # Gaussian NB, Multinomial can't accept negative values.
from sklearn.tree import DecisionTreeClassifier # Decision Trees
from sklearn.dummy import DummyClassifier # Dummy Classifier
from sklearn.svm import SVC # SVM

# Importing libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA # PCA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
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

# Dummy Classifier
dummy_pipeline = Pipeline([
    ('features', ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), feature_columns)
    ])),
    ('classifier', DummyClassifier(strategy='most_frequent'))  # You can adjust the strategy as needed
])

dummy_pipeline.fit(X_train, y_train)
y_pred_dummy = dummy_pipeline.predict(X_test)
accuracy_dummy = accuracy_score(y_test, y_pred_dummy)
print("Dummy Classifier Accuracy:", accuracy_dummy)

# Gaussian Naive Bayes
gnb_pipeline = Pipeline([
    ('features', ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), feature_columns)
    ])),
    ('classifier', GaussianNB())
])

gnb_pipeline.fit(X_train, y_train)
y_pred_gnb = gnb_pipeline.predict(X_test)
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
print("Gaussian Naive Bayes Accuracy:", accuracy_gnb)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_pipeline = Pipeline([
    ('features', ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), feature_columns)
    ])),
    ('classifier', RandomForestClassifier(random_state=42))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)

# K-Nearest Neighbors (KNN) Classifier
from sklearn.neighbors import KNeighborsClassifier
knn_pipeline = Pipeline([
    ('features', ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), feature_columns)
    ])),
    ('classifier', KNeighborsClassifier())
])

knn_pipeline.fit(X_train, y_train)
y_pred_knn = knn_pipeline.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("K-Nearest Neighbors Accuracy:", accuracy_knn)

# SVM (Linear)
svm_pipeline = Pipeline([
    ('features', ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=6))  # Add PCA here with the appropriate number of components
        ]), feature_columns)
    ])),
    ('classifier', SVC(kernel='linear'))
])

svm_pipeline.fit(X_train, y_train)
y_pred_svm = svm_pipeline.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM (Linear) Accuracy:", accuracy_svm)

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
all_predictions = svm_pipeline.predict(X)

# Add the predicted labels to the original DataFrame
spam_data['Predicted_Label'] = all_predictions

# Check for no missing values
spam_data_no_missing = spam_data.dropna()

# Get the PCA-transformed data for the entire dataset
all_data_pca = svm_pipeline.named_steps['features'].transform(spam_data_no_missing)

# Scree plot to show explained variance for the entire dataset
# Scree plot is used to see elbow point
explained_variance_ratio = svm_pipeline.named_steps['features'].transformers_[0][1].named_steps['pca'].explained_variance_ratio_

cumulative_explained_variance = explained_variance_ratio.cumsum()

# Biplot to visualize feature contributions to principal components for the entire dataset
pca_components = svm_pipeline.named_steps['features'].transformers_[0][1].named_steps['pca'].components_

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

explained_variance_ratio = svm_pipeline.named_steps['features'].transformers_[0][1].named_steps['pca'].explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot - Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()
