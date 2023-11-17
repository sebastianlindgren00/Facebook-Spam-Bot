# Dataset from: https://www.kaggle.com/datasets/khajahussainsk/facebook-spam-dataset/
from sklearn.naive_bayes import MultinomialNB # Naive Bayes
from sklearn.tree import DecisionTreeClassifier # Decision Trees
from sklearn.dummy import DummyClassifier # Dummy Classifier

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
        ('numeric', StandardScaler(), feature_columns)
    ])),
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
    ('classifier', SVC())
])

# Train the model using the pipeline
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

# For all features

# Predictions for all examples in the dataset
all_predictions = pipeline.predict(X)

# Add the predicted labels to the original DataFrame
spam_data['Predicted_Label'] = all_predictions

random_subset = spam_data.sample(n=10, random_state=42) # To write out only a handful (n) of the 600 examples.

# Print details for 50 random examples in the dataset
print("Details for Each Random Example:")
for index, row in random_subset.iterrows():
    true_label = row['Label']
    predicted_label = row['Predicted_Label']
    
    print(f"Example {index + 1}:")
    print(f"  True Label: {true_label}")
    print(f"  Predicted Label: {predicted_label}")
    
    for feature_category in feature_columns:
        feature_value = row[feature_category]
        print(f"  {feature_category}: {feature_value}")

    print("---")

# Evaluate the overall model accuracy
overall_accuracy = accuracy_score(spam_data['Label'], all_predictions)
print("Overall Model Accuracy:", overall_accuracy)

