# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import tree

# Load the dataset
file_path = 'data/data.csv'  # Replace with the actual file path
data = pd.read_csv(file_path)

# Drop irrelevant columns
irrelevant_columns = ['id', 'customer_id', 'name', 'ssn', 'month']
data = data.drop(columns=irrelevant_columns)

# Inspect data types to identify categorical features
print("Data Types:\n", data.dtypes)

# Apply Label Encoding to categorical features
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    print(f"Encoding column: {col}")
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))  # Convert to string before encoding
    label_encoders[col] = le

# Define features (X) and target (y)
X = data.drop(columns=['credit_score'])
y = data['credit_score']

# Ensure all columns in X are numeric
print("Feature Data Types After Encoding:\n", X.dtypes)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on the testing set
y_pred = clf.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the Decision Tree:", accuracy)


# Analyze feature importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:\n", feature_importances)
