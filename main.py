import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'data/data.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Drop irrelevant columns
irrelevant_columns = ['id', 'customer_id', 'name', 'ssn', 'month']
data = data.drop(columns=irrelevant_columns)

# Handle 'type_of_loan' column that contains list-like strings
if 'type_of_loan' in data.columns:
    data['type_of_loan'] = data['type_of_loan'].astype(str).str.strip("[]").str.replace("'", "").str.split(", ")
    mlb = pd.get_dummies(data['type_of_loan'].explode()).groupby(level=0).max()
    data = pd.concat([data.drop(columns=['type_of_loan']), mlb], axis=1)

# Identify remaining categorical columns
categorical_columns = [col for col in ['occupation', 'credit_mix', 'payment_of_min_amount', 'payment_behaviour']
                       if col in data.columns]

# One-hot encode all remaining categorical variables
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Define features (X) and target (y)
X = data.drop(columns=['credit_score'])
y = data['credit_score']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=88888)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=2478, n_jobs=-1)
clf.fit(X_train, y_train)

# Predict on the testing and training sets
y_pred = clf.predict(X_test)
y_pred_train = clf.predict(X_train)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracy_train = accuracy_score(y_train, y_pred_train)

# Retrieve and display feature importance scores
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Output results
print("Test Accuracy:", accuracy)
print("Training Accuracy:", accuracy_train)
print("Feature Importances:\n", feature_importances)
