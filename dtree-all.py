# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'data/data.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Drop irrelevant columns
irrelevant_columns = ['id', 'customer_id', 'name', 'ssn', 'month']
data = data.drop(columns=irrelevant_columns)

# Apply label encoding to categorical variables
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Separate features (X) and target (y)
X = data.drop(columns=['credit_score'])
y = data['credit_score']

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Train an XGBoost Classifier
xgb_clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_clf.fit(X_train, y_train)
xgb_pred = xgb_clf.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)

# Combine models in a Voting Classifier
lr_clf = LogisticRegression(random_state=42, max_iter=1000)
svc_clf = SVC(probability=True, random_state=42)

ensemble_clf = VotingClassifier(estimators=[
    ('rf', rf_clf),
    ('xgb', xgb_clf),
    ('svc', svc_clf),
    ('lr', lr_clf)
], voting='soft')
ensemble_clf.fit(X_train, y_train)
ensemble_pred = ensemble_clf.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

# Output accuracies of all models
print({
    "Random Forest Accuracy": rf_accuracy,
    "XGBoost Accuracy": xgb_accuracy,
    "Ensemble Accuracy": ensemble_accuracy
})
