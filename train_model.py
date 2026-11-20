import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
df = pd.read_csv("dataset/student_data.csv")

# Convert text columns automatically
df = pd.get_dummies(df)

# Features and target
X = df.drop("G3", axis=1)
y = df["G3"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)

print("MAE:", mae)

# Save model
joblib.dump(model, "student_prediction_model.pkl")

print("✅ Model Saved Successfully")