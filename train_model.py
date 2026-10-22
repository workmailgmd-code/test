import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# ■■ Training Data ■■
# Each row: [study_hours, prev_score]
X = np.array([
    [1, 40], # Fail
    [2, 45], # Fail
    [3, 50], # Fail
    [4, 60], # Pass
    [5, 65], # Pass
    [6, 70], # Pass
    [7, 75], # Pass
    [8, 85], # Pass
    [2, 80], # Pass (high score offset)
    [1, 30], # Fail
])

# Labels: 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0])

# ■■ Train the Model ■■
model = LogisticRegression()
model.fit(X, y)

# ■■ Save to Disk ■■
joblib.dump(model, "model.pkl")
print("Model trained and saved to model.pkl successfully!")