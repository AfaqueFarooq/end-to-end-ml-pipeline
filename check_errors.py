import dill
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# Load model and preprocessor
model = dill.load(open('artifacts/model.pkl', 'rb'))
preprocessor = dill.load(open('artifacts/preprocessor.pkl', 'rb'))

# Test set
test = pd.read_csv('artifacts/test.csv')
X_test = test.drop('math_score', axis=1)
y_test = test['math_score']
X_test_transformed = preprocessor.transform(X_test)
test_preds = model.predict(X_test_transformed)

# Train set
train = pd.read_csv('artifacts/train.csv')
X_train = train.drop('math_score', axis=1)
y_train = train['math_score']
X_train_transformed = preprocessor.transform(X_train)
train_preds = model.predict(X_train_transformed)

# Results
print("Train R²:", r2_score(y_train, train_preds))
print("Test R²:", r2_score(y_test, test_preds))

# Worst predictions
errors = abs(test_preds - y_test.values)
worst_idx = np.argsort(errors)[-3:]
print("\n3 Worst Predictions:")
print(test.iloc[worst_idx])
print("Predicted:", test_preds[worst_idx])
print("Actual:", y_test.values[worst_idx])