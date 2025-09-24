import pickle
import pandas as pd
import numpy as np

# ✅ Load the model (use correct path)
model = pickle.load(open('C:/Users/adity/OneDrive/Desktop/fraud_detection_app/trained_model.sav', 'rb'))

# ✅ Correct feature names
columns = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
    'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

# ✅ Dummy input (replace with real transaction data if available)
example_input = pd.DataFrame([np.random.rand(30)], columns=columns)

# ✅ Predict
prediction = model.predict(example_input)

# ✅ Print result
print("Prediction:", prediction)
