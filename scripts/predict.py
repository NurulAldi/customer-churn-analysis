import joblib
import pandas as pd
import os

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'model', 'churn_model.pkl')

# Load the trained model
model = joblib.load(MODEL_PATH)

def predict_churn(data, threshold=0.3):
    """
    Predict churn for new customer data
    
    Parameters:
    -----------
    data : pd.DataFrame
        Customer data with same features as training data (excluding 'churn' and 'customer_id')
    threshold : float, default=0.3
        Probability threshold for churn prediction
    
    Returns:
    --------
    predictions : numpy.ndarray
        Binary predictions (0 = No churn, 1 = Churn)
    probabilities : numpy.ndarray
        Churn probabilities for each customer
    """
    probabilities = model.predict_proba(data)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities