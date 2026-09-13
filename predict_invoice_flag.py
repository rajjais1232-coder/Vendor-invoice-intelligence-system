import joblib
import pandas as pd

MODEL_PATH = "models/predict_flag_invoice.pkl"

def load_model(model_path: str = MODEL_PATH):
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_invoice_flag(input_data):
    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['Predicted_Flag'] = model.predict(input_df).round()
    return input_df

if __name__ == "__main__":
    sample_data = {
        "invoice_quantity": [50],
        "invoice_dollars": [500.0],
        "Freight": [1.73],
        "total_item_quantity": [150],
        "total_item_dollars": [2450.0]
    }
    prediction = predict_invoice_flag(sample_data)
    print(prediction)