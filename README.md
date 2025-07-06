Credit--Card-Approval-Prediction


This project predicts whether a user’s credit will be *Approved* or *Not Approved* based on several features like age, debt, income, credit score, etc.

🚀 Tech Stack:
- Machine Learning*: Random Forest Classifier (sklearn)
- Frontend*: Streamlit
- Backend API*: FastAPI
- Model Serialization*: joblib
- Deployment Ready*

📁 Dataset
Cleaned and used the dataset with fields like:
- Gender, Age, Debt, Income, Credit Score
- Employment, Marital Status, Citizenship
- Years Employed, Industry, etc.

🧠 How it Works:
1. User inputs data in Streamlit form.
2. Streamlit sends data to FastAPI.
3. API loads the trained model & scaler.
4. Predicts if the credit should be approved.
5. Displays result in real-time.



📂 Files
- streamlit.py: Frontend interface
- main.py: FastAPI backend
- rf_model.pkl, scaler.pkl: Trained model & scaler
- clean_dataset.csv: Cleaned dataset
- demo.py: Sample JSON data

🛠 How to Run Locally
1. Clone the repo  
2. Install packages
