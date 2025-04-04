# 📌 Customer Churn Prediction

**Predict whether a customer will churn using machine learning!**  

![Churn Prediction](https://miro.medium.com/v2/resize:fit:1400/1*MoDTJbRsSwFLA4qylStQxA.png)  

---

## 🚀 Project Overview  
This project predicts customer churn based on various customer attributes like tenure, contract type, payment method, etc. It uses **machine learning models** to analyze customer behavior and identify potential churners.  

The web app is built using **Streamlit**, and the machine learning model is trained with **scikit-learn**.  

---

## 💂️ Project Structure  

```
customer-churn-prediction/
│️── app/  
│️   ├── app.py              # Streamlit web app  
│️   ├── model.pkl           # Trained ML model  
│️   └── requirements.txt    # Dependencies  
│️── data/  
│️   └── telco_customer_data.csv  # Raw dataset  
│️── notebooks/  
│️   └── model_training.ipynb  # Jupyter Notebook for model training  
│️── README.md  
│️── .gitignore  
```

---

## 🔧 Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2️⃣ Install Dependencies  
Make sure you have Python installed, then run:  
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App  
```bash
streamlit run app/app.py
```

---

## 📊 Dataset  
We use a **Telco Customer Churn Dataset**, which contains:  
- Customer demographics  
- Subscription details  
- Payment methods  
- Tenure information  
- Churn labels  

📌 **Target Variable**: `"Churn"` (Yes/No)  

---

## 🤖 Machine Learning Model  
The model pipeline:  
👉 **Data Preprocessing**: Handling missing values, encoding categorical features  
👉 **Feature Scaling**: Standardization using `StandardScaler`  
👉 **Model Used**: `XGBoost`
👉 **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score  

📌 **Trained model is saved as `model.pkl`**  

---

## 🎯 Usage  
1️⃣ Open the Streamlit app  
2️⃣ Enter customer details in the UI  
3️⃣ Click "Predict" to see whether the customer is likely to churn or not  

---

## 💡 To-Do / Future Improvements  
- [ ] Improve accuracy with hyperparameter tuning  
- [ ] Deploy on Streamlit Sharing / Hugging Face Spaces 
- [ ] Integrate more advanced models (e.g.ANN)  
- [ ] Add an API using FastAPI

---

## 🏆 Contributing  
Feel free to submit issues and pull requests to improve this project!  

---

## 🐝 License  
This project is open-source under the **MIT License**.  

---

## 📬 Contact  
👤 **Nagarajan Venugopal**  
📧 nagarajan.v1595@gmail.com  
🔗 [LinkedIn](www.linkedin.com/in/nagarajan-venugopal-06b6a2293) | [GitHub](https://github.com/nagavenu1595)  

