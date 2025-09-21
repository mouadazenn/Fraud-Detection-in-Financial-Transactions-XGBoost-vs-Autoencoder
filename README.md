# 💳 Fraud Detection in Financial Transactions: XGBoost vs Autoencoder

This project applies **machine learning** techniques to detect fraudulent transactions using the **PaySim dataset**, a large-scale financial simulation.  
We compare a **supervised approach (XGBoost)** with an **unsupervised approach (Autoencoder)** to highlight their strengths and limitations in fraud detection.

---

## 📂 Project Structure
- `notebook.ipynb` → Full code and analysis (data exploration, feature engineering, modeling, evaluation).  
- `data/` → PaySim dataset (not included here, can be downloaded separately).  
- `README.md` → Project overview and results summary.  

---

## ⚙️ Methods

### 🔹 Supervised Learning: XGBoost
- Feature engineering: transaction type encoding, balance consistency checks, ratios, time features.  
- Time-based train/validation/test split to mimic real-world deployment.  
- Handled class imbalance with `scale_pos_weight`.  
- Metrics: Confusion Matrix, Classification Report, ROC-AUC, PR-AUC.  

### 🔹 Unsupervised Learning: Autoencoder
- Trained only on **normal transactions** to learn typical patterns.  
- Fraud detected via **reconstruction error** thresholding.  
- Compared anomaly scores with labels on test data.  

---

## 📊 Results

| Model        | ROC-AUC | PR-AUC | Precision (Fraud) | Recall (Fraud) |
|--------------|---------|--------|-------------------|----------------|
| XGBoost      | ~1.000  | ~1.000 | 0.997             | 0.999          |
| Autoencoder  | 0.846   | 0.221  | 0.016             | 0.603          |

- **XGBoost** achieved near-perfect results on this dataset (likely due to strong signals in balances).  
- **Autoencoder** detected some frauds but had many false positives, reflecting the real-world challenge of anomaly detection.  

---

## 🔑 Key Takeaways
- Supervised models (XGBoost) are powerful when **high-quality labels** exist.  
- Autoencoders are valuable for detecting **new or unseen fraud patterns**, especially when labels are incomplete.  
- The best production systems often **combine supervised + unsupervised methods**.  

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/fraud-detection-xgb-ae.git
   cd fraud-detection-xgb-ae
pip install -r requirements.txt
jupyter notebook notebook.ipynb
