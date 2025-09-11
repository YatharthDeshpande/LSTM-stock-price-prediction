# 📈 Stock Price Prediction using LSTM & Keras Tuner

<div align="center">

**A sophisticated machine learning application that predicts stock prices using Long Short-Term Memory (LSTM) networks with automated hyperparameter optimization.**

</div>

---

## 🌟 Overview

This project leverages deep learning to predict stock prices with high accuracy using LSTM neural networks. The model analyzes 120 days of historical stock data to predict next-day prices, featuring automatic hyperparameter tuning for optimal performance.

### ✨ Key Highlights
- **🎯 High Accuracy**: Advanced LSTM architecture for precise predictions
- **🤖 Auto-Optimization**: Keras Tuner for intelligent hyperparameter selection  
- **📊 Real-time Data**: Live stock data from Yahoo Finance API
- **🖥️ Interactive UI**: Beautiful Streamlit web interface
- **📈 Visual Analytics**: Comprehensive charts and performance metrics

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 📊 **Data Fetching** | Automatic retrieval of historical stock data from Yahoo Finance |
| 🔧 **Hyperparameter Tuning** | Intelligent optimization using Keras Tuner |
| ⏳ **Time Series Analysis** | Analyzes 120-day patterns for next-day predictions |
| 🖥️ **Web Interface** | Interactive Streamlit dashboard with real-time predictions |
| 📈 **Performance Metrics** | Comprehensive accuracy and error analysis |
| 💾 **Model Persistence** | Automated model saving and loading |

---

## 📂 Project Structure

```plaintext
stock-price-prediction-lstm/
├── 📄 train_model.py        # LSTM model training with hyperparameter tuning
├── 🌐 app.py                # Streamlit web application
├── 📋 requirements.txt      # Python dependencies
├── 📚 README.md             # Project documentation
└── 📁 models/               # Trained model storage (auto-created)
    └── best_stock_model.keras
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git

### Quick Start

```bash
# 1️⃣ Clone the repository
cd LSTM-stock-price-prediction

# 2️⃣ Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt
```

---

## 🎯 Usage

### Training the Model

```bash
python train_model.py
```

**What happens during training:**
- 📥 Downloads historical stock data
- 🔍 Performs hyperparameter optimization
- 🧠 Trains the optimal LSTM model
- 💾 Saves the best model as `best_stock_model.keras`

### Running the Web Application

```bash
streamlit run app.py
```

**Features of the web app:**
- 🔍 Enter any stock ticker symbol (e.g., AAPL, TSLA, GOOGL)
- 📊 View predicted vs actual price comparisons
- 📈 Interactive charts and visualizations
- 📋 Detailed performance metrics and accuracy scores

---

## 📸 Screenshots

<div align="center">

### 🏪 Walmart (WMT) Prediction
<img src="https://github.com/user-attachments/assets/0b42fac2-2e21-456b-961a-2e3274961b6c" alt="Walmart Stock Prediction" width="800"/>

### 🍎 Apple (AAPL) Prediction  
<img src="https://github.com/user-attachments/assets/0803d304-c860-4d23-8d71-4fe2760908a2" alt="Apple Stock Prediction" width="800"/>

### 🚗 Tesla (TSLA) Prediction
<img src="https://github.com/user-attachments/assets/df1d6412-0f0a-4077-8d02-74d421f1141a" alt="Tesla Stock Prediction" width="800"/>

</div>

---

## 🧠 Technical Details

### Model Architecture
- **Network Type**: Long Short-Term Memory (LSTM)
- **Input Window**: 120 trading days
- **Prediction Horizon**: Next-day closing price
- **Optimization**: Keras Tuner with RandomSearch

### Data Processing
- **Source**: Yahoo Finance API via `yfinance`
- **Features**: OHLCV (Open, High, Low, Close, Volume)
- **Scaling**: MinMax normalization for neural network compatibility
- **Split**: 80% training, 20% validation

### Performance Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R² Score (Coefficient of Determination)

---

## 📦 Dependencies

Key libraries used in this project:

- **TensorFlow/Keras**: Deep learning framework
- **Keras Tuner**: Hyperparameter optimization
- **Streamlit**: Web application framework
- **yfinance**: Stock data retrieval
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **matplotlib/plotly**: Data visualization
- **scikit-learn**: Machine learning utilities

---

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free stock data API
- **TensorFlow Team** for the excellent deep learning framework
- **Streamlit** for making web app development simple and elegant

## 👨‍💻 Author

<div align="center">

**[A. Jayavanth](https://github.com/jayavanth18)**

[![GitHub](https://img.shields.io/badge/GitHub-jayavanth18-black?logo=github&logoColor=white)](https://github.com/jayavanth18)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jayavanth18/)

</div>

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

</div>

