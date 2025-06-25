import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load data based on user input
@st.cache_data
def load_data(ticker):
    df = pd.read_csv(f"{ticker}.csv")
    
    # Handle Yahoo-style datasets with Date column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    df.ffill(inplace=True)
    return df

# ML Models
def knn_model(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

def linear_regression_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.score(X_test, y_test), model.predict(X_test)

def bayes_classifier(X_train, X_test, y_train, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

def plot_prediction(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test.index, y_test.values, label='Actual', color='blue')
    ax.plot(y_test.index, y_pred, label='Predicted', color='red')
    ax.set_title('Actual vs Predicted Close Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    return fig

# --- Streamlit App ---
st.title("📈 Stock Market Prediction System")

# Select ticker
ticker = st.selectbox("Choose Stock Ticker", ["AAPL", "GOOG", "TSLA", "MSFT", "AMZN"])

# Load dataset
df = load_data(ticker)

st.subheader(f"{ticker} Dataset Preview")
st.dataframe(df.head())

# Feature engineering
print(df.columns)
features = df[['Open', 'High', 'Low', 'Volume']]
labels = (df['Close'] > df['Open']).astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)

# ML Results
st.subheader("📊 Model Results")

knn_acc = knn_model(X_train, X_test, y_train, y_test)
st.write(f"**KNN Accuracy:** {knn_acc:.2f}")

bayes_acc = bayes_classifier(X_train, X_test, y_train, y_test)
st.write(f"**Naive Bayes Accuracy:** {bayes_acc:.2f}")

y_train_reg = df['Close'].loc[y_train.index]
y_test_reg = df['Close'].loc[y_test.index]
lr_score, y_pred_lr = linear_regression_model(X_train, X_test, y_train_reg, y_test_reg)
st.write(f"**Linear Regression :** {lr_score:.2f}")

# Plot
st.subheader("📉 Linear Regression: Actual vs Predicted")
st.pyplot(plot_prediction(y_test_reg, y_pred_lr))

    
   

