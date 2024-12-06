import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error

class MLPredictor:
    def __init__(self, data):
        self.data = data
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self, look_back=60):
        scaled_data = self.scaler.fit_transform(self.data['Close'].values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test

    def create_model(self, look_back=60):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def predict(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        model = self.create_model()
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
        
        y_pred = model.predict(X_test)
        y_pred = self.scaler.inverse_transform(y_pred)
        y_test = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        model_score = 1 - (mse / np.var(y_test))  # R-squared
        
        # Predict next day's price
        last_60_days = self.data['Close'].values[-60:]
        last_60_days_scaled = self.scaler.transform(last_60_days.reshape(-1, 1))
        X_predict = np.array([last_60_days_scaled])
        X_predict = np.reshape(X_predict, (X_predict.shape[0], X_predict.shape[1], 1))
        predicted_price = model.predict(X_predict)
        predicted_price = self.scaler.inverse_transform(predicted_price)[0][0]
        
        return predicted_price, model_score, y_test, y_pred, mae, mse
