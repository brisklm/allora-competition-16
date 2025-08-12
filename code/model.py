import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense  # For LSTM hybrid
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from config import SELECTED_FEATURES, MODEL_PARAMS, training_price_data_path, USE_SYNTHETIC_DATA, SYNTHETIC_DATA_PATH  # Import updated config

# Initialize VADER for sentiment
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def add_vader_sentiment(df):
    df['sentiment_score'] = df['some_text_column'].apply(lambda x: sia.polarity_scores(x)['compound'])  # Assume a text column; adapt as needed
    return df

def train_model():
    # Load and blend data
    real_data = pd.read_csv(training_price_data_path)
    if USE_SYNTHETIC_DATA:
        synthetic_data = pd.read_csv(SYNTHETIC_DATA_PATH)
        data = pd.concat([real_data, synthetic_data]).dropna()  # Blend and fix NaNs
    else:
        data = real_data.dropna()  # Fix NaNs
    
    # Add VADER sentiment
    data = add_vader_sentiment(data)
    
    # Feature engineering and fix low variance
    data = data[SELECTED_FEATURES]  # Use selected features
    data = data.loc[:, data.var() > 0.1]  # Remove low variance features
    
    # LSTM hybrid: Create sequences for LSTM
    def create_lstm_features(data):
        X_lstm = []
        for i in range(30, len(data)):
            X_lstm.append(data.iloc[i-30:i].values)  # 30-day window
        X_lstm = np.array(X_lstm)
        return X_lstm
    
    X_lstm = create_lstm_features(data)
    lstm_model = Sequential([
        LSTM(50, input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_lstm, data['target'][30:], epochs=10)  # Train LSTM
    
    lstm_output = lstm_model.predict(X_lstm)  # Get LSTM output as feature
    data['lstm_output'] = np.concatenate([np.zeros(30), lstm_output.flatten()])
    
    # Optuna tuning
    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 30, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000)
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        return model.score(X_val, y_val)  # Use for ZPTAE reduction
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    best_params = study.best_params
    
    # Final model with best params
    final_model = LGBMRegressor(**{**MODEL_PARAMS, **best_params})
    X = data[SELECTED_FEATURES]
    y = data['target']  # Assume target is log-return
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    final_model.fit(X_train, y_train)
    return final_model

def get_inference(model, input_data):
    return model.predict(input_data)