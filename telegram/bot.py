# stock_trading_bot.py
import os
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import telegram
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import alpaca_trade_api as tradeapi
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Configuration
from dotenv import load_dotenv
import os

load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ALPACA_API_KEY = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("APCA_API_BASE_URL")

# Initialize Alpaca API
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

# Neural Network Model Architecture
class StockPredictor(nn.Module):
    def __init__(self, input_size=15, hidden_size=128, num_layers=3, output_size=3):
        super(StockPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Advanced Data Processor
class StockDataEngine:
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'bollinger_upper', 'bollinger_lower',
            'atr', 'obv', 'vwap', 'ema_20', 'ema_50', 'cci'
        ]
    
    async def prepare_data(self, symbol, timeframe='15Min', lookback=100):
        data = await self._fetch_alpaca_data(symbol, timeframe, lookback)
        features = self._create_technical_indicators(data)
        scaled_data = self._scale_features(features)
        return self._create_sequences(scaled_data)
    
    def _create_technical_indicators(self, df):
        # Calculate multiple technical indicators
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'], df['signal'] = self._calculate_macd(df['close'])
        df['bollinger_upper'], df['bollinger_lower'] = self._bollinger_bands(df['close'])
        df['atr'] = self._average_true_range(df)
        df['obv'] = self._on_balance_volume(df)
        df['vwap'] = self._calculate_vwap(df)
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['cci'] = self._commodity_channel_index(df)
        return df[self.feature_columns].dropna()

    def _scale_features(self, df):
        return pd.DataFrame(self.scaler.fit_transform(df), 
                          columns=df.columns, index=df.index)
    
    def _create_sequences(self, data, window_size=30):
        sequences = []
        for i in range(len(data) - window_size):
            sequences.append(data[i:i+window_size].values)
        return np.array(sequences)

# Trading Strategy Core
class StockTradingSystem:
    def __init__(self):
        self.model = self.load_model()  # Calling load_model()

    def load_model(self):
        print("Loading model...")  
        return None  # Replace with actual model loading code
        self.data_engine = StockDataEngine()
        self.risk_params = {
            'max_position_size': 0.1,  # 10% of portfolio per trade
            'stop_loss': 0.03,         # 3% stop loss
            'take_profit': 0.05        # 5% take profit
        }
    
    async def generate_signal(self, symbol):
        processed_data = await self.data_engine.prepare_data(symbol)
        prediction = self.model.predict(processed_data[-1].unsqueeze(0))
        return self._interpret_prediction(prediction)
    
    def _interpret_prediction(self, prediction):
        # Returns (action, confidence, price_targets)
        probabilities = torch.softmax(prediction, dim=1)
        action_idx = torch.argmax(probabilities).item()
        actions = ['BUY', 'SELL', 'HOLD']
        return (
            actions[action_idx],
            probabilities[0][action_idx].item(),
            self._calculate_price_targets(prediction)
        )
    
    def _calculate_price_targets(self, prediction):
        # Implement regression targets for price predictions
        pass

# Telegram Interface
class StockTradingBot:
    def __init__(self):
        self.application = Application.builder().token(TELEGRAM_TOKEN).build()
        self.trading_system = StockTradingSystem()
        self._register_commands()
    
    def _register_commands(self):
        self.application.add_handler(CommandHandler("analyze", self.analyze_handler))
        self.application.add_handler(CommandHandler("signals", self.signals_handler))
        self.application.add_handler(CommandHandler("portfolio", self.portfolio_handler))
    
    async def analyze_handler(self, update, context):
        symbol = context.args[0].upper() if context.args else 'SPY'
        action, confidence, targets = await self.trading_system.generate_signal(symbol)
        message = (
            f"📊 {symbol} Analysis:\n"
            f"Action: {action} (Confidence: {confidence:.2%})\n"
            f"Targets: Entry: ${targets['entry']:.2f} | "
            f"Stop: ${targets['stop_loss']:.2f} | "
            f"Take Profit: ${targets['take_profit']:.2f}"
        )
        await update.message.reply_text(message)
    
    async def portfolio_handler(self, update, context):
        # Implement portfolio management features
        pass
    
    async def signals_handler(self, update, context):
        # Implement real-time signal monitoring
        pass

# Execution
if __name__ == '__main__':
    bot = StockTradingBot()
    print("Stock Trading Bot Activated")
    asyncio.run(bot.application.run_polling())
