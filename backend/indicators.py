import pandas as pd
import numpy as np

def add_indicators(df):
    """Enhanced technical indicators"""
    
    # Moving Averages
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    df['BB_Upper'] = df['SMA_20'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['Close'].rolling(window=20).std() * 2)
    
    # Volume Analysis
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    
    # VWAP - Used for intraday trend direction
    df['Cum_Vol'] = df['Volume'].cumsum()
    df['Cum_VolPrice'] = (df['Close'] * df['Volume']).cumsum()
    df['VWAP'] = df['Cum_VolPrice'] / df['Cum_Vol']

    # SuperTrend - Used for swing trading signals
    atr_period = 10
    multiplier = 3.0
    hl2 = (df['High'] + df['Low']) / 2

    # Calculate ATR for SuperTrend (separate from existing ATR_14)
    high_low_st = df['High'] - df['Low']
    high_close_st = np.abs(df['High'] - df['Close'].shift())
    low_close_st = np.abs(df['Low'] - df['Close'].shift())
    ranges_st = pd.concat([high_low_st, high_close_st, low_close_st], axis=1)
    true_range_st = np.max(ranges_st, axis=1)
    atr_st = true_range_st.rolling(atr_period).mean()

    # SuperTrend bands
    basic_upper = hl2 + (multiplier * atr_st)
    basic_lower = hl2 - (multiplier * atr_st)

    supertrend = pd.Series(index=df.index, dtype=float)
    supertrend_direction = pd.Series(index=df.index, dtype=float)

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    for i in range(1, len(df)):
        if basic_upper.iloc[i] < final_upper.iloc[i-1] or df['Close'].iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]
        
        if basic_lower.iloc[i] > final_lower.iloc[i-1] or df['Close'].iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]

    for i in range(1, len(df)):
        if df['Close'].iloc[i] > final_upper.iloc[i-1]:
            supertrend_direction.iloc[i] = 1  # Bullish
            supertrend.iloc[i] = final_lower.iloc[i]
        elif df['Close'].iloc[i] < final_lower.iloc[i-1]:
            supertrend_direction.iloc[i] = -1  # Bearish
            supertrend.iloc[i] = final_upper.iloc[i]
        else:
            supertrend_direction.iloc[i] = supertrend_direction.iloc[i-1]
            if supertrend_direction.iloc[i] == 1:
                supertrend.iloc[i] = final_lower.iloc[i]
            else:
                supertrend.iloc[i] = final_upper.iloc[i]

    df['SuperTrend'] = supertrend
    df['SuperTrend_Direction'] = supertrend_direction

    # EMA 5 for swing trading
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()

    # Volume spike detection
    df['Volume_Spike'] = df['Volume_Ratio'] > 2.0

    # Price distance from 200 EMA in percent
    df['Dist_200EMA_Pct'] = ((df['Close'] - df['EMA_200']) / df['EMA_200']) * 100

    # MACD histogram increasing for 3 candles
    df['MACD_Hist_Rising'] = (
        (df['MACD_Hist'] > df['MACD_Hist'].shift(1)) &
        (df['MACD_Hist'].shift(1) > df['MACD_Hist'].shift(2))
    )

    return df
