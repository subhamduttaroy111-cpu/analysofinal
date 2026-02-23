def intraday_logic(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 30  # Start lower, require more conditions to reach high scores
    reasons = []
    
    # ========== MANDATORY FILTER: EMA Alignment ==========
    # This is now REQUIRED for a bullish signal
    ema_aligned = last['Close'] > last['EMA_9'] > last['EMA_21']
    
    if ema_aligned:
        score += 25
        reasons.append("✓ STRONG: Price > 9 EMA > 21 EMA (Trend Confirmed)")
    elif last['EMA_9'] > last['EMA_21']:
        score += 10
        reasons.append("→ Building momentum (9 > 21 EMA)")
    else:
        # Penalize if EMAs not aligned - prevents bullish signal
        score -= 10
        reasons.append("⚠ EMAs not aligned - weak trend")

    # ========== RSI: Stricter Zone + Overbought Rejection ==========
    if last['RSI'] > 75:
        # REJECT overbought - high chance of reversal
        score -= 15
        reasons.append("⚠ RSI OVERBOUGHT (>75) - Reversal Risk!")
    elif 55 <= last['RSI'] <= 70:
        score += 18
        reasons.append("✓ RSI in power zone (55-70)")
    elif 50 <= last['RSI'] < 55:
        score += 10
        reasons.append("→ RSI bullish (50-55)")
    elif last['RSI'] < 40:
        score -= 5
        reasons.append("⚠ RSI weak (<40)")

    # ========== MACD: Stronger Requirements ==========
    if last['MACD'] > last['MACD_Signal'] and last['MACD_Hist'] > prev['MACD_Hist'] and last['MACD_Hist'] > 0:
        score += 20
        reasons.append("✓ MACD strong bullish (histogram expanding)")
    elif last['MACD'] > last['MACD_Signal'] and last['MACD_Hist'] > prev['MACD_Hist']:
        score += 12
        reasons.append("✓ MACD bullish crossover")
    elif last['MACD'] > last['MACD_Signal']:
        score += 6
        reasons.append("→ MACD above signal")

    # ========== VOLUME: Higher Threshold (1.8x) ==========
    if last['Volume_Ratio'] > 1.8:
        score += 18
        reasons.append("✓ HIGH VOLUME breakout (>1.8x avg)")
    elif last['Volume_Ratio'] > 1.3:
        score += 10
        reasons.append("✓ Good volume (>1.3x avg)")
    elif last['Volume_Ratio'] > 1.0:
        score += 4
        reasons.append("→ Above avg volume")
    else:
        score -= 5
        reasons.append("⚠ Low volume - weak conviction")

    # ========== Bollinger Band Position ==========
    if last['Close'] > last['SMA_20'] and last['Close'] < last['BB_Upper']:
        score += 8
        reasons.append("✓ In upper BB zone (room to run)")
    elif last['Close'] > last['BB_Upper']:
        score -= 5
        reasons.append("⚠ Above BB Upper - overextended")

    # ========== BIAS DETERMINATION (Stricter Thresholds) ==========
    # Require higher scores for bullish, and EMA must be aligned
    if score >= 75 and ema_aligned:
        bias = "BULLISH"
    elif score >= 60:
        bias = "NEUTRAL"
    else:
        bias = "BEARISH"
    
    sl = round(last['Close'] - (last['ATR'] * 1.2), 2)
    tgt = round(last['Close'] + (last['ATR'] * 2.5), 2)
    
    risk = last['Close'] - sl
    reward = tgt - last['Close']
    rr_ratio = round(reward / risk, 2) if risk > 0 else 0

    return score, bias, reasons, sl, tgt, rr_ratio

def swing_logic(df):
    last = df.iloc[-1]
    score = 30  # Start lower for stricter filtering
    reasons = []

    # ========== MANDATORY FILTER: EMA Stack ==========
    ema_stacked = last['EMA_21'] > last['EMA_50'] > last['EMA_200']
    
    if ema_stacked:
        score += 30
        reasons.append("✓ STRONG: Perfect EMA stack (21>50>200)")
    elif last['EMA_21'] > last['EMA_50']:
        score += 15
        reasons.append("→ Mid-term uptrend (21>50)")
    else:
        score -= 10
        reasons.append("⚠ EMAs not stacked - weak structure")

    # ========== Above 200 EMA ==========
    if last['Close'] > last['EMA_200']:
        score += 15
        reasons.append("✓ Above 200 EMA (long-term bullish)")
    else:
        score -= 10
        reasons.append("⚠ Below 200 EMA - bearish territory")
    
    # ========== RSI: Stricter + Overbought Rejection ==========
    if last['RSI'] > 75:
        score -= 15
        reasons.append("⚠ RSI OVERBOUGHT (>75) - Reversal Risk!")
    elif 50 <= last['RSI'] <= 65:
        score += 15
        reasons.append("✓ RSI healthy zone (50-65)")
    elif 45 <= last['RSI'] < 50:
        score += 8
        reasons.append("→ RSI neutral-bullish")
    elif last['RSI'] < 40:
        score -= 5
        reasons.append("⚠ RSI weak (<40)")

    # ========== MACD: Stricter ==========
    if last['MACD'] > last['MACD_Signal'] and last['MACD'] > 0:
        score += 18
        reasons.append("✓ MACD bullish (above zero)")
    elif last['MACD'] > last['MACD_Signal']:
        score += 8
        reasons.append("→ MACD crossover")

    # ========== Momentum ==========
    if last['Momentum'] > 0:
        score += 8
        reasons.append("✓ Positive momentum")

    # ========== VOLUME: Higher Threshold ==========
    if last['Volume_Ratio'] > 1.5:
        score += 15
        reasons.append("✓ HIGH volume (>1.5x avg)")
    elif last['Volume_Ratio'] > 1.1:
        score += 8
        reasons.append("✓ Good volume")
    else:
        score -= 5
        reasons.append("⚠ Low volume - weak conviction")

    # ========== BIAS: Stricter Thresholds ==========
    if score >= 75 and ema_stacked:
        bias = "BULLISH"
    elif score >= 60:
        bias = "NEUTRAL"
    else:
        bias = "BEARISH"
    
    sl = round(last['Close'] - (last['ATR'] * 2.0), 2)
    tgt = round(last['Close'] + (last['ATR'] * 4.0), 2)
    
    risk = last['Close'] - sl
    reward = tgt - last['Close']
    rr_ratio = round(reward / risk, 2) if risk > 0 else 0

    return score, bias, reasons, sl, tgt, rr_ratio

def longterm_logic(df):
    last = df.iloc[-1]
    score = 40
    reasons = []

    if last['Close'] > last['EMA_200']:
        score += 30
        reasons.append("✓ Above 200 EMA")
        
        distance_pct = ((last['Close'] - last['EMA_200']) / last['EMA_200']) * 100
        if distance_pct < 15:
            score += 10
            reasons.append(f"✓ Not overextended ({distance_pct:.1f}%)")

    if last['EMA_50'] > last['EMA_200']:
        score += 15
        reasons.append("✓ 50>200 EMA")

    if 35 <= last['RSI'] <= 60:
        score += 15
        reasons.append("✓ RSI accumulation zone")
    elif last['RSI'] < 35:
        score += 8
        reasons.append("→ Oversold opportunity")

    if last['MACD'] > 0 and last['MACD'] > last['MACD_Signal']:
        score += 12
        reasons.append("✓ MACD positive")

    if last['Close'] < last['BB_Upper'] and last['Close'] > last['BB_Lower']:
        score += 8
        reasons.append("✓ Within BB range")

    bias = "BULLISH" if score >= 65 else "NEUTRAL" if score >= 50 else "BEARISH"
    
    sl = round(last['Close'] * 0.88, 2)
    tgt = round(last['Close'] * 1.35, 2)
    
    risk = last['Close'] - sl
    reward = tgt - last['Close']
    rr_ratio = round(reward / risk, 2) if risk > 0 else 0

    return score, bias, reasons, sl, tgt, rr_ratio


# ============================================================================
# AI-POWERED STRATEGIES
# ============================================================================

def intraday_logic_ai(df):
    """AI-powered intraday strategy with fallback to rule-based"""
    try:
        from model_manager import get_model_manager
        
        manager = get_model_manager()
        
        if manager.is_available():
            # Get AI prediction
            prediction = manager.predict(df, model_type='auto', strategy_mode='INTRADAY')
            
            if prediction['available']:
                last = df.iloc[-1]
                
                # Convert AI signal to score
                signal = prediction['signal']
                confidence = prediction['confidence']
                
                # Map confidence to score (60-100 range for better differentiation)
                base_score = {
                    'BULLISH': 80,
                    'NEUTRAL': 60,
                    'BEARISH': 40
                }[signal]
                
                # Adjust score based on confidence
                score = base_score + (confidence - 0.5) * 40
                score = max(0, min(100, score))
                
                # AI reasons
                reasons = [
                    f"🤖 AI Prediction: {signal}",
                    f"📊 Confidence: {confidence*100:.1f}%",
                    f"🔬 Model: {prediction.get('model', 'Ensemble')}"
                ]
                
                # Add probability breakdown
                proba = prediction.get('probabilities', {})
                reasons.append(f"📈 Probabilities: Bull {proba.get('BULLISH', 0)*100:.0f}% | Neutral {proba.get('NEUTRAL', 0)*100:.0f}% | Bear {proba.get('BEARISH', 0)*100:.0f}%")
                
                # Calculate stop loss and target (same as rule-based)
                sl = round(last['Close'] - (last['ATR'] * 1.2), 2)
                tgt = round(last['Close'] + (last['ATR'] * 2.5), 2)
                
                risk = last['Close'] - sl
                reward = tgt - last['Close']
                rr_ratio = round(reward / risk, 2) if risk > 0 else 0
                
                return score, signal, reasons, sl, tgt, rr_ratio
    
    except Exception as e:
        print(f"⚠️  AI model error: {e}")
    
    # Fallback to rule-based
    return intraday_logic(df)


def swing_logic_ai(df):
    """AI-powered swing strategy with fallback to rule-based"""
    try:
        from model_manager import get_model_manager
        
        manager = get_model_manager()
        
        if manager.is_available():
            prediction = manager.predict(df, model_type='auto', strategy_mode='SWING')
            
            if prediction['available']:
                last = df.iloc[-1]
                
                signal = prediction['signal']
                confidence = prediction['confidence']
                
                base_score = {
                    'BULLISH': 75,
                    'NEUTRAL': 60,
                    'BEARISH': 45
                }[signal]
                
                score = base_score + (confidence - 0.5) * 40
                score = max(0, min(100, score))
                
                reasons = [
                    f"🤖 AI Prediction: {signal}",
                    f"📊 Confidence: {confidence*100:.1f}%",
                    f"🔬 Model: {prediction.get('model', 'Ensemble')}"
                ]
                
                proba = prediction.get('probabilities', {})
                reasons.append(f"📈 Probabilities: Bull {proba.get('BULLISH', 0)*100:.0f}% | Neutral {proba.get('NEUTRAL', 0)*100:.0f}% | Bear {proba.get('BEARISH', 0)*100:.0f}%")
                
                sl = round(last['Close'] - (last['ATR'] * 2.0), 2)
                tgt = round(last['Close'] + (last['ATR'] * 4.0), 2)
                
                risk = last['Close'] - sl
                reward = tgt - last['Close']
                rr_ratio = round(reward / risk, 2) if risk > 0 else 0
                
                return score, signal, reasons, sl, tgt, rr_ratio
    
    except Exception as e:
        print(f"⚠️  AI model error: {e}")
    
    return swing_logic(df)


def longterm_logic_ai(df):
    """AI-powered long-term strategy with fallback to rule-based"""
    try:
        from model_manager import get_model_manager
        
        manager = get_model_manager()
        
        if manager.is_available():
            prediction = manager.predict(df, model_type='auto', strategy_mode='LONGTERM')
            
            if prediction['available']:
                last = df.iloc[-1]
                
                signal = prediction['signal']
                confidence = prediction['confidence']
                
                base_score = {
                    'BULLISH': 70,
                    'NEUTRAL': 55,
                    'BEARISH': 40
                }[signal]
                
                score = base_score + (confidence - 0.5) * 40
                score = max(0, min(100, score))
                
                reasons = [
                    f"🤖 AI Prediction: {signal}",
                    f"📊 Confidence: {confidence*100:.1f}%",
                    f"🔬 Model: {prediction.get('model', 'Ensemble')}"
                ]
                
                proba = prediction.get('probabilities', {})
                reasons.append(f"📈 Probabilities: Bull {proba.get('BULLISH', 0)*100:.0f}% | Neutral {proba.get('NEUTRAL', 0)*100:.0f}% | Bear {proba.get('BEARISH', 0)*100:.0f}%")
                
                sl = round(last['Close'] * 0.88, 2)
                tgt = round(last['Close'] * 1.35, 2)
                
                risk = last['Close'] - sl
                reward = tgt - last['Close']
                rr_ratio = round(reward / risk, 2) if risk > 0 else 0
                
                return score, signal, reasons, sl, tgt, rr_ratio
    
    except Exception as e:
        print(f"⚠️  AI model error: {e}")
    
    return longterm_logic(df)

