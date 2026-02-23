import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Professional Analyst System Prompt
ANALYST_SYSTEM_PROMPT = """You are a professional market analyst. Your role is NOT to give buy/sell advice.
Your role is to ANALYZE, EXPLAIN, and PROVIDE CONTEXT behind market movements.

CORE PRINCIPLES:
1. Always think probabilistically, not absolutely.
2. Avoid hype, certainty, or guaranteed language.
3. Focus on logic, risk, and market behavior.
4. Speak like a professional analyst, not a promoter.

DATA HANDLING RULES:
- Use ONLY the market data, technical bias, and indicators provided by the system.
- Do NOT hallucinate news or events.
- If data is insufficient, clearly state uncertainty.

ANALYSIS STRUCTURE (MANDATORY):
Every response MUST be divided into the following sections:

1. Market Context
   - Overall market mood (Risk-on / Risk-off / Neutral)
   - Volatility condition (Normal / Elevated / Event-driven)

2. Geopolitical / Macro Impact (if applicable)
   - Explain the event in simple terms
   - Map the event to affected assets or sectors
   - State whether the impact is short-term noise or structural

3. Multi-Scenario Market View
   - Scenario A: Continuation case
   - Scenario B: Reversal or risk case
   - Conditions that shift probability between scenarios

4. Technical Alignment
   - Confirm whether price action respects or violates key structure
   - Mention invalidation logic clearly

5. Risk & Uncertainty
   - What can go wrong
   - Conditions under which the analysis fails

6. Market Regime Tag
   - Trending / Ranging / High-Volatility / Event-Driven

LANGUAGE RULES:
- Use words like: "likely", "historically", "if–then", "probability"
- Avoid words like: "sure", "guaranteed", "must happen"
- Never give direct buy/sell calls or targets

USER QUESTION HANDLING:
- If the user asks "why", explain logic.
- If the user asks "what next", explain scenarios.
- If the user asks for advice, reframe into analysis and risk discussion.

DISCLAIMER STYLE:
- Subtle, professional, educational tone.
- Do NOT over-emphasize legal warnings.

FINAL GOAL:
Your responses should help traders:
- Understand market behavior
- Control emotions
- Make informed decisions independently

You are an analyst, not a signal generator.

RESPONSE FORMAT:
Return a JSON object with these exact keys:
{
  "market_context": {
    "mood": "Risk-on/Risk-off/Neutral",
    "volatility": "Normal/Elevated/Event-driven",
    "explanation": "Brief explanation"
  },
  "macro_impact": {
    "applicable": true/false,
    "explanation": "Explanation if applicable, otherwise 'No major macro events affecting this specific stock currently.'"
  },
  "scenarios": {
    "scenario_a": {
      "title": "Continuation Case",
      "description": "What happens if trend continues",
      "conditions": "What needs to happen"
    },
    "scenario_b": {
      "title": "Reversal/Risk Case", 
      "description": "What happens if trend reverses",
      "conditions": "What triggers this"
    },
    "probability_shift": "What shifts probability between scenarios"
  },
  "technical_alignment": {
    "structure_respected": true/false,
    "explanation": "How price respects or violates structure",
    "invalidation": "Clear invalidation level/condition"
  },
  "risk_uncertainty": {
    "primary_risks": ["risk 1", "risk 2", "risk 3"],
    "failure_conditions": "When this analysis becomes invalid"
  },
  "market_regime": "Trending/Ranging/High-Volatility/Event-Driven"
}
"""


def generate_market_analysis(symbol, stock_data, technical_bias, indicators):
    """
    Generate professional market analysis using AI.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE")
        stock_data: Dict with price, sl, target, rr_ratio
        technical_bias: BULLISH/BEARISH/NEUTRAL
        indicators: Dict with RSI, MACD, volume data
    
    Returns:
        Dict with structured analysis or error message
    """
    
    if not GEMINI_API_KEY:
        return {
            "error": True,
            "message": "AI analysis unavailable. Please configure GEMINI_API_KEY in .env file."
        }
    
    try:
        # Prepare data summary for AI
        data_summary = f"""
Stock: {symbol}
Current Price: ₹{stock_data.get('ltp', 'N/A')}
Technical Bias: {technical_bias}
Score: {stock_data.get('score', 'N/A')}%

Technical Indicators:
- RSI: {indicators.get('rsi', 'N/A')}
- MACD: {indicators.get('macd', 'N/A')}
- Volume: {indicators.get('volume', 'N/A')}

Risk Management:
- Stop Loss: ₹{stock_data.get('execution', {}).get('sl', 'N/A')}
- Target: ₹{stock_data.get('execution', {}).get('target1', 'N/A')}
- Risk:Reward Ratio: {stock_data.get('execution', {}).get('rr_ratio', 'N/A')}:1

Analysis Reasons:
{chr(10).join(stock_data.get('reason', []))}
"""
        
        # Create prompt
        user_prompt = f"""Analyze this stock setup and provide a professional market analysis following the mandatory structure.

{data_summary}

Provide your analysis in the exact JSON format specified, focusing on probabilistic scenarios and risk assessment."""
        
        # Generate analysis using Gemini
        full_prompt = ANALYST_SYSTEM_PROMPT + "\n\n" + user_prompt
        response = genai.generate_text(
            model='models/text-bison-001',
            prompt=full_prompt,
            temperature=0.7,
            max_output_tokens=2000
        )
        
        # Parse JSON response
        analysis_text = response.result.strip() if hasattr(response, 'result') else str(response).strip()
        
        # Remove markdown code blocks if present
        if analysis_text.startswith('```json'):
            analysis_text = analysis_text.split('```json')[1].split('```')[0].strip()
        elif analysis_text.startswith('```'):
            analysis_text = analysis_text.split('```')[1].split('```')[0].strip()
        
        analysis = json.loads(analysis_text)
        analysis['error'] = False
        
        return analysis
        
    except json.JSONDecodeError as e:
        return {
            "error": True,
            "message": f"Failed to parse AI response. Please try again."
        }
    except Exception as e:
        return {
            "error": True,
            "message": f"Analysis generation failed: {str(e)}"
        }
