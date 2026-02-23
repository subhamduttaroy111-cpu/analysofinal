import yfinance as yf
from datetime import datetime, timedelta

def get_stock_news(symbol):
    """
    Fetch recent news for a stock symbol.
    
    Args:
        symbol: Stock symbol with .NS suffix (e.g., "RELIANCE.NS")
    
    Returns:
        List of top 5 recent news items
    """
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        if not news:
            return []
        
        # Get top 5 news items
        top_news = []
        for item in news[:5]:
            # Try to get content, defaulting to item itself if content matches structure
            content = item.get('content', item) 
            
            # Helper to safely get nested text
            def get_val(obj, key, default=None):
                if not obj: return default
                return obj.get(key, default)

            # 1. Title
            title = get_val(content, 'title') or get_val(content, 'headline') or get_val(content, 'summary') or "Market News"
            
            # 2. Publisher
            provider = get_val(content, 'provider') or {}
            publisher = get_val(provider, 'displayName', 'Unknown Source')
            
            # 3. Link
            click_through = get_val(content, 'clickThroughUrl') or {}
            link = get_val(click_through, 'url') or item.get('link') or '#'
            
            # 4. Timestamp
            timestamp = get_val(content, 'pubDate') or item.get('providerPublishTime') or 0
            
            news_item = {
                "title": title[:150],
                "publisher": publisher,
                "link": link,
                "published": format_timestamp(timestamp)
            }
            top_news.append(news_item)
        
        return top_news
    
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        return []

def format_timestamp(timestamp):
    """Convert Unix timestamp to readable format"""
    try:
        if timestamp:
            dt = datetime.fromtimestamp(timestamp)
            # Calculate time ago
            now = datetime.now()
            diff = now - dt
            
            if diff.days > 0:
                return f"{diff.days}d ago"
            elif diff.seconds >= 3600:
                return f"{diff.seconds // 3600}h ago"
            else:
                return f"{diff.seconds // 60}m ago"
        return "Recently"
    except:
        return "Recently"
