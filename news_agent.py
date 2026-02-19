import os
import yfinance as yf


def search_news(ticker: str, lookback_days: int = 3) -> list[str]:
    """Fetch recent news headlines for a ticker via yfinance."""
    try:
        t = yf.Ticker(ticker)
        news_items = t.news or []
        headlines = []
        for item in news_items:
            # yfinance >=1.2: title is nested under item["content"]["title"]
            content = item.get("content", {})
            title = content.get("title", "") if isinstance(content, dict) else ""
            # yfinance <1.0: title is at top level
            if not title:
                title = item.get("title", "")
            if title:
                headlines.append(title)
        return headlines[:20]
    except Exception:
        return []


def analyze_sentiment(ticker: str, news_items: list[str]) -> float:
    """Call Claude API to score headline sentiment as a float in [-1, 1]."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or not news_items:
        return 0.0

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        headlines_text = "\n".join(f"- {h}" for h in news_items)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=64,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"You are a financial sentiment analyst. Given these recent news "
                        f"headlines about {ticker}, respond with ONLY a single float between "
                        f"-1.0 (very bearish) and 1.0 (very bullish). No explanation.\n\n"
                        f"{headlines_text}"
                    ),
                }
            ],
        )

        raw = message.content[0].text.strip()
        score = float(raw)
        return max(-1.0, min(1.0, score))
    except Exception:
        return 0.0


def analyze(ticker: str, date: str = "", lookback_days: int = 3) -> float:
    """Return a sentiment signal in [-1, 1] for the given ticker.

    The date parameter is accepted for interface consistency but yfinance
    news always returns the most recent headlines.
    """
    headlines = search_news(ticker, lookback_days)
    return round(analyze_sentiment(ticker, headlines), 4)
