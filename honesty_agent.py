import os
import json
import requests
import yfinance as yf


# SEC EDGAR requires a User-Agent header with contact info
SEC_HEADERS = {
    "User-Agent": "StockSimulator/1.0 (contact@example.com)",
    "Accept": "application/json",
}


def fetch_executive_statements(ticker: str) -> list[str]:
    """Search for recent C-level executive quotes and earnings call excerpts.

    Uses yfinance news as the primary source for recent executive statements.
    """
    statements = []
    try:
        t = yf.Ticker(ticker)
        news_items = t.news or []
        for item in news_items:
            content = item.get("content", {})
            title = content.get("title", "") if isinstance(content, dict) else ""
            if not title:
                title = item.get("title", "")
            summary = ""
            if isinstance(content, dict):
                summary = content.get("summary", "")
            if not summary:
                summary = item.get("summary", "")
            combined = f"{title}. {summary}".strip() if summary else title
            if combined:
                statements.append(combined)
        return statements[:25]
    except Exception:
        return []


def fetch_sec_filings(ticker: str) -> str:
    """Pull the Management Discussion & Analysis section from the most recent
    10-K or 10-Q filing via the SEC EDGAR full-text search API.
    """
    try:
        # Look up the company CIK
        cik_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2024-01-01&forms=10-K,10-Q"
        search_url = (
            f"https://efts.sec.gov/LATEST/search-index?"
            f"q=%22{ticker}%22&forms=10-K,10-Q&dateRange=custom"
            f"&startdt=2024-01-01"
        )

        # Use EDGAR company search
        company_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company=&CIK={ticker}&type=10-K&dateb=&owner=include&count=1&search_text=&action=getcompany"

        # Try the EDGAR full-text search API
        search_api = (
            f"https://efts.sec.gov/LATEST/search-index?"
            f"q=%22{ticker}%22&forms=10-K,10-Q"
        )

        # Use the simpler EDGAR company filings API
        tickers_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K&dateb=&owner=include&count=1&search_text=&action=getcompany&output=atom"
        resp = requests.get(tickers_url, headers=SEC_HEADERS, timeout=10)

        if resp.status_code != 200:
            return ""

        # Try the EDGAR full-text search for MD&A content
        search_resp = requests.get(
            f"https://efts.sec.gov/LATEST/search-index?q=%22management+discussion%22+%22{ticker}%22&forms=10-K,10-Q",
            headers=SEC_HEADERS,
            timeout=10,
        )

        if search_resp.status_code == 200:
            try:
                data = search_resp.json()
                hits = data.get("hits", {}).get("hits", [])
                if hits:
                    filing_url = hits[0].get("_source", {}).get("file_url", "")
                    if filing_url:
                        doc_resp = requests.get(
                            f"https://www.sec.gov{filing_url}",
                            headers=SEC_HEADERS,
                            timeout=15,
                        )
                        if doc_resp.status_code == 200:
                            text = doc_resp.text[:15000]
                            return text
            except (json.JSONDecodeError, KeyError):
                pass

        return ""
    except Exception:
        return ""


def fetch_financials(ticker: str) -> dict:
    """Get actual financial data from yfinance for cross-referencing executive claims.

    Returns revenue, earnings, margins, and guidance-related data.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        financials = {
            "revenue": info.get("totalRevenue"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings": info.get("netIncomeToCommon"),
            "earnings_growth": info.get("earningsGrowth"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "gross_margin": info.get("grossMargins"),
            "forward_pe": info.get("forwardPE"),
            "trailing_pe": info.get("trailingPE"),
            "analyst_target": info.get("targetMeanPrice"),
            "current_price": info.get("currentPrice"),
            "recommendation": info.get("recommendationKey"),
            "debt_to_equity": info.get("debtToEquity"),
            "free_cash_flow": info.get("freeCashflow"),
        }

        # Get quarterly earnings history for guidance accuracy
        try:
            earnings_hist = t.earnings_history
            if earnings_hist is not None and not earnings_hist.empty:
                surprises = []
                for _, row in earnings_hist.iterrows():
                    est = row.get("epsEstimate")
                    actual = row.get("epsActual")
                    if est and actual and est != 0:
                        surprise_pct = (actual - est) / abs(est)
                        surprises.append(round(surprise_pct, 4))
                financials["earnings_surprises"] = surprises
        except Exception:
            financials["earnings_surprises"] = []

        return financials
    except Exception:
        return {}


def assess_veracity(
    ticker: str,
    statements: list[str],
    filings_text: str,
    financials: dict,
) -> dict:
    """Use Claude API to assess veracity of executive statements by combining
    linguistic analysis with cross-referencing against actual financial data.

    Returns:
        {
            'score': float [-1, 1],
            'consistency': float [-1, 1],
            'hedging': float [-1, 1],
            'guidance_accuracy': float [-1, 1],
            'flagged_statements': list[str]
        }
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _neutral_result()

    if not statements and not filings_text:
        return _neutral_result()

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        statements_text = "\n".join(f"- {s}" for s in statements) if statements else "No recent statements found."
        filings_summary = filings_text[:5000] if filings_text else "No SEC filings retrieved."

        financials_text = json.dumps(
            {k: v for k, v in financials.items() if v is not None},
            indent=2,
            default=str,
        ) if financials else "No financial data available."

        prompt = f"""You are a forensic financial analyst specializing in executive credibility assessment.

Analyze the following data for {ticker} and assess the veracity of C-level executive communications.

## Recent Executive Statements & News
{statements_text}

## SEC Filing Excerpts (MD&A)
{filings_summary}

## Actual Financial Data
{financials_text}

## Your Task
Assess executive credibility across three dimensions:

1. **Consistency** [-1 to 1]: Are executive statements consistent with each other and over time? Do they contradict prior guidance or change narratives without acknowledgment?
   - Positive = consistent and transparent
   - Negative = contradictory or shifting narratives

2. **Hedging** [-1 to 1]: How much hedging, vague language, or deflection is used?
   - Positive = direct and specific language
   - Negative = excessive hedging, weasel words, vague promises

3. **Guidance Accuracy** [-1 to 1]: How well do executive claims match actual financial performance? Compare stated expectations against the real numbers.
   - Positive = guidance matches or conservatively beats actuals
   - Negative = persistent over-promising vs actual results

Also identify any specific flagged statements that are misleading, contradictory, or unsupported by the data.

Respond in this exact JSON format only:
{{
  "score": <overall credibility score, float -1 to 1>,
  "consistency": <float -1 to 1>,
  "hedging": <float -1 to 1>,
  "guidance_accuracy": <float -1 to 1>,
  "flagged_statements": ["statement 1 concern", "statement 2 concern"]
}}"""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = message.content[0].text.strip()
        # Extract JSON from response (handle markdown code blocks)
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)

        # Validate and clamp scores
        for key in ("score", "consistency", "hedging", "guidance_accuracy"):
            val = float(result.get(key, 0))
            result[key] = max(-1.0, min(1.0, round(val, 4)))

        if "flagged_statements" not in result:
            result["flagged_statements"] = []

        return result

    except Exception:
        return _neutral_result()


def _neutral_result() -> dict:
    """Return a neutral result when analysis cannot be performed."""
    return {
        "score": 0.0,
        "consistency": 0.0,
        "hedging": 0.0,
        "guidance_accuracy": 0.0,
        "flagged_statements": [],
    }


def analyze(ticker: str) -> float:
    """Public interface matching the agent pattern. Returns honesty signal in [-1, 1]."""
    result = analyze_detailed(ticker)
    return result["score"]


def analyze_detailed(ticker: str) -> dict:
    """Full analysis with score breakdown.

    Returns:
        {
            'score': float [-1, 1],
            'consistency': float [-1, 1],
            'hedging': float [-1, 1],
            'guidance_accuracy': float [-1, 1],
            'flagged_statements': list[str]
        }
    """
    statements = fetch_executive_statements(ticker)
    filings_text = fetch_sec_filings(ticker)
    financials = fetch_financials(ticker)
    return assess_veracity(ticker, statements, filings_text, financials)
