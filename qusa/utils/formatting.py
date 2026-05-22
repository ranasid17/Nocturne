# qusa/qusa/utils/formatting.py

"""
Utilities for pretty-printing and formatting CLI output.
"""

def format_header(title, width=80, char="="):
    """
    Format a section header.
    
    Example:
    ================================================================================
    TITLE
    ================================================================================
    """
    border = char * width
    return f"\n{border}\n{title.upper()}\n{border}"


def format_box(lines, title=None, width=80):
    """
    Format text inside an ASCII box.
    """
    top_border = f"┌{'─' * (width - 2)}┐"
    bottom_border = f"└{'─' * (width - 2)}┘"
    sep = f"├{'─' * (width - 2)}┤"
    
    output = [top_border]
    
    if title:
        output.append(f"│ {title.ljust(width - 4)} │")
        output.append(sep)
        
    for line in lines:
        output.append(f"│ {line.ljust(width - 4)} │")
        
    output.append(bottom_border)
    return "\n".join(output)


def format_prediction_card(prediction, ticker=None, width=50):
    """
    Format a prediction result as a card.
    """
    title = f"PREDICTION: {ticker}" if ticker else "PREDICTION"
    
    lines = []
    if prediction.get("date"):
        # Format date if it's a timestamp
        date_val = prediction["date"]
        if hasattr(date_val, "strftime"):
            date_str = date_val.strftime("%Y-%m-%d")
        else:
            date_str = str(date_val)
        lines.append(f"Date:       {date_str}")
    
    lines.append(f"Direction:  {prediction.get('direction', 'N/A')}")
    
    prob = prediction.get("probability_up", 0)
    lines.append(f"Probability: {prob:.1%}")
    
    conf = prediction.get("confidence", "N/A")
    lines.append(f"Confidence: {conf}")
    
    vol_triggered = prediction.get("volatility_filter_triggered", False)
    if vol_triggered:
        atr_val = prediction.get("atr_pct", 0.0)
        lines.append(f"ATR %:       {atr_val:.2f}% (EXCEEDS LIMIT)")
    
    lines.append("") # Spacer
    
    # Signal interpretation
    if vol_triggered:
        lines.append("⚠️ VOLATILITY FILTER - NO TRADE")
    elif conf == "HIGH":
        if prediction.get("prediction") == 1:
            lines.append("🚀 STRONG BUY SIGNAL")
        else:
            lines.append("📉 STRONG SELL SIGNAL")
    else:
        lines.append("⚖️ LOW CONFIDENCE - NO CLEAR SIGNAL")
        
    return format_box(lines, title=title, width=width)
