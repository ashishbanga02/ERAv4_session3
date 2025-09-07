import os
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, Response
from nsepython import nse_optionchain_scrapper
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))  # read from OS env
model = genai.GenerativeModel("gemini-2.5-flash")

app = Flask(__name__)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

NSE_SYMBOLS_FILE = "nse_symbols.json"  # Local file with derivative symbols


# -------- SYMBOLS ----------
def get_default_derivative_symbols():
    return [
        "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
        "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", "LT"
    ]

def load_nse_derivative_symbols(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                symbols = json.load(f)
                if isinstance(symbols, list):
                    return sorted(set(symbols))
                return get_default_derivative_symbols()
            except:
                return get_default_derivative_symbols()
    return get_default_derivative_symbols()

def get_derivative_symbols():
    return load_nse_derivative_symbols(NSE_SYMBOLS_FILE)


# -------- CACHE ----------
def fetch_and_cache_options_data(symbol, expiry_date):
    file_path = os.path.join(DATA_DIR, f"{symbol}_{expiry_date}.json")
    current_time = datetime.now()

    if os.path.exists(file_path):
        last_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
        if (current_time - last_modified) < timedelta(minutes=30):
            with open(file_path, "r") as f:
                return json.load(f)

    try:
        options_data = nse_optionchain_scrapper(symbol)
        options_data["fetch_timestamp"] = current_time.isoformat()
        with open(file_path, "w") as f:
            json.dump(options_data, f, indent=4)
        return options_data
    except Exception as e:
        print(f"Error fetching {symbol} {expiry_date}: {e}")
        return None


# -------- GEMINI STREAM ----------
def stream_analysis_with_gemini(symbol, expiry_date, options_data_json):
    current_date = datetime.now().strftime("%Y-%m-%d") # Provide current date to Gemini
    prompt = f"""
    You are an expert options trader. Provide a highly detailed, insightful, and actionable analysis of the following NSE options chain data for **{symbol}** with an expiry date of **{expiry_date}**.
    The current date is {current_date}.

    Format your response using **Markdown**, ensuring excellent readability with clear headings, bolded key terms, and well-structured bullet points. Each bullet point should be comprehensive and provide sufficient detail for understanding.

    Specifically:
    *   **Market Snapshot & Expiry Context:** Start with an engaging title. Clearly state the symbol and expiry date, and provide a detailed assessment of whether this is a near-term, mid-term, or long-dated expiry relative to the current date. Discuss its implications for time decay and volatility.
    *   **Top 3 Strategy Picks (Detailed):**
        *   Identify and thoroughly explain the top 3 potential options trading strategies (e.g., Iron Condor, Strangle, Bull Call Spread, Bear Put Spread, Covered Call, Protective Put, Butterfly Spread, Calendar Spread).
        *   For EACH strategy, elaborate on:
            *   **Why it's suggested:** Link directly to Open Interest (OI), Volume, and Implied Volatility (IV) trends observed in the data.
            *   **Market Outlook:** What kind of market movement (bullish, bearish, neutral, volatile) would favor this strategy?
            *   **Key Strikes/Levels:** Suggest potential strike prices or ranges based on the data.
            *   **Risk/Reward Profile:** Briefly explain the typical risk and reward characteristics.
    *   **Unusual Activity & Volatility Alert (In-depth):**
        *   Detect and describe any significant unusual activity. This includes large OI buildups or reductions on specific strike prices, unusually high or low trading volumes, or sudden and notable shifts in Implied Volatility (IV) across the chain.
        *   Explain what these anomalies might imply for future price movements or market sentiment.
    *   **Actionable Insights & Entry/Exit Considerations:**
        *   Suggest specific, actionable trading opportunities. This could involve identifying key support/resistance levels from OI, potential entry/exit conditions based on price action or IV, or conditions under which a strategy might be initiated or adjusted.
        *   Provide specific criteria for decision-making.
    *   **Credit Spreads Spotlight (High Liquidity & Premium):**
        *   Summarize in detail any attractive credit spreads (call + put) that exhibit both high liquidity (significant OI and Volume) and offer good premium capture opportunities.
        *   For each suggested credit spread, specify the likely strike prices, the rationale behind the selection, and the expected premium.

    Data:
    {json.dumps(options_data_json, indent=2)}
    """
    try:
        response_stream = model.generate_content(prompt, stream=True)
        for chunk in response_stream:
            if chunk.candidates and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, "text"):
                        yield f"data: {part.text}\n\n"
        yield "data: [END]\n\n"
    except Exception as e:
        yield f"data: Error analyzing with Gemini: {e}\n\n"


# -------- ROUTES ----------
@app.route("/")
def index():
    symbols = get_derivative_symbols()
    return render_template("index.html", symbols=symbols)

@app.route("/get_expiries", methods=["POST"])
def get_expiries():
    symbol = request.form["symbol"]
    try:
        options_data = nse_optionchain_scrapper(symbol)
        raw_expiries = options_data["records"]["expiryDates"]

        # Sort expiry dates chronologically
        sorted_expiries = sorted(
            raw_expiries,
            key=lambda x: datetime.strptime(x, "%d-%b-%Y")
        )
    except Exception as e:
        print(f"Error fetching expiries for {symbol}: {e}")
        sorted_expiries = []
    return jsonify({"expiries": sorted_expiries})

@app.route("/stream_analysis", methods=["POST"])
def stream_analysis():
    symbol = request.form["symbol"]
    expiry_date = request.form["expiry_date"]

    options_data = fetch_and_cache_options_data(symbol, expiry_date)
    if not options_data:
        return Response("data: Error fetching data\n\n", mimetype="text/event-stream")

    return Response(stream_analysis_with_gemini(symbol, expiry_date, options_data), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True)