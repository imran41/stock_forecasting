import yfinance as yf
import pandas as pd

# Function: Data load karne ke liye
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)       # Yahoo Finance se data download
    if data.empty:                                         # Agar koi data nahi mila
        print(f"No data found for ticker: {ticker}")
        return None
    if isinstance(data.columns, pd.MultiIndex):            # Agar data multi-level columns me aaya
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)                         # Date ko ek column me convert karo
    return data

# Main code
if __name__ == "__main__":
    ticker = "TCS.NS"                                # Example: Gold ETF
    start_date = "2025-10-01"
    end_date = "2025-10-25"

    df = load_data(ticker, start_date, end_date)           # Data fetch karna

    if df is not None:
        file_name = f"{ticker}_data.csv"
        df.to_csv(file_name, index=False)                  # CSV file me save karna
        print(f"✅ Data saved successfully: {file_name}")
    else:
        print("❌ Data not available.")
