import pandas as pd
import requests
from io import StringIO

print('Fetching Amazon (AMZN) historical data...')
print('Using Alpha Vantage API (last 100 days)')

API_KEY = 'cccx'
ticker = 'AMZN'

# Use compact size for free tier (last 100 trading days)
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=compact&datatype=csv&apikey={API_KEY}'

print(f'\nFetching data from Alpha Vantage...')

try:
    response = requests.get(url, timeout=30)
    
    # Parse CSV
    df = pd.read_csv(StringIO(response.text))
    
    if len(df) == 0:
        raise Exception('No data returned')
    
    print(f'✓ Fetched {len(df)} rows from API')
    
    # Clean column names
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    
    # Rename timestamp to date
    if 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})
    
    # Convert date and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f'✓ Downloaded {len(df)} rows of Amazon stock data')
    print(f'✓ Date range: {df["date"].min()} to {df["date"].max()}')
    print(f'✓ Price range: ${df["close"].min():.2f} to ${df["close"].max():.2f}')
    print(f'✓ Columns: {list(df.columns)}')
    
    # Save to CSV
    output_path = 'C:/Users/rraga/Projects/qusa/data/raw/AMZN_2025-08-19_2026-01-09.csv'
    df.to_csv(output_path, index=False)
    
    print(f'✓ Saved to: {output_path}')
    print(f'\nFirst 5 rows:')
    print(df.head())
    print(f'\nLast 5 rows:')
    print(df.tail())
    
except Exception as e:
    print(f'\n✗ Error: {e}')
    import traceback
    traceback.print_exc()
