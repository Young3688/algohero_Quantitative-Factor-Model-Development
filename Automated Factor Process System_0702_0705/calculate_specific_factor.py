def generate_hours(period_labels):
    time_units = {'h': 1, 'd': 24}
    periods = {}
    for label in period_labels:
        number = int(label[:-1])
        unit = label[-1]
        total_hours = number * time_units[unit]
        periods[label] = total_hours
    return periods

def calculate_specific_factor(df, cols, period_labels=None):
    
    periods = generate_hours(period_labels)
    
    for col in cols:
        for label, hours in periods.items():
            if label == '1h':
                window = '2h'
            else:
                window = label
                
            df[f'{col}_sma_{label}'] = df[col].rolling(window).mean()
            df[f'{col}_ema_{label}'] = df[col].ewm(span=hours, adjust=False).mean()
            df[f'{col}_diff_sma_{label}'] = df[col] - df[f'{col}_sma_{label}']
            df[f'{col}_volatility_{label}'] = df[col].rolling(window=window).std()
            df[f'{col}_quote_change_{label}'] = (df[col] - df[col].shift(hours)) / df[col].shift(hours) * 100
            df[f'{col}_tvmi_{label}'] = (df[col] - df[f'{col}_sma_{label}']) / df[f'{col}_sma_{label}'] * 100

            rolling_max = df[col].rolling(window).max()
            rolling_min = df[col].rolling(window).min()
            df[f'{col}_tvmo_{label}'] = (df[col] - df[f'{col}_sma_{label}']) / (rolling_max - rolling_min) * 100
            
        df[f'{col}_trade_close_ratio'] = df[col] / df['close']
        df[f'{col}_trades_volume_ratio'] = df[col] / df['volume']

    return df