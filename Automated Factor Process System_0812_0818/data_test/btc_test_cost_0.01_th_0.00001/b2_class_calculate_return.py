# calculate return

def generate_periods(period_labels):
    return {f'return_{label}': 1 for label in period_labels}

def compute_returns(df, index_column='date_time', value_column='close', period_labels=None, threshold = None):
    df = df.set_index(index_column)
    periods = generate_periods(period_labels)
    for label, period in periods.items():
        df[label] = df[value_column].pct_change(periods=period)
        df[label] = df[label].apply(lambda x: 1 if x > threshold else (-1 if x < -threshold else 0))
    return df