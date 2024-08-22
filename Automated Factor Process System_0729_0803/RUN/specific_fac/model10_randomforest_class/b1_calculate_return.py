# calculate return
# def generate_periods(period_labels):
#     time_units = {'h': 1, 'd': 24}
#     periods = {}
#     for label in period_labels:
#         if label[-1] not in time_units:
#             raise ValueError(f"Invalid time unit in '{label}'. Allowed units are 'h' for hours and 'd' for days.")        
#         number = int(label[:-1])
#         unit = label[-1]        
#         total_hours = number * time_units[unit]        
#         periods[f'return_{label}'] = total_hours
#     return periods

# def compute_returns(df, index_column='date_time', value_column='close', period_labels=None):
#     df.set_index(index_column, inplace=True)
#     periods = generate_periods(period_labels)
#     for label, period in periods.items():
#         df[label] = df[value_column].pct_change(periods=period)
#     return df


def generate_periods(period_labels):
    return {f'return_{label}': 1 for label in period_labels}

def compute_returns(df, index_column='date_time', value_column='close', period_labels=None):
    df.set_index(index_column, inplace=True)
    periods = generate_periods(period_labels)
    for label in periods:
        df[label] = df[value_column].pct_change(periods=periods[label])
    return df