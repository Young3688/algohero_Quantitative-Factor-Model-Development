import pandas as pd
import quantstats as qs
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def basic_strategy_backtesting(df, cost):
    return_base = df['close'].pct_change()

    
    return_columns = [col for col in df.columns if col.startswith('return_')]
    last_return_col_index = max(df.columns.get_loc(col) for col in return_columns)
    prediction_start_index = last_return_col_index + 1

    for pre in df.columns[prediction_start_index:]: 
        parts = pre.split('___')
        if len(parts) > 1 and '_pred' in parts[1]:
            return_type = parts[1].split('_pred')[0]
        if return_type in return_columns:
            position = df[pre].shift(1)
            df[f'{pre}_sta_rtn'] = position * return_base
            df[f'{pre}_cum_rtn'] = df[f'{pre}_sta_rtn'].cumsum()    
            
    df[f'bas_cum_rtn_return_base'] = return_base.cumsum()  
    # basic backtesting

    for x_col in df.columns:
        if '_cum_rtn' in x_col and 'bas_cum_rtn' not in x_col:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['bas_cum_rtn_return_base'], name='Cumulative Basic Return', line_color='navy'))
            fig.add_trace(go.Scatter(x=df.index, y=df[x_col], name='Cumulative Strategy Return', line_color='teal'))
            fig.update_layout(title=f'Cumulative Returns for {x_col}', height=400, width=900, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), margin=dict(l=20, r=20, t=20, b=20))
            fig.show()

            x_col_str = f'{x_col.replace("_cum_rtn", "_sta_rtn")}'
            report_filename = f'{x_col_str}_report.html'
            qs.reports.html(df[x_col_str], return_base, output=report_filename)

    return df
