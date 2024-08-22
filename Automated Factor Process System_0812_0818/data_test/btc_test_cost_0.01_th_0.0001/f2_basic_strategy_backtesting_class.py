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
            shifted_position = position.shift(1).fillna(0)
            transaction_costs = abs((position - shifted_position) * cost)
            df[f'{pre}_sta_rtn'] = position * return_base - transaction_costs
            df[f'{pre}_cum_rtn'] = df[f'{pre}_sta_rtn'].cumsum()    
            
    df[f'bas_cum_rtn_return_base'] = return_base.cumsum()  
    
    # basic backtesting

    sharpe_ratio_return = qs.stats.sharpe(return_base, periods=365*24)
    max_drawdown_return = qs.stats.max_drawdown(return_base)
    final_cumulative_return = df['bas_cum_rtn_return_base'].iloc[-1]

    result_df = pd.DataFrame({
        'Strategy': ['Benchmark'],
        'Sharpe Ratio': [sharpe_ratio_return],
        'Max Drawdown': [max_drawdown_return],
        'Final Cumulative Return': [final_cumulative_return]
    })
    
    for x_col in df.columns:
        if '_cum_rtn' in x_col and 'bas_cum_rtn' not in x_col:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['bas_cum_rtn_return_base'], name='Cumulative Basic Return', line_color='navy'))
            fig.add_trace(go.Scatter(x=df.index, y=df[x_col], name='Cumulative Strategy Return', line_color='teal'))
            fig.update_layout(title=f'Cumulative Returns for {x_col}', height=400, width=900, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), margin=dict(l=20, r=20, t=20, b=20))
            fig.show()

            x_col_str = f'{x_col.replace("_cum_rtn", "_sta_rtn")}'
            sharpe_ratio_strategy = qs.stats.sharpe(df[x_col_str], periods=365*24)
            max_drawdown_strategy = qs.stats.max_drawdown(df[x_col_str])
            final_cumulative_strategy = df[x_col].iloc[-1]
            
            new_row = pd.DataFrame({
                'Strategy': [x_col_str],
                'Sharpe Ratio': [sharpe_ratio_strategy],
                'Max Drawdown': [max_drawdown_strategy],
                'Final Cumulative Return': [final_cumulative_strategy]
            })

            result_df = pd.concat([result_df, new_row], ignore_index=True)

    return df, result_df
