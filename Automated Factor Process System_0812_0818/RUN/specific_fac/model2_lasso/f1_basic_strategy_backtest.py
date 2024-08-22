import pandas as pd
import quantstats as qs
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# def basic_strategy_backtesting(df, threshold, cost):

#     return_columns = [col for col in df.columns if col.startswith('return_')]
#     last_return_col_index = max(df.columns.get_loc(col) for col in return_columns)
#     prediction_start_index = last_return_col_index + 1

#     for pre in df.columns[prediction_start_index:]: 

#         df[f'{pre}_sta_rtn'] = 0

#         position = df[pre].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).shift(1)
        
#         df[f'{pre}_sta_rtn'] = position * df.iloc[:, 1]
        
#         df[f'{pre}_cum_rtn'] = df[f'{pre}_sta_rtn'].cumsum()

#     df['bas_cum_rtn'] = df.iloc[:,1].cumsum()
    
def basic_strategy_backtesting(df, threshold, cost):
    return_columns = [col for col in df.columns if col.startswith('return_')]
    last_return_col_index = max(df.columns.get_loc(col) for col in return_columns)
    prediction_start_index = last_return_col_index + 1


    for pre in df.columns[prediction_start_index:]:
        df[f'{pre}_sta_rtn'] = 0

        position = df[pre].apply(lambda x: 1 if x > threshold else (-1 if x < -threshold else 0)).shift(1)

        shifted_position = position.shift(1).fillna(0)
        
        transaction_costs = abs((position - shifted_position) * cost)
        
        df[f'{pre}_sta_rtn'] = position * df.iloc[:, 1] - transaction_costs
        df[f'{pre}_cum_rtn'] = df[f'{pre}_sta_rtn'].cumsum()

    df['bas_cum_rtn'] = df.iloc[:, 1].cumsum()
        
    # basic backtesting
    sharpe_ratio_return = qs.stats.sharpe(df.iloc[:, 1], periods=365*24)
    max_drawdown_return = qs.stats.max_drawdown(df.iloc[:, 1])
    final_cumulative_return = df['bas_cum_rtn'].iloc[-1]

    result_df = pd.DataFrame({
        'Strategy': ['Benchmark'],
        'Sharpe Ratio': [sharpe_ratio_return],
        'Max Drawdown': [max_drawdown_return],
        'Final Cumulative Return': [final_cumulative_return]
    })
    
    for x_col in df.columns:        
        if '_cum_rtn' in x_col:
            if df[x_col].abs().sum() == 0:
                print(f"Skipping {x_col} as it only contains zero values.")
                continue
            if df[x_col].equals(df['bas_cum_rtn']):
                print(f"Skipping.")
                continue
            print(f'{x_col}:')
            fig = go.Figure()
            fig.add_trace(go.Line(x=df.index, y=df['bas_cum_rtn'], name = 'Cumulative_Outcome_Basic', line_color = 'navy'))
            fig.add_trace(go.Line(x= df.index, y=df[x_col], 
                                name = 'Cumulative_Outcome_strategy', line_color = 'teal'))
            fig.update_layout(height=400, width=900, legend=dict(yanchor="top", y=0.99, xanchor="left",x=0.01), margin=dict(l=20, r=20, t=20, b=20))
            fig.show()
            
            # report_filename = f'{x_col_str}_report.html'
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

            # qs.reports.html(df[x_col_str], df.iloc[:,1], output=report_filename, periods=365*24)
        
    return df, result_df