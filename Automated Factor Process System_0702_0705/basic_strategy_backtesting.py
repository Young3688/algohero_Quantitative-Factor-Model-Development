import pandas as pd
import quantstats as qs
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def basic_strategy_backtesting(df, threshold, cost):

    for pre in df.columns[2:]:

        df[f'{pre}_sta_rtn'] = 0

        position = df[pre].apply(lambda x: 1 if x > threshold else (-1 if x < -threshold else 0)).shift(1)
        
        df[f'{pre}_sta_rtn'] = position * df.iloc[:, 1]
        
        df[f'{pre}_cum_rtn'] = df[f'{pre}_sta_rtn'].cumsum()

    df['bas_cum_rtn'] = df.iloc[:,1].cumsum()
    
    
    # basic backtesting

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
            
            x_col_str = f'{x_col.replace("_cum_rtn", "_sta_rtn")}'
            report_filename = f'{x_col_str}_report.html'
            qs.reports.html(df[x_col_str], df.iloc[:,1], output=report_filename)
        
    return df
