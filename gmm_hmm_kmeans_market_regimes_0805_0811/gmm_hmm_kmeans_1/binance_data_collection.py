import csv
import pandas as pd
from binance.um_futures import UMFutures
from datetime import datetime

def binance_data_collection(symbol, interval, delay_minutes, start_year, start_month, start_day, end_year, end_month, end_day):
    client = UMFutures()
    start_time = int(datetime(start_year, start_month, start_day, 0, 0, 0).timestamp() * 1000)
    end_time = int(datetime(end_year, end_month, end_day, 23, 59, 59).timestamp() * 1000)


    klines = []
    klines_data = []
    while True:
        klines_response = client.klines(symbol=symbol, interval=interval, startTime=start_time, endTime=end_time, limit=1500)
        if not klines_response:
            break

        for k in klines_response:
            utc_time = datetime.utcfromtimestamp(k[0] // 1000)
            date_time = utc_time + pd.Timedelta(minutes=delay_minutes)
            # offset = datetime.timedelta(hours=8)
            # beijing_time = utc_time + offset
            if datetime(start_year, start_month, start_day) <= utc_time < datetime(end_year, end_month, end_day):
                item = {
                    'date_time': date_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'real_time': utc_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'timestamp': k[0],
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                    # 'close_time': float(k[6]),
                    'quote_asset_volume': float(k[7]),
                    'number_of_trades': float(k[8]),
                    'taker_buy_base_asset_volume': float(k[9]),
                    'taker_buy_quote_asset_volume': float(k[10])
                }
                klines_data.append(item)

        start_time = klines_response[-1][0] + 1
        if start_time >= end_time:
            break
        
        
    csv_file = 'BTCUSDT_{0}_{1}_{2}_to_{3}_{4}_{5}_with_{6}.csv'.format(start_year, start_month, start_day, end_year, end_month, end_day, interval)
    csv_columns = ['date_time', 'real_time', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for row in klines_data:
            writer.writerow(row)

    print(f'Data has been saved to {csv_file}')
    
    return csv_file
