import numpy as np
import pandas as pd
from numpy import abs
from numpy import log
from numpy import sign
from scipy.stats import rankdata

# region Auxiliary functions
def ts_sum(x, window, freq = 'h'):
    """
    Wrapper function to estimate rolling sum for cryptocurrency data.
    :param df: a pandas DataFrame.
    :param window: the rolling window size in terms of days.
    :param freq: frequency of the data ('h' for hour, 'd' for day, etc.)
    :return: a pandas DataFrame with the time-series sum over the past 'window' days.
    """
    # 计算基于频率的窗口大小
    if freq == 'd':
        roll_window = window / 24
    elif freq == 'h':
        roll_window = window
    else:
        raise ValueError("Unsupported frequency")
    return x.rolling(roll_window).sum()

def sma(x, window, freq = 'h'):
    """
    Wrapper function to estimate SMA.
    :param x: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    # 计算基于频率的窗口大小
    if freq == 'd':
        roll_window = window / 24
    elif freq == 'h':
        roll_window = window
    else:
        raise ValueError("Unsupported frequency")
    
    return x.rolling(roll_window).mean()

def stddev(x, window, freq = 'h'):
    """
    Wrapper function to estimate rolling standard deviation.
    :param x: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    # 计算基于频率的窗口大小
    if freq == 'd':
        roll_window = window / 24
    elif freq == 'h':
        roll_window = window
    else:
        raise ValueError("Unsupported frequency")
    
    return x.rolling(roll_window).std()

def correlation(x, y, window, freq = 'h'):
    """
    Wrapper function to estimate rolling corelations.
    :param x: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    # 计算基于频率的窗口大小
    if freq == 'd':
        roll_window = window / 24
    elif freq == 'h':
        roll_window = window
    else:
        raise ValueError("Unsupported frequency")
    
    return x.rolling(roll_window).corr(y)


def covariance(x, y, window, freq = 'h'):
    """
    Wrapper function to estimate rolling covariance.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    # 计算基于频率的窗口大小
    if freq == 'd':
        roll_window = window / 24
    elif freq == 'h':
        roll_window = window
    else:
        raise ValueError("Unsupported frequency")
    
    return x.rolling(roll_window).cov(y)

def rolling_rank(na):
    """
    Helper function to calculate the rank of the last value in a rolling window.
    :param x: a numpy array of values in the rolling window.
    :return: the rank of the last value as a percentage of the total number of values.
    """
    rank = rankdata(na)
    last_value_rank = rank[-1]
    return last_value_rank / len(na)


def ts_rank(x, window, freq = 'h'):
    """
    Wrapper function to estimate rolling rank.
    :param x: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """
    # 计算基于频率的窗口大小
    if freq == 'd':
        roll_window = window / 24
    elif freq == 'h':
        roll_window = window
    else:
        raise ValueError("Unsupported frequency")
    
    return x.rolling(roll_window).apply(rolling_rank, raw=True)

def rolling_prod(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return np.prod(na)

def product(x, window, freq = 'h'):
    """
    Wrapper function to estimate rolling product.
    :param x: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series product over the past 'window' days.
    """
    # 计算基于频率的窗口大小
    if freq == 'd':
        roll_window = window / 24
    elif freq == 'h':
        roll_window = window
    else:
        raise ValueError("Unsupported frequency")
    
    return x.rolling(roll_window).apply(rolling_prod)

def ts_min(x, window, freq = 'h'):
    """
    Wrapper function to estimate rolling min.
    :param x: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    # 计算基于频率的窗口大小
    if freq == 'd':
        roll_window = window / 24
    elif freq == 'h':
        roll_window = window
    else:
        raise ValueError("Unsupported frequency")
    
    return x.rolling(roll_window).min()

def ts_max(x, window, freq = 'h'):
    """
    Wrapper function to estimate rolling min.
    :param x: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    if freq == 'd':
        roll_window = window / 24
    elif freq == 'h':
        roll_window = window
    else:
        raise ValueError("Unsupported frequency")
    
    return x.rolling(roll_window).max()

def delta(x, period):
    """
    Wrapper function to estimate difference.
    :param x: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return x.diff(period)

def delay(x, period):
    """
    Wrapper function to estimate lag.
    :param x: a pandas DataFrame.
    :param period: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return x.shift(period)

# def rank(x):
#     """
#     Cross sectional rank
#     :param x: a pandas DataFrame.
#     :return: a pandas DataFrame with rank along columns.
#     """
#     #return x.rank(axis=1, pct=True)
#     return x.rank(pct=True)

def scale(x, k=1):
    """
    Scaling time serie.
    :param x: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """
    return x.mul(k).div(np.abs(x).sum())

def ts_argmax(x, window, freq = 'h'):
    """
    Wrapper function to estimate which day ts_max(x, window) occurred on
    :param x: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    if freq == 'd':
        roll_window = window / 24
    elif freq == 'h':
        roll_window = window
    else:
        raise ValueError("Unsupported frequency")
    
    return x.rolling(roll_window).apply(np.argmax) + 1 

def ts_argmin(x, window, freq = 'h'):
    """
    Wrapper function to estimate which day ts_min(x, window) occurred on
    :param x: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    if freq == 'd':
        roll_window = window / 24
    elif freq == 'h':
        roll_window = window
    else:
        raise ValueError("Unsupported frequency")
    
    return x.rolling(roll_window).apply(np.argmin) + 1

def signedpower(x, a):
    return np.power(x,a)
    
def decay_linear(x, window, freq='h'):
    """
    Linear weighted moving average implementation.
    :param x: a pandas Series or DataFrame.
    :param window: the LWMA window size
    :param freq: the frequency of the data (default is 'h' for hourly)
    :return: a pandas Series or DataFrame with the LWMA.
    """
    if freq == 'd':
        roll_window = window / 24
    elif freq == 'h':
        roll_window = window
    else:
        raise ValueError("Unsupported frequency")

    is_series = False
    if isinstance(x, pd.Series):
        x = x.to_frame()
        is_series = True
    lwma = pd.DataFrame(index=x.index, columns=x.columns)
    weights = np.arange(1, roll_window + 1)
    weights = weights / weights.sum()

    for column in x.columns:
        series = x[column].values
        lwma_series = np.full(len(series), np.nan)
        for i in range(int(roll_window) - 1, len(series)):
            lwma_series[i] = np.dot(series[i - int(roll_window) + 1:i + 1], weights)
        lwma[column] = lwma_series
    if is_series:
        lwma = lwma.squeeze()
    return lwma
# endregion



def get_alpha(df):
    volume_columns = [
        'volume', 
        'quote_asset_volume', 
        'number_of_trades', 
        'taker_buy_base_asset_volume', 
        'taker_buy_quote_asset_volume'
    ]
    for volume_col in volume_columns:
        if volume_col in df.columns:
            stock = Alphas(df, volume_col)
            df[f'alpha001_{volume_col}'] = stock.alpha001()
            df[f'alpha002_{volume_col}']=stock.alpha002()
            df[f'alpha003_1_{volume_col}']=stock.alpha003_1()
            df[f'alpha003_2_{volume_col}']=stock.alpha003_2()
            df[f'alpha004_{volume_col}']=stock.alpha004()
            df[f'alpha005_{volume_col}']=stock.alpha005()
            df[f'alpha006_{volume_col}']=stock.alpha006()
            df[f'alpha007_{volume_col}']=stock.alpha007()
            df[f'alpha008_{volume_col}']=stock.alpha008()
            df[f'alpha009_{volume_col}']=stock.alpha009()
            df[f'alpha010_{volume_col}']=stock.alpha010()
            df[f'alpha011_{volume_col}']=stock.alpha011()
            df[f'alpha012_{volume_col}']=stock.alpha012()
            df[f'alpha013_{volume_col}']=stock.alpha013()
            df[f'alpha014_{volume_col}']=stock.alpha014()
            df[f'alpha015_{volume_col}']=stock.alpha015()
            df[f'alpha016_{volume_col}']=stock.alpha016()
            df[f'alpha017_{volume_col}']=stock.alpha017()
            df[f'alpha018_{volume_col}']=stock.alpha018()
            df[f'alpha019_{volume_col}']=stock.alpha019()
            df[f'alpha020_{volume_col}']=stock.alpha020()
            df[f'alpha021_{volume_col}']=stock.alpha021()
            df[f'alpha022_{volume_col}']=stock.alpha022()
            df[f'alpha023_{volume_col}']=stock.alpha023()
            df[f'alpha024_{volume_col}']=stock.alpha024()
            df[f'alpha025_{volume_col}']=stock.alpha025()
            df[f'alpha026_{volume_col}']=stock.alpha026()
            df[f'alpha027_{volume_col}']=stock.alpha027()
            df[f'alpha028_1_{volume_col}']=stock.alpha028_1()
            df[f'alpha028_2_{volume_col}']=stock.alpha028_2()
            df[f'alpha029_{volume_col}']=stock.alpha029()
            df[f'alpha030_{volume_col}']=stock.alpha030()
            df[f'alpha031_{volume_col}']=stock.alpha031()
            df[f'alpha032_{volume_col}']=stock.alpha032()
            df[f'alpha033_{volume_col}']=stock.alpha033()
            df[f'alpha034_{volume_col}']=stock.alpha034()
            df[f'alpha035_1_{volume_col}']=stock.alpha035_1()
            df[f'alpha035_2_{volume_col}']=stock.alpha035_2()
            df[f'alpha036_{volume_col}']=stock.alpha036()
            df[f'alpha037_{volume_col}']=stock.alpha037()
            df[f'alpha038_{volume_col}']=stock.alpha038()
            df[f'alpha039_{volume_col}']=stock.alpha039()
            df[f'alpha040_{volume_col}']=stock.alpha040()
            df[f'alpha041_{volume_col}']=stock.alpha041()
            df[f'alpha042_1_{volume_col}']=stock.alpha042_1()
            df[f'alpha042_2_{volume_col}']=stock.alpha042_2()
            df[f'alpha043_{volume_col}']=stock.alpha043()
            df[f'alpha044_{volume_col}']=stock.alpha044()
            df[f'alpha045_{volume_col}']=stock.alpha045()
            df[f'alpha046_{volume_col}']=stock.alpha046()
            df[f'alpha047_{volume_col}']=stock.alpha047()
            df[f'alpha048_{volume_col}']=stock.alpha048()
            df[f'alpha049_{volume_col}']=stock.alpha049()
            df[f'alpha050_{volume_col}']=stock.alpha050()
            df[f'alpha051_{volume_col}']=stock.alpha051()
            df[f'alpha052_{volume_col}']=stock.alpha052()
            df[f'alpha053_{volume_col}']=stock.alpha053()
            df[f'alpha054_1_{volume_col}']=stock.alpha054_1()
            df[f'alpha054_2_{volume_col}']=stock.alpha054_2()
            df[f'alpha055_{volume_col}']=stock.alpha055()
            df[f'alpha056_{volume_col}']=stock.alpha056()
            df[f'alpha057_{volume_col}']=stock.alpha057()
            df[f'alpha058_{volume_col}']=stock.alpha058()
            df[f'alpha059_{volume_col}']=stock.alpha059()
            df[f'alpha060_{volume_col}']=stock.alpha060()
            df[f'alpha061_{volume_col}']=stock.alpha061()
            df[f'alpha062_{volume_col}']=stock.alpha062()
            df[f'alpha063_{volume_col}']=stock.alpha063()
            df[f'alpha064_{volume_col}']=stock.alpha064()
            df[f'alpha065_{volume_col}']=stock.alpha065()
            df[f'alpha066_{volume_col}']=stock.alpha066()
            df[f'alpha067_{volume_col}']=stock.alpha067()
            df[f'alpha068_{volume_col}']=stock.alpha068()
            df[f'alpha069_{volume_col}']=stock.alpha069()
            df[f'alpha070_{volume_col}']=stock.alpha070()
            df[f'alpha071_{volume_col}']=stock.alpha071()
            # df[f'alpha072_{volume_col}']=stock.alpha072()
            df[f'alpha073_{volume_col}']=stock.alpha073()
            df[f'alpha074_{volume_col}']=stock.alpha074()
            df[f'alpha075_{volume_col}']=stock.alpha075()
            df[f'alpha076_{volume_col}']=stock.alpha076()
            df[f'alpha077_{volume_col}']=stock.alpha077()
            df[f'alpha078_{volume_col}']=stock.alpha078()
            df[f'alpha079_{volume_col}']=stock.alpha079()
            df[f'alpha080_{volume_col}']=stock.alpha080()
            df[f'alpha081_{volume_col}']=stock.alpha081()
            df[f'alpha082_{volume_col}']=stock.alpha082()
            df[f'alpha083_{volume_col}']=stock.alpha083()
            df[f'alpha084_{volume_col}']=stock.alpha084()
            df[f'alpha085_{volume_col}']=stock.alpha085()
            df[f'alpha086_{volume_col}']=stock.alpha086()
            df[f'alpha087_{volume_col}']=stock.alpha087()
            df[f'alpha088_{volume_col}']=stock.alpha088()
            df[f'alpha089_{volume_col}']=stock.alpha089()
            df[f'alpha090_{volume_col}']=stock.alpha090()
            df[f'alpha091_{volume_col}']=stock.alpha091()
            df[f'alpha092_{volume_col}']=stock.alpha092()
            df[f'alpha093_{volume_col}']=stock.alpha093()
            df[f'alpha094_{volume_col}']=stock.alpha094()
            df[f'alpha095_{volume_col}']=stock.alpha095()
            df[f'alpha096_{volume_col}']=stock.alpha096()
            # df[f'alpha097_{volume_col}']=stock.alpha097()
            df[f'alpha098_{volume_col}']=stock.alpha098()
            df[f'alpha099_{volume_col}']=stock.alpha099()
            df[f'alpha100_{volume_col}']=stock.alpha100()
            df[f'alpha101_{volume_col}']=stock.alpha101()  
    return df

class Alphas(object):
    
    def __init__(self, df_data, volume_col):
        self.open = df_data['open'] 
        self.high = df_data['high'] 
        self.low = df_data['low']   
        self.close = df_data['close']
        self.volume = df_data[volume_col] * 100
        # volume_columns = ['volume', 'quote_asset_volume', 'number_of_trades', 
        #           'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        # for volume_col in volume_columns:
        #     if volume_col in df_data.columns:
        #         self.volume = df_data[volume_col]*100 
        #     else:
        #         raise ValueError(f"Column {volume_col} does not exist in DataFrame")

        return_col = next((col for col in df_data.columns if col.startswith('return_')), None)
        self.returns = df_data[return_col] 
        self.vwap = df_data['vwap']
        

    # Alpha#1:
    # (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
    # (Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 3) : close), 2.), 3)) - 0.5
    def alpha001(self):
        condition = self.returns < 0
        alpha = self.close.copy()
        alpha[condition] = stddev(self.returns, window=3)
        alpha = signedpower(alpha, 2)
        alpha = ts_argmax(alpha, window=3) - 0.5
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#2:
    # (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    # (-1 * correlation(delta(log(volume), 2), ((close - open) / open), 3))
    def alpha002(self):
        delta_log_volume = delta(np.log(self.volume), 2)
        close_minus_open = self.close - self.open
        open_ratio = close_minus_open / self.open
        alpha = -1 * correlation(delta_log_volume, open_ratio, window=3)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#3:
    # (-1 * correlation(rank(open), rank(volume), 10))
    # (-1 * correlation(open, volume, 3))
    # (-1 * correlation(sma(open), delta(volume, 1) / volume, 3))

    def alpha003_1(self):
        alpha = -1 * correlation(self.open, self.volume, window=3)
        return alpha

    def alpha003_2(self):
        open_sma = sma(self.open, window=2)
        volume_delta = delta(self.volume, 1)
        volume_ratio = volume_delta / self.volume
        alpha = -1 * correlation(open_sma, volume_ratio, window=3)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#4:
    # (-1 * Ts_Rank(rank(low), 9))
    # (-1 * Ts_Rank((low / close.shift(3) - 1), 3))
    def alpha004(self):
        low_close_ratio = self.low / self.close.shift(3) - 1
        alpha = -1 * ts_rank(low_close_ratio, window=3)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha
    
    # Alpha#5:
    # (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    # ((open - (sum(vwap, 3) / 3)) * (-1 * (close - vwap)))
    def alpha005(self):
        vwap_sum = ts_sum(self.vwap, window=3)
        vwap_mean = vwap_sum / 3
        open_vwap_diff = self.open - vwap_mean
        close_vwap_diff = self.close - self.vwap
        alpha = open_vwap_diff * (-1 * close_vwap_diff)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#6:
    # (-1 * correlation(open, volume, 10))
    # (-1 * correlation(open, volume, 3))
    def alpha006(self):
        alpha = -1 * correlation(self.open, self.volume, window=3)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha
    
    # Alpha#7:
    # ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1* 1))
    # ((sma(close, 3) < volume) ? ((-1 * ts_rank(abs(delta(close, 3)), 3)) * sign(delta(close, 3))) : (-1* 1))
    def alpha007(self):
        close_volume_sma = sma(self.close, window=3)
        close_delta = delta(self.close, 3)
        close_delta_abs = abs(close_delta)
        close_delta_sign = np.sign(close_delta)
        
        condition = close_volume_sma < self.volume
        alpha = pd.Series(index=self.close.index)
        alpha[condition] = -1 * ts_rank(close_delta_abs, window=3) * close_delta_sign
        alpha[~condition] = -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha
    
    # Alpha#8:
    # (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))
    # (-1 * ((sum(open, 3) * sum(returns, 3)) - delay((sum(open, 3) * sum(returns, 3)), 3)))

    def alpha008(self):
        open_sum = ts_sum(self.open, window=3)
        returns_sum = ts_sum(self.returns, window=3)
        open_returns_product = open_sum * returns_sum
        open_returns_product_delay = delay(open_returns_product, 3)
        alpha = -1 * (open_returns_product - open_returns_product_delay)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#9:
    # ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
    # ((0 < ts_min(delta(close, 1), 3)) ? delta(close, 1) : ((ts_max(delta(close, 1), 3) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
    def alpha009(self):
        close_delta = delta(self.close, 1)
        close_delta_min = ts_min(close_delta, window=3)
        close_delta_max = ts_max(close_delta, window=3)
        
        alpha = pd.Series(index=self.close.index)
        condition1 = close_delta_min > 0
        condition2 = close_delta_max < 0
        alpha[condition1] = close_delta[condition1]
        alpha[~condition1 & condition2] = close_delta[~condition1 & condition2]
        alpha[~condition1 & ~condition2] = -1 * close_delta[~condition1 & ~condition2]
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#10:   
    # ((0 < ts_min(delta(close, 1), 3)) ? delta(close, 1) : (-1 * delta(close, 1)))

    def alpha010(self):
        close_delta = delta(self.close, 1)
        close_delta_min = ts_min(close_delta, window=3)
        
        alpha = pd.Series(index=self.close.index)
        condition = close_delta_min > 0
        alpha[condition] = close_delta[condition]
        alpha[~condition] = -1 * close_delta[~condition]
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#11:
    # ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
    # ((ts_max((vwap - close), 3) + ts_min((vwap - close), 3)) * delta(volume, 3))

    def alpha011(self):
        vwap_close_diff = self.vwap - self.close
        vwap_close_diff_max = ts_max(vwap_close_diff, window=3)
        vwap_close_diff_min = ts_min(vwap_close_diff, window=3)
        volume_delta = delta(self.volume, 3)
        alpha = (vwap_close_diff_max + vwap_close_diff_min) * volume_delta
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#12:
    # (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    # (sign(delta(sma(volume, 3), 3)) * (-1 * delta(sma(close, 3), 3)))

    def alpha012(self):
        volume_sma = sma(self.volume, window=3)
        volume_sma_delta = delta(volume_sma, 3)
        close_sma = sma(self.close, window=3)
        close_sma_delta = delta(close_sma, 3)
        alpha = np.sign(volume_sma_delta) * (-1 * close_sma_delta)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#13:
    # (-1 * rank(covariance(rank(close), rank(volume), 5)))
    # (-1 * covariance(close, volume, 3))

    def alpha013(self):
        alpha = -1 * covariance(self.close, self.volume, window=3)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#14:
    # ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
    # ((-1 * delta(returns, 2)) * correlation(sma(open, 3), sma(volume, 3), 3))
    def alpha014(self):
        returns_delta = delta(self.returns, 2)
        open_sma = sma(self.open, window=3)
        volume_sma = sma(self.volume, window=3)
        corr = correlation(open_sma, volume_sma, window=3)
        alpha = (-1 * returns_delta) * corr
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#15:
    # (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
    # (-1 * sum(correlation(sma(high, 2), sma(volume, 2), 2), 2))

    def alpha015(self):
        high_sma = sma(self.high, window=2)
        volume_sma = sma(self.volume, window=2)
        corr = correlation(high_sma, volume_sma, window=2)
        alpha = -1 * ts_sum(corr, window=2)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#16:
    # (-1 * rank(covariance(rank(high), rank(volume), 5)))
    # (-1 * covariance(sma(high, 3), sma(volume, 3), 3))

    def alpha016(self):
        high_sma = sma(self.high, window=3)
        volume_sma = sma(self.volume, window=3)
        alpha = -1 * covariance(high_sma, volume_sma, window=3)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#17:
    # (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))
    # (((-1 * ts_rank(close, 3)) * delta(delta(close, 1), 1)) * ts_rank((volume / sma(self.close,3), 3))
    def alpha017(self):
        close_rank = ts_rank(self.close, window=3)
        close_delta = delta(self.close, 1)
        close_delta_delta = delta(close_delta, 1)
        volume_close_sma = sma(self.close, window=3)
        volume_close_sma_rank = ts_rank(self.volume / volume_close_sma, window=3)
        alpha = (-1 * close_rank * close_delta_delta) * volume_close_sma_rank
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#18:
    # (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))
    # (-1 * ((stddev(abs((close - open)), 3) + (close - open)) + correlation(close, open, 3)))

    def alpha018(self):
        close_open_diff_abs = abs(self.close - self.open)
        close_open_diff_abs_std = stddev(close_open_diff_abs, window=3)
        close_open_diff = self.close - self.open
        close_open_corr = correlation(self.close, self.open, window=3)
        alpha = -1 * (close_open_diff_abs_std + close_open_diff + close_open_corr)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#19:
    # ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))
    #(-1 * (1 + (1 + sum(returns, 12))))

    def alpha019(self):
        returns_sum = ts_sum(self.returns, window=12)
        alpha = -1 * (1 + (1 + returns_sum))
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#20:
    # (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
    # ((-1 * (open - mean(high, 3))) * (open - mean(close, 3))) * (open - mean(low, 3))

    def alpha020(self):
        high_mean = sma(self.high, window=3)
        close_mean = sma(self.close, window=3)
        low_mean = sma(self.low, window=3)
        alpha = (-1 * (self.open - high_mean) * (self.open - close_mean)) * (self.open - low_mean)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#21:
    # ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1))))
    # (((sum(close, 3) / 3) + stddev(close, 3)) < (sum(close, 2) / 2)) ? (-1) : (((sum(close, 2) / 2) < ((sum(close, 3) / 3) - stddev(close, 3))) ? 1 : (((volume / sma(volume, 3)) >= 1) ? 1 : (-1)))
    def alpha021(self):
        close_sum_3 = ts_sum(self.close, window=3)
        close_sum_2 = ts_sum(self.close, window=2)
        close_std_3 = stddev(self.close, window=3)
        volume_sma_3 = sma(self.volume, window=3)
        
        alpha = pd.Series(index=self.close.index)
        condition1 = (close_sum_3 / 3 + close_std_3) < (close_sum_2 / 2)
        condition2 = (close_sum_2 / 2) < (close_sum_3 / 3 - close_std_3)
        condition3 = (self.volume / volume_sma_3) >= 1
        alpha[condition1] = -1
        alpha[~condition1 & condition2] = 1
        alpha[~condition1 & ~condition2 & condition3] = 1
        alpha[~condition1 & ~condition2 & ~condition3] = -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#22:
    # (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
    # (-1 * delta(correlation(high, volume, 3), 3) * stddev(close, 3))

    def alpha022(self):
        high_volume_corr = correlation(self.high, self.volume, window=3)
        high_volume_corr_delta = delta(high_volume_corr, 3)
        close_std = stddev(self.close, window=3)
        alpha = -1 * high_volume_corr_delta * close_std
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#23:
    # (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
    # (((sum(high, 3) / 3) < high) ? (-1 * delta(high, 1)) : delta(high, 1))


    def alpha023(self):
        high_sum_3 = ts_sum(self.high, window=3)
        high_delta = delta(self.high, 1)
        
        alpha = pd.Series(index=self.high.index)
        condition = (high_sum_3 / 3) < self.high
        alpha[condition] = -1 * high_delta[condition]
        alpha[~condition] = high_delta[~condition]
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#24:
    # ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))
    # ((((delta((sum(close, 24) / 24), 24) / delay(close, 24)) < 0.1) || ((delta((sum(close, 24) / 24), 24) / delay(close, 24)) == 0.1)) ? (-1 * (close - ts_min(close, 50))) : (-1 * delta(close, 2)))

    def alpha024(self):
        close_sum_24 = ts_sum(self.close, window=24)
        close_sum_24_delta = delta(close_sum_24 / 24, 24)
        close_delay_24 = delay(self.close, 24)
        close_min_50 = ts_min(self.close, window=50)
        close_delta_2 = delta(self.close, 2)
        
        alpha = pd.Series(index=self.close.index)
        condition = (close_sum_24_delta / close_delay_24 < 0.1) | (close_sum_24_delta / close_delay_24 == 0.1)
        alpha[condition] = -1 * (self.close[condition] - close_min_50[condition])
        alpha[~condition] = -1 * close_delta_2[~condition]
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#25:
    # rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    # (((close / delay(close, 1) - 1) * sma(volume, 24)) / (vwap * (high / close - 1)))

    def alpha025(self):
        close_delay_1 = delay(self.close, 1)
        close_return = self.close / close_delay_1 - 1
        volume_sma_24 = sma(self.volume, window=24)
        high_close_ratio = self.high / self.close - 1
        alpha = (close_return * volume_sma_24) / (self.vwap * high_close_ratio)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#26:
    # (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    # (-1 * ts_max(correlation(ts_rank(volume, 3), ts_rank(close, 3), 3), 2))

    def alpha026(self):
        volume_rank = ts_rank(self.volume, window=3)
        close_rank = ts_rank(self.close, window=3)
        corr = correlation(volume_rank, close_rank, window=3)
        alpha = -1 * ts_max(corr, window=2)
        alpha = alpha.interpolate()
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#27:
    # ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
    # ((0.5 < ((sum(correlation(volume, vwap, 3), 2) / 2.0))) ? (-1 * 1) : 1)
    def alpha027(self):
        corr = correlation(self.volume, self.vwap, window=3)
        corr_sum = ts_sum(corr, window=2)
        alpha = pd.Series(index=self.close.index)
        condition = 0.5 < (corr_sum / 2.0)
        alpha[condition] = -1
        alpha[~condition] = 1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#28:
    # scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    # (((correlation(sma(self.close,10), low, 3) + ((open + high + low + close) / 4)) - close) > 0 ? 1 : -1)
    # # scale(((correlation(sma(self.close,10), low, 3) + ((high + low) / 2)) - close))

    def alpha028_1(self):
        close_volume_sma = sma(self.close, window=10)
        corr = correlation(close_volume_sma, self.low, window=3)
        ohlc_mean = (self.open + self.high + self.low + self.close) / 4
        alpha = pd.Series(index=self.close.index)
        condition = (corr + ohlc_mean - self.close) > 0
        alpha[condition] = 1
        alpha[~condition] = -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    def alpha028_2(self):
        close_volume_sma = sma(self.close, window=10)
        corr = correlation(close_volume_sma, self.low, window=3)
        hl_mean = (self.high + self.low) / 2
        alpha = scale(corr + hl_mean - self.close)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#29:
    # (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
    # (min(product((scale(log(sum(ts_min((-1 * delta((close - 1), 3)), 2), 1)))), 1), 3) + ts_mean(delay((-1 * returns), 3), 3))

    def alpha029(self):
        close_delay_1 = delay(self.close, 1)
        close_delta = -1 * delta(close_delay_1, 3)
        close_delta_min_sum = ts_sum(ts_min(close_delta, 2), 1)
        log_scale = scale(log(close_delta_min_sum))
        prod = product(log_scale, 1)
        prod_min = ts_min(prod, 3)
        returns_delay = -1 * delay(self.returns, 3)
        returns_delay_mean = sma(returns_delay, 3)
        alpha = prod_min + returns_delay_mean
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#30:
    # (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
    # (((1.0 - ((sign((close - delay(close, 1)))) + 1) / 2)) * sum(volume, 3)) / sum(volume, 24)

    def alpha030(self):
        close_delay_1 = delay(self.close, 1)
        sign_close_diff = sign(self.close - close_delay_1)
        condition = (1.0 - ((sign_close_diff + 1) / 2))
        volume_sum_3 = ts_sum(self.volume, 3)
        volume_sum_24 = ts_sum(self.volume, 24)
        alpha = (condition * volume_sum_3) / volume_sum_24
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#31:
    # ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    # (decay_linear((-1 * delta(close, 3)), 3) + (-1 * delta(close, 2)))

    def alpha031(self):
        close_delta_3 = -1 * delta(self.close, 3)
        close_delta_3_decay = decay_linear(close_delta_3, window=3, freq='h')
        close_delta_2 = -1 * delta(self.close, 2)
        alpha = close_delta_3_decay + close_delta_2
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#32:
    # (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))
    # (scale(((sum(close, 3) / 3) - close)) + (10 * scale(correlation(vwap, delay(close, 3), 24))))
    
    def alpha032(self):
        close_sum_3 = ts_sum(self.close, window=3, freq='h')
        close_sum_3_scaled = scale(close_sum_3 / 3 - self.close)
        close_delay_3 = delay(self.close, 3)
        vwap_close_corr = correlation(self.vwap, close_delay_3, window=24, freq='h')
        vwap_close_corr_scaled = scale(vwap_close_corr, k=1)
        alpha = close_sum_3_scaled + vwap_close_corr_scaled
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#33:
    # rank((-1 * ((1 - (open / close))^1)))
    # (1 - (open / close))

    def alpha033(self):
        alpha = 1 - (self.open / self.close)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#34:
    # rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
    # ((stddev(returns, 3) / stddev(returns, 6)) - delta(close, 1))

    def alpha034(self):
        returns_stddev_3 = stddev(self.returns, window=3, freq='h')
        returns_stddev_6 = stddev(self.returns, window=6, freq='h')
        close_delta_1 = delta(self.close, 1)
        alpha = (returns_stddev_3 / returns_stddev_6) - close_delta_1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#35:
    # ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))
    # ((Ts_Rank(volume, 24) * (1 - Ts_Rank(((close + high) - low), 12))) * (1 - Ts_Rank(returns, 24)))
    # (mean(volume, 14) * (1 - mean(((close + high) - low), 7)) * (1 - mean(returns, 14)))

    def alpha035_1(self):
        volume_rank = ts_rank(self.volume, window=24, freq='h')
        high_low_diff = self.high + self.low - self.close
        high_low_diff_rank = ts_rank(high_low_diff, window=12, freq='h')
        returns_rank = ts_rank(self.returns, window=24, freq='h')
        alpha = (volume_rank * (1 - high_low_diff_rank)) * (1 - returns_rank)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    def alpha035_2(self):
        volume_mean = sma(self.volume, window=14, freq='h')
        high_low_diff = self.high + self.low - self.close
        high_low_diff_mean = sma(high_low_diff, window=7, freq='h')
        returns_mean = sma(self.returns, window=14, freq='h')
        alpha = (volume_mean * (1 - high_low_diff_mean)) * (1 - returns_mean)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha
    
    # Alpha#36:
    # (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
    # ((2.21 * correlation((close - open), delay(volume, 1), 12)) + (0.73 * delay((-1 * returns), 3)) + abs(correlation(vwap, sma(self.close, 12), 12)))

    def alpha036(self):
        close_open_diff = self.close - self.open
        volume_delay_1 = delay(self.volume, 1)
        returns_delay_3 = delay(-self.returns, 3)
        close_volume_product = self.close * self.volume
        close_volume_sma_12 = sma(close_volume_product, window=12, freq='h')
        
        term1 = 2.21 * correlation(close_open_diff, volume_delay_1, window=12, freq='h')
        term2 = 0.73 * returns_delay_3
        term3 = abs(correlation(self.vwap, close_volume_sma_12, window=12, freq='h'))
        
        alpha = term1 + term2 + term3
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#37:
    # (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
    # (correlation(delay((open - close), 1), close, 24))
    def alpha037(self):
        open_close_diff_delay_1 = delay(self.open - self.close, 1)
        alpha = correlation(open_close_diff_delay_1, self.close, window=24, freq='h')
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#38:
    # ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    # ((-1 * (Ts_Rank(close - sma(close, 4)),4)) * (close / open))
    def alpha038(self):
        close_sma_4 = sma(self.close, window=4, freq='h')
        close_sma_diff = self.close - close_sma_4
        close_sma_diff_rank = ts_rank(close_sma_diff, window=4)
        alpha = -close_sma_diff_rank * (self.close / self.open)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#39:
    # ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))
    # ((-1 * ((close - open) * (1 - (decay_linear(volume / sma(self.close, 24)))))) * (1 + sum(returns, 24)))
    def alpha039(self):
        close_open_diff = self.close - self.open
        close_volume_product = self.close * self.volume
        close_volume_sma_24 = sma(close_volume_product, window=24, freq='h')
        volume_sma_ratio = self.volume / close_volume_sma_24
        volume_sma_ratio_decay = decay_linear(volume_sma_ratio, window=24, freq='h')
        returns_sum_24 = ts_sum(self.returns, window=24, freq='h')
        alpha = -close_open_diff * (1 - volume_sma_ratio_decay) * (1 + returns_sum_24)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#40:
    # ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
    # ((-1 * stddev(high, 12)) * correlation(high, volume, 12))
    def alpha040(self):
        high_stddev_12 = stddev(self.high, window=12, freq='h')
        high_volume_corr_12 = correlation(self.high, self.volume, window=12, freq='h')
        alpha = -high_stddev_12 * high_volume_corr_12
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha
    
    # Alpha#41:
    # (((high * low)^0.5) - vwap)

    def alpha041(self):
        alpha = (self.high * self.low).pow(0.5) - self.vwap
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#42:
    # (rank((vwap - close)) / rank((vwap + close)))
    # ((vwap - close) / close)
    # ((vwap - close) / vwap)

    def alpha042_1(self):
        alpha = (self.vwap - self.close) / self.close
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    def alpha042_2(self):
        alpha = (self.vwap - self.close) / self.vwap
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#43:
    # (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    # (ts_rank((volume / sma(self.close, 12)), 12) * ts_rank((-1 * delta(close, 3)), 8))
    def alpha043(self):
        close_volume_product = self.close * self.volume
        close_volume_sma_12 = sma(close_volume_product, window=12, freq='h')
        volume_sma_ratio = self.volume / close_volume_sma_12
        volume_sma_ratio_rank_12 = ts_rank(volume_sma_ratio, window=12, freq='h')
        
        close_delta_3 = -delta(self.close, 3)
        close_delta_3_rank_8 = ts_rank(close_delta_3, window=8, freq='h')
        
        alpha = volume_sma_ratio_rank_12 * close_delta_3_rank_8
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#44:
    # (-1 * correlation(high, rank(volume), 5))
    # (-1 * correlation(high, sma(volume, 3), 3))
    def alpha044(self):
        volume_sma_3 = sma(self.volume, window=3, freq='h')
        high_volume_corr_3 = correlation(self.high, volume_sma_3, window=3, freq='h')
        alpha = -high_volume_corr_3
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#45:
    # (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))
    # (-1 * (sma(close, 24) * correlation(close, volume, 3)))
    def alpha045(self):
        close_sma_24 = sma(self.close, window=24, freq='h')
        close_volume_corr_3 = correlation(self.close, self.volume, window=3, freq='h')
        alpha = -close_sma_24 * close_volume_corr_3
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha
    
    # Alpha#46:
    # ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))
    # (((delay(close, 12) - delay(close, 3)) - (delay(close, 3) - close)) > 0.25) ? (-1) : ((((delay(close, 12) - delay(close, 3)) - (delay(close, 3) - close)) < 0) ? 1 : ((-1) * (close - delay(close, 1))))

    def alpha046(self):
        close_delay_12 = delay(self.close, 12)
        close_delay_3 = delay(self.close, 3)
        close_delay_1 = delay(self.close, 1)
        
        condition1 = (close_delay_12 - close_delay_3) - (close_delay_3 - self.close) > 0.25
        condition2 = (close_delay_12 - close_delay_3) - (close_delay_3 - self.close) < 0
        
        alpha = pd.Series(np.where(condition1, -1, np.where(condition2, 1, -1 * (self.close - close_delay_1))), index=self.close.index)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#47:
    # ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))
    # (((1 / close) * (volume / sma(self.close, 12))) * ((high - close) / sma(high, 3))) - (vwap - delay(vwap, 3))


    def alpha047(self):
        close_volume_product = self.close * self.volume
        close_volume_sma_12 = sma(close_volume_product, window=12, freq='h')
        volume_sma_ratio = self.volume / close_volume_sma_12
        close_reciprocal = 1 / self.close
        
        high_close_diff = self.high - self.close
        high_sma_3 = sma(self.high, window=3, freq='h')
        high_close_sma_ratio = high_close_diff / high_sma_3
        
        vwap_delay_3 = delay(self.vwap, 3)
        
        alpha = (close_reciprocal * volume_sma_ratio * high_close_sma_ratio) - (self.vwap - vwap_delay_3)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#48:
    # (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))
    # ((correlation(delta(close, 1), delta(delay(close, 1), 1), 24) * delta(close, 1)) / close) / sum(((delta(close, 1) / delay(close, 1))^2), 24)

    def alpha048(self):
        close_delta_1 = delta(self.close, 1)
        close_delay_1 = delay(self.close, 1)
        close_delay_1_delta_1 = delta(close_delay_1, 1)
        
        corr_24 = correlation(close_delta_1, close_delay_1_delta_1, window=24, freq='h')
        
        numerator = corr_24 * close_delta_1 / self.close
        denominator = ts_sum((close_delta_1 / close_delay_1).pow(2), window=24, freq='h')
        
        alpha = numerator / denominator
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#49:
    # (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    # (((delay(close, 12) - delay(close, 6)) / 6) - ((delay(close, 6) - close) / 6)) < -0.05 ? 1 : (-1 * (close - delay(close, 1)))

    def alpha049(self):
        close_delay_12 = delay(self.close, 12)
        close_delay_6 = delay(self.close, 6)
        close_delay_1 = delay(self.close, 1)
        
        condition = ((close_delay_12 - close_delay_6) / 6) - ((close_delay_6 - self.close) / 6) < -0.05
        
        alpha = pd.Series(np.where(condition, 1, -1 * (self.close - close_delay_1)), index=self.close.index)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    
    # Alpha#50:
    # (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    # (-1 * ts_max(correlation(volume, vwap, 3), 3))

    def alpha050(self):
        volume_vwap_corr_3 = correlation(self.volume, self.vwap, window=3, freq='h')
        alpha = -ts_max(volume_vwap_corr_3, window=3, freq='h')
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha
    
    # Alpha#51:
    # (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    # (((delay(close, 12) - delay(close, 6)) / 6) - ((delay(close, 6) - close) / 6)) < -0.05 ? 1 : (-1 * (close - delay(close, 1)))

    def alpha051(self):
        close_delay_12 = delay(self.close, 12)
        close_delay_6 = delay(self.close, 6)
        close_delay_1 = delay(self.close, 1)
        condition = ((close_delay_12 - close_delay_6) / 6) - ((close_delay_6 - self.close) / 6) < -0.05
        alpha = pd.Series(np.where(condition, 1, -1 * (self.close - close_delay_1)), index=self.close.index)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#52:
    # ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    # ((-1 * ts_min(low, 3)) + delay(ts_min(low, 3), 3)) * ((sum(returns, 120) - sum(returns, 10)) / 110) * volume

    def alpha052(self):
        low_ts_min_3 = ts_min(self.low, window=3, freq='h')
        low_ts_min_3_delay_3 = delay(low_ts_min_3, 3)
        returns_sum_120 = ts_sum(self.returns, window=120, freq='h')
        returns_sum_10 = ts_sum(self.returns, window=10, freq='h')
        alpha = (-low_ts_min_3 + low_ts_min_3_delay_3) * ((returns_sum_120 - returns_sum_10) / 110) * self.volume
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#53:
    # (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    # -1 * delta((((close - low) - (high - close)) / (close - low)), 4)
    def alpha053(self):
        numerator = (self.close - self.low) - (self.high - self.close)
        denominator = self.close - self.low
        alpha = -delta(numerator / denominator, 4)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#54:
    # ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    # -1 * ((close - low) / (high - low))
    def alpha054_1(self):
        numerator = -((self.low - self.close) * (self.open ** 5))
        denominator = (self.low - self.high) * (self.close ** 5)
        alpha = numerator / denominator
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    def alpha054_2(self):
        alpha = -((self.close - self.low) / (self.high - self.low))
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#55:
    # (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))
    # -1 * correlation(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), volume, 6)
    def alpha055(self):
        numerator = self.close - ts_min(self.low, window=12, freq='h')
        denominator = ts_max(self.high, window=12, freq='h') - ts_min(self.low, window=12, freq='h')
        alpha = -correlation(numerator / denominator, self.volume, window=6, freq='h')
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha
    
    # Alpha#56:
    # (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
    # -1 * (sum(returns, 6) / sum(returns, 3))
    def alpha056(self):
        returns_sum_6 = ts_sum(self.returns, window=6, freq='h')
        returns_sum_3 = ts_sum(self.returns, window=3, freq='h')
        alpha = -returns_sum_6 / returns_sum_3
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#57:
    # (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
    # # (0 - (1 * ((close - vwap) / decay_linear(ts_argmax(close, 24), 2))))
    def alpha057(self):
        close_vwap_diff = self.close - self.vwap
        close_argmax_24 = ts_argmax(self.close, window=24, freq='h')
        decay_lin_argmax_24 = decay_linear(close_argmax_24, window=2, freq='h')
        alpha = 0 - (close_vwap_diff / decay_lin_argmax_24)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#58:
    # (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))
    # (-1 * Ts_Rank(decay_linear(correlation(vwap, volume, 3), 7.89291), 5.50322))
    def alpha058(self):
        vwap_volume_corr_3 = correlation(self.vwap, self.volume, window=3, freq='h')
        decay_lin_corr_3 = decay_linear(vwap_volume_corr_3, window=int(7.89291), freq='h')
        alpha = -ts_rank(decay_lin_corr_3, window=int(5.50322), freq='h')
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#59:
    # (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))
    # (-1 * Ts_Rank(decay_linear(correlation(((vwap * 0.728317) + (vwap * (1 - 0.728317)), volume, 4.25197), 16.2289), 8.19648))
    def alpha059(self):
        vwap_scaled = (self.vwap * 0.728317) + (self.vwap * (1 - 0.728317))
        vwap_volume_corr_4 = correlation(vwap_scaled, self.volume, window=int(4.25197), freq='h')
        decay_lin_corr_4 = decay_linear(vwap_volume_corr_4, window=int(16.2289), freq='h')
        alpha = -ts_rank(decay_lin_corr_4, window=int(8.19648), freq='h')
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#60:
    # (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))
    # (0 - (1 * ((2 * scale(((((close - low) - (high - close)) / (high - low)) * volume))) - scale(ts_argmax(close, 12)))))
    def alpha060(self):
        high_low_diff = self.high - self.low
        close_low_diff = self.close - self.low
        high_close_diff = self.high - self.close
        inner_value = (close_low_diff - high_close_diff) / high_low_diff
        inner_value_scaled = scale(inner_value * self.volume, k=2)
        close_argmax_12 = ts_argmax(self.close, window=12, freq='h')
        close_argmax_12_scaled = scale(close_argmax_12)
        alpha = 0 - (inner_value_scaled - close_argmax_12_scaled)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha
    
    # Alpha#61:
    # (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
    # (vwap - ts_min(vwap, 15)) < correlation(vwap, sma(close, 240), 24)

    def alpha061(self):
        vwap_min_15 = ts_min(self.vwap, window=15, freq='h')
        close_volume_sma_240 = sma(self.close, window=240, freq='h')
        vwap_sma_corr_24 = correlation(self.vwap, close_volume_sma_240, window=24, freq='h')
        alpha = (self.vwap - vwap_min_15) < vwap_sma_corr_24
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha.astype(int)

    # Alpha#62:
    # ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
    # (correlation(vwap, sum(sma(close, 24), 24), 12) < ((open + open) < ((high + low) / 2 + high))) * -1

    def alpha062(self):
        close_volume_sma_24 = sma(self.close, window=24, freq='h')
        close_volume_sma_24_sum_24 = ts_sum(close_volume_sma_24, window=24, freq='h')
        vwap_sma_corr_12 = correlation(self.vwap, close_volume_sma_24_sum_24, window=12, freq='h')
        high_low_avg = (self.high + self.low) / 2
        open_condition = (self.open + self.open) < (high_low_avg + self.high)
        alpha = (vwap_sma_corr_12 < open_condition) * -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#63:
    # ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)
    # (decay_linear(delta(close, 2), 12) - decay_linear(correlation((vwap * 0.3) + (open * 0.7), sum(sma(close, 240),24), 24), 12)) * -1
    def alpha063(self):
        close_diff_2 = delta(self.close, period=2)
        close_diff_2_decay = decay_linear(close_diff_2, window=12, freq='h')
        vwap_scaled = (self.vwap * 0.3) + (self.open * 0.7)
        close_volume_sma_240 = sma(self.close, window=240, freq='h')
        close_volume_sma_240_sum_24 = ts_sum(close_volume_sma_240, window=24, freq='h')
        vwap_sma_corr_24 = correlation(vwap_scaled, close_volume_sma_240_sum_24, window=24, freq='h')
        vwap_sma_corr_24_decay = decay_linear(vwap_sma_corr_24, window=12, freq='h')
        alpha = (close_diff_2_decay - vwap_sma_corr_24_decay) * -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#64:
    # ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)
    # (correlation(sum((open * 0.2) + (low * 0.8), 12), sum(sma(close, 120), 12), 24) < delta(((high + low) / 2 * 0.2) + (vwap * 0.8), 4)) * -1
    def alpha064(self):
        open_low_sum_12 = ts_sum((self.open * 0.2) + (self.low * 0.8), window=12, freq='h')
        close_volume_sma_120 = sma(self.close, window=120, freq='h')
        close_volume_sma_120_sum_12 = ts_sum(close_volume_sma_120, window=12, freq='h')
        open_low_sma_corr_24 = correlation(open_low_sum_12, close_volume_sma_120_sum_12, window=24, freq='h')
        high_low_avg = (self.high + self.low) / 2
        vwap_scaled = (high_low_avg * 0.2) + (self.vwap * 0.8)
        vwap_scaled_diff_4 = delta(vwap_scaled, period=4)
        alpha = (open_low_sma_corr_24 < vwap_scaled_diff_4) * -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#65:
    # ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
    # (correlation((open * 0.01) + (vwap * 0.99), sum(sma(self.close, 120), 12), 6) < (open - ts_min(open, 12))) * -1
    def alpha065(self):
        open_scaled = self.open * 0.01
        vwap_scaled = self.vwap * 0.99
        open_vwap_sum = open_scaled + vwap_scaled
        close_volume_sma_120 = sma(self.close, window=120, freq='h')
        close_volume_sma_120_sum_12 = ts_sum(close_volume_sma_120, window=12, freq='h')
        open_vwap_sma_corr_6 = correlation(open_vwap_sum, close_volume_sma_120_sum_12, window=6, freq='h')
        open_min_12 = ts_min(self.open, window=12, freq='h')
        open_diff = self.open - open_min_12
        alpha = (open_vwap_sma_corr_6 < open_diff) * -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha
    
    # Alpha#66:
    # ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
    # (decay_linear(delta(vwap, 3), 6) + Ts_Rank(decay_linear(((low - vwap) / (open - ((high + low) / 2))), 12), 6)) * -1
    def alpha066(self):
        vwap_diff_3 = delta(self.vwap, period=3)
        vwap_diff_3_decay = decay_linear(vwap_diff_3, window=6, freq='h')
        low_vwap_diff = (self.low - self.vwap) / (self.open - ((self.high + self.low) / 2))
        low_vwap_diff_decay = decay_linear(low_vwap_diff, window=12, freq='h')
        low_vwap_diff_decay_rank = ts_rank(low_vwap_diff_decay, window=6, freq='h')
        alpha = (vwap_diff_3_decay + low_vwap_diff_decay_rank) * -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#67:
    # ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)
    # ((high - ts_min(high, 2))^correlation(vwap, sma(self.close,12), 6)) * -1

    def alpha067(self):
        high_min_2 = self.high - ts_min(self.high, window=2, freq='h')
        close_volume_sma_12 = sma(self.close, window=12, freq='h')
        vwap_sma_corr_6 = correlation(self.vwap, close_volume_sma_12, window=6, freq='h')
        alpha = (high_min_2 ** vwap_sma_corr_6) * -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#68:
    # ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
    # (Ts_Rank(correlation(high, sma(self.close,12), 6), 12) < delta(((close * 0.5) + (low * 0.5)), 1)) * -1
    def alpha068(self):
        close_volume_sma_12 = sma(self.close, window=12, freq='h')
        high_sma_corr_6 = correlation(self.high, close_volume_sma_12, window=6, freq='h')
        high_sma_corr_6_rank_12 = ts_rank(high_sma_corr_6, window=12, freq='h')
        close_low_avg = (self.close * 0.5) + (self.low * 0.5)
        close_low_avg_diff_1 = delta(close_low_avg, period=1)
        alpha = (high_sma_corr_6_rank_12 < close_low_avg_diff_1) * -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#69:
    # ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)
    # (ts_max(delta(vwap, 3), 6)^Ts_Rank(correlation((close * 0.5) + (vwap * 0.5), sma(self.close, 24), 6), 12)) * -1
    def alpha069(self):
        vwap_diff_3 = delta(self.vwap, period=3)
        vwap_diff_3_max_6 = ts_max(vwap_diff_3, window=6, freq='h')
        close_vwap_avg = (self.close * 0.5) + (self.vwap * 0.5)
        close_vwap_adv_corr_6 = correlation(close_vwap_avg, sma(self.close*self.volume, 24), window=6, freq='h')
        close_vwap_adv_corr_6_rank_12 = ts_rank(close_vwap_adv_corr_6, window=12, freq='h')
        alpha = (vwap_diff_3_max_6 ** close_vwap_adv_corr_6_rank_12) * -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#70:
    # ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)
    # (delta(vwap, 1)^Ts_Rank(correlation(close, sma(self.close,24), 12), 12)) * -1

    def alpha070(self):
        vwap_diff_1 = delta(self.vwap, period=1)
        close_volume_sma_24 = sma(self.close, window=24, freq='h')
        close_sma_corr_12 = correlation(self.close, close_volume_sma_24, window=12, freq='h')
        close_sma_corr_12_rank_12 = ts_rank(close_sma_corr_12, window=12, freq='h')
        alpha = (vwap_diff_1 ** close_sma_corr_12_rank_12) * -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha
    
    # Alpha#71:
    # max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))
    # max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3), sma(self.close,240), 12), 4), 12), Ts_Rank(decay_linear((((low + open) / 2 - vwap)^2), 12), 4))
    def alpha071(self):
        close_rank_3 = ts_rank(self.close, window=3, freq='h')
        close_volume_sma_240 = sma(self.close, window=240, freq='h')
        close_rank_sma_corr_12 = correlation(close_rank_3, close_volume_sma_240, window=12, freq='h')
        close_rank_sma_corr_12_decay_4 = decay_linear(close_rank_sma_corr_12, window=4, freq='h')
        close_rank_sma_corr_12_decay_4_rank_12 = ts_rank(close_rank_sma_corr_12_decay_4, window=12, freq='h')
        low_open_avg = (self.low + self.open) / 2
        low_open_vwap_diff_sq = (low_open_avg - self.vwap) ** 2
        low_open_vwap_diff_sq_decay_12 = decay_linear(low_open_vwap_diff_sq, window=12, freq='h')
        low_open_vwap_diff_sq_decay_12_rank_4 = ts_rank(low_open_vwap_diff_sq_decay_12, window=4, freq='h')
        alpha = pd.concat([close_rank_sma_corr_12_decay_4_rank_12, low_open_vwap_diff_sq_decay_12_rank_4], axis=1).max(axis=1)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # # Alpha#72:
    # # (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))
    # # ((decay_linear(correlation(((high + low) / 2), sma(self.close,24), 6), 12)) / (decay_linear(correlation(Ts_Rank(vwap, 4), Ts_Rank(volume, 12), 6), 4)))
    # def alpha072(self):
    #     high_low_avg = (self.high + self.low) / 2
    #     close_volume_sma_24 = sma(self.close, window=24, freq='h')
    #     high_low_sma_corr_6 = correlation(high_low_avg, close_volume_sma_24, window=6, freq='h')
    #     high_low_sma_corr_6_decay_12 = decay_linear(high_low_sma_corr_6, window=12, freq='h')
    #     vwap_rank_12 = ts_rank(self.vwap, window=12, freq='h')
    #     volume_rank_12 = ts_rank(self.volume, window=12, freq='h')
    #     vwap_volume_corr_4 = correlation(vwap_rank_12, volume_rank_12, window=4, freq='h')
    #     vwap_volume_corr_4_decay_4 = decay_linear(vwap_volume_corr_4, window=4, freq='h')
    #     alpha = high_low_sma_corr_6_decay_12 / vwap_volume_corr_4_decay_4
    #     print(high_low_sma_corr_6)
    #     print(high_low_sma_corr_6_decay_12)
    #     print(vwap_rank_12)
    #     print(volume_rank_12)
    #     print(vwap_volume_corr_4)
    #     print(vwap_volume_corr_4_decay_4)
    #     return alpha


    # Alpha#73:
    # (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
    # (max(rank(decay_linear(delta(vwap, 6), 3)), Ts_Rank(decay_linear(((delta(((open * 0.15) + (low * 0.85)), 2) / ((open * 0.15) + (low * 0.85))) * -1), 3), 12)) * -1)
    def alpha073(self):
        vwap_diff_6 = delta(self.vwap, period=6)
        vwap_diff_6_decay_3 = decay_linear(vwap_diff_6, window=3, freq='h')
        open_low_weighted = (self.open * 0.15) + (self.low * 0.85)
        open_low_weighted_diff_2 = delta(open_low_weighted, period=2)
        open_low_weighted_diff_2_ratio = (open_low_weighted_diff_2 / open_low_weighted) * -1
        open_low_weighted_diff_2_ratio_decay_3 = decay_linear(open_low_weighted_diff_2_ratio, window=3, freq='h')
        open_low_weighted_diff_2_ratio_decay_3_rank_12 = ts_rank(open_low_weighted_diff_2_ratio_decay_3, window=12, freq='h')
        alpha = pd.concat([vwap_diff_6_decay_3, open_low_weighted_diff_2_ratio_decay_3_rank_12], axis=1).max(axis=1) * -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#74:
    # ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)
    # (correlation(close, sum(sma(self.close,24), 24), 12) < correlation(((high * 0.03) + (vwap * 0.97)), volume, 12)) * -1
    def alpha074(self):
        close_volume_sma_24 = sma(self.close, window=24, freq='h')
        close_volume_sma_24_sum_24 = ts_sum(close_volume_sma_24, window=24, freq='h')
        close_sma_corr_12 = correlation(self.close, close_volume_sma_24_sum_24, window=12, freq='h')
        high_scaled = self.high * 0.03
        vwap_scaled = self.vwap * 0.97
        high_vwap_sum = high_scaled + vwap_scaled
        high_vwap_volume_corr_12 = correlation(high_vwap_sum, self.volume, window=12, freq='h')
        alpha = (close_sma_corr_12 < high_vwap_volume_corr_12) * -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#75:
    # (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))
    # correlation(vwap, volume, 4) < correlation(low, sma(self.close,24), 12)
    def alpha075(self):
        vwap_volume_corr_4 = correlation(self.vwap, self.volume, window=4, freq='h')
        close_volume_sma_24 = sma(self.close, window=24, freq='h')
        low_sma_corr_12 = correlation(self.low, close_volume_sma_24, window=12, freq='h')
        alpha = vwap_volume_corr_4 < low_sma_corr_12
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha.astype(int)

    # Alpha#76:
    # (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1)
    # (max(decay_linear(delta(vwap, 1), 12), Ts_Rank(decay_linear(Ts_Rank(correlation(low, sma(self.close,24), 12), 24), 12), 12)) * -1)

    def alpha076(self):
        vwap_diff_1 = delta(self.vwap, period=1)
        vwap_diff_1_decay_12 = decay_linear(vwap_diff_1, window=12, freq='h')
        low_close_volume_sma_24_corr_12 = correlation(self.low, sma(self.close, window=24, freq='h'), window=12, freq='h')
        low_close_volume_sma_24_corr_12_rank_24 = ts_rank(low_close_volume_sma_24_corr_12, window=24, freq='h')
        low_close_volume_sma_24_corr_12_rank_24_decay_12 = decay_linear(low_close_volume_sma_24_corr_12_rank_24, window=12, freq='h')
        low_close_volume_sma_24_corr_12_rank_24_decay_12_rank_12 = ts_rank(low_close_volume_sma_24_corr_12_rank_24_decay_12, window=12, freq='h')
        alpha = pd.concat([vwap_diff_1_decay_12, low_close_volume_sma_24_corr_12_rank_24_decay_12_rank_12], axis=1).max(axis=1) * -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#77: 
    # min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
    # min(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 24), decay_linear(correlation(((high + low) / 2), sma(self.close,24), 6), 12))
    def alpha077(self):
        high_low_avg = (self.high + self.low) / 2
        high_low_avg_high_sum = high_low_avg + self.high
        vwap_high_sum = self.vwap + self.high
        high_low_avg_high_sum_diff = high_low_avg_high_sum - vwap_high_sum
        high_low_avg_high_sum_diff_decay_24 = decay_linear(high_low_avg_high_sum_diff, window=24, freq='h')
        close_volume_sma_24 = sma(self.close, window=24, freq='h')
        high_low_avg_sma_corr_6 = correlation(high_low_avg, close_volume_sma_24, window=6, freq='h')
        high_low_avg_sma_corr_6_decay_12 = decay_linear(high_low_avg_sma_corr_6, window=12, freq='h')
        alpha = pd.concat([high_low_avg_high_sum_diff_decay_24, high_low_avg_sma_corr_6_decay_12], axis=1).min(axis=1)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#78:
    # (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
    # correlation(sum(((low * 0.35) + (vwap * 0.65)), 24), sum(sma(self.close,24), 24), 6)^correlation(vwap, volume, 6)
    def alpha078(self):
        low_scaled = self.low * 0.35
        vwap_scaled = self.vwap * 0.65
        low_vwap_sum = low_scaled + vwap_scaled
        low_vwap_sum_24 = ts_sum(low_vwap_sum, window=24, freq='h')
        close_volume_sma_24 = sma(self.close, window=24, freq='h')
        close_volume_sma_24_sum_24 = ts_sum(close_volume_sma_24, window=24, freq='h')
        low_vwap_close_volume_corr_6 = correlation(low_vwap_sum_24, close_volume_sma_24_sum_24, window=6, freq='h')
        vwap_volume_corr_6 = correlation(self.vwap, self.volume, window=6, freq='h')
        alpha = low_vwap_close_volume_corr_6 ** vwap_volume_corr_6
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha
    
    # Alpha#79:
    # (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))), IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644)))
    # delta(((close * 0.60733) + (open * (1 - 0.60733))), 1) < correlation(Ts_Rank(vwap, 4), Ts_Rank(sma(self.close,240), 12), 12)
    def alpha079(self):
        open_scaled = self.open * (1 - 0.60733)
        close_scaled = self.close * 0.60733
        open_close_weighted = close_scaled + open_scaled
        open_close_weighted_diff_1 = delta(open_close_weighted, period=1)
        vwap_rank_4 = ts_rank(self.vwap, window=4, freq='h')
        close_volume_sma_240 = sma(self.close, window=240, freq='h')
        close_volume_sma_240_rank_12 = ts_rank(close_volume_sma_240, window=12, freq='h')
        vwap_close_volume_corr_12 = correlation(vwap_rank_4, close_volume_sma_240_rank_12, window=12, freq='h')
        alpha = open_close_weighted_diff_1 < vwap_close_volume_corr_12
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha.astype(int)


    # Alpha#80:
    # ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))), IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)
    # (delta(((open * 0.868128) + (high * (1 - 0.868128))), 4)^Ts_Rank(correlation(high, sma(self.close,12), 6), 6)) * -1
    def alpha080(self):
        open_scaled = self.open * 0.868128
        high_scaled = self.high * (1 - 0.868128)
        open_high_weighted = open_scaled + high_scaled
        open_high_weighted_diff_4 = delta(open_high_weighted, period=4)
        close_volume_sma_12 = sma(self.close, window=12, freq='h')
        high_close_volume_corr_6 = correlation(self.high, close_volume_sma_12, window=6, freq='h')
        high_close_volume_corr_6_rank_6 = ts_rank(high_close_volume_corr_6, window=6, freq='h')
        alpha = (open_high_weighted_diff_4 ** high_close_volume_corr_6_rank_6) * -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha
    
    # Alpha#81:
    # ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
    # ((correlation(vwap, sum(sma(self.close,12), 24), 6)^4) < correlation(vwap, volume, 4)) * -1

    def alpha081(self):
        vwap_close_volume_sma_12_sum_24_corr_6 = correlation(self.vwap, ts_sum(sma(self.close, window=12, freq='h'), window=24, freq='h'), window=6, freq='h')
        vwap_volume_corr_4 = correlation(self.vwap, self.volume, window=4, freq='h')
        alpha = ((vwap_close_volume_sma_12_sum_24_corr_6 ** 4) < vwap_volume_corr_4) * -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha.astype(int)


    # Alpha#82:
    # (min(rank(decay_linear(delta(open, 1.46063), 14.8717)), Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)
    # (min(decay_linear(delta(open, 1), 12), Ts_Rank(decay_linear(correlation(volume, open, 12), 6), 12)) * -1)

    def alpha082(self):
        open_diff_1 = delta(self.open, period=1)
        open_diff_1_decay_12 = decay_linear(open_diff_1, window=12, freq='h')
        volume_open_corr_12 = correlation(self.volume, self.open, window=12, freq='h') 
        volume_open_corr_12_decay_6 = decay_linear(volume_open_corr_12, window=6, freq='h')
        volume_open_corr_12_decay_6_rank_12 = ts_rank(volume_open_corr_12_decay_6, window=12, freq='h')
        alpha = pd.concat([open_diff_1_decay_12, volume_open_corr_12_decay_6_rank_12], axis=1).min(axis=1) * -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#83:
    # ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))
    # ((delay(((high - low) / (sum(close, 4) / 4)), 2) * volume) / (((high - low) / (sum(close, 4) / 4)) / (vwap - close)))
    def alpha083(self):
        high_low_diff = self.high - self.low
        close_sum_4 = ts_sum(self.close, window=4, freq='h')
        high_low_diff_close_sum_4_ratio = high_low_diff / (close_sum_4 / 4)
        high_low_diff_close_sum_4_ratio_delay_2 = delay(high_low_diff_close_sum_4_ratio, period=2)
        vwap_close_diff = self.vwap - self.close
        alpha = (high_low_diff_close_sum_4_ratio_delay_2 * self.volume) / (high_low_diff_close_sum_4_ratio / vwap_close_diff)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#84:
    # SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))
    # SignedPower(Ts_Rank((vwap - ts_max(vwap, 12)), 24), delta(close, 4))
    def alpha084(self):
        vwap_vwap_max_12_diff = self.vwap - ts_max(self.vwap, window=12, freq='h')
        vwap_vwap_max_12_diff_rank_24 = ts_rank(vwap_vwap_max_12_diff, window=24, freq='h')
        close_diff_4 = delta(self.close, period=4)
        alpha = signedpower(vwap_vwap_max_12_diff_rank_24, close_diff_4)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#85:
    # (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))
    # correlation(((high * 0.88) + (close * 0.12)), sma(self.close,24), 12)^correlation(Ts_Rank(((high + low) / 2), 4), Ts_Rank(volume, 12), 6)

    def alpha085(self):
        high_scaled = self.high * 0.88
        close_scaled = self.close * 0.12
        high_close_weighted = high_scaled + close_scaled
        close_volume_sma_24 = sma(self.close, window=24, freq='h')
        high_close_weighted_close_volume_sma_24_corr_12 = correlation(high_close_weighted, close_volume_sma_24, window=12, freq='h')
        high_low_avg = (self.high + self.low) / 2
        high_low_avg_rank_4 = ts_rank(high_low_avg, window=4, freq='h')
        volume_rank_12 = ts_rank(self.volume, window=12, freq='h')
        high_low_avg_rank_4_volume_rank_12_corr_6 = correlation(high_low_avg_rank_4, volume_rank_12, window=6, freq='h')
        alpha = high_close_weighted_close_volume_sma_24_corr_12 ** high_low_avg_rank_4_volume_rank_12_corr_6
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha
    


    # Alpha#86:
    # ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open + close) - (vwap + open)))) * -1)
    # ((Ts_Rank(correlation(close, sum(sma(self.close,24), 12), 6), 24) < ((open + close) - (vwap + open))) * -1)
    def alpha086(self):
        close_close_volume_sma_24_sum_12_corr_6 = correlation(self.close, ts_sum(sma(self.close, window=24, freq='h'), window=12, freq='h'), window=6, freq='h')
        close_close_volume_sma_24_sum_12_corr_6_rank_24 = ts_rank(close_close_volume_sma_24_sum_12_corr_6, window=24, freq='h')
        open_close_sum = self.open + self.close
        vwap_open_sum = self.vwap + self.open
        alpha = ((close_close_volume_sma_24_sum_12_corr_6_rank_24 < (open_close_sum - vwap_open_sum)) * -1)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha.astype(int)


    # Alpha#87:
    # (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
    # (max(decay_linear(delta(((close * 0.37) + (vwap * 0.63)), 2), 3), Ts_Rank(decay_linear(abs(correlation(sma(self.close,24), close, 12)), 3), 12)) * -1)

    def alpha087(self):
        close_scaled = self.close * 0.37
        vwap_scaled = self.vwap * 0.63
        close_vwap_weighted = close_scaled + vwap_scaled
        close_vwap_weighted_diff_2 = delta(close_vwap_weighted, period=2)
        close_vwap_weighted_diff_2_decay_3 = decay_linear(close_vwap_weighted_diff_2, window=3, freq='h')
        close_volume_sma_24 = sma(self.close, window=24, freq='h')
        close_volume_sma_24_close_corr_12 = correlation(close_volume_sma_24, self.close, window=12, freq='h')
        close_volume_sma_24_close_corr_12_abs = np.abs(close_volume_sma_24_close_corr_12)
        close_volume_sma_24_close_corr_12_abs_decay_3 = decay_linear(close_volume_sma_24_close_corr_12_abs, window=3, freq='h')
        close_volume_sma_24_close_corr_12_abs_decay_3_rank_12 = ts_rank(close_volume_sma_24_close_corr_12_abs_decay_3, window=12, freq='h')
        alpha = pd.concat([close_vwap_weighted_diff_2_decay_3, close_volume_sma_24_close_corr_12_abs_decay_3_rank_12], axis=1).max(axis=1) * -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#88:
    # min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))
    # min(decay_linear(((open + low) - (high + close)), 6), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8), Ts_Rank(sma(self.close,24), 24), 6), 4), 3))

    def alpha088(self):
        open_low_sum = self.open + self.low
        high_close_sum = self.high + self.close
        open_low_high_close_diff = open_low_sum - high_close_sum
        open_low_high_close_diff_decay_6 = decay_linear(open_low_high_close_diff, window=6, freq='h')
        close_rank_8 = ts_rank(self.close, window=8, freq='h')
        close_volume_sma_24 = sma(self.close, window=24, freq='h')
        close_volume_sma_24_rank_24 = ts_rank(close_volume_sma_24, window=24, freq='h')
        close_rank_8_close_volume_sma_24_rank_24_corr_6 = correlation(close_rank_8, close_volume_sma_24_rank_24, window=6, freq='h')
        close_rank_8_close_volume_sma_24_rank_24_corr_6_decay_4 = decay_linear(close_rank_8_close_volume_sma_24_rank_24_corr_6, window=4, freq='h')
        close_rank_8_close_volume_sma_24_rank_24_corr_6_decay_4_rank_3 = ts_rank(close_rank_8_close_volume_sma_24_rank_24_corr_6_decay_4, window=3, freq='h')
        alpha = pd.concat([open_low_high_close_diff_decay_6, close_rank_8_close_volume_sma_24_rank_24_corr_6_decay_4_rank_3], axis=1).min(axis=1)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#89:
    # (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012))
    # (Ts_Rank(decay_linear(correlation(low, sma(self.close,12), 12), 6), 4) - Ts_Rank(decay_linear(delta(vwap, 3), 6), 12))

    def alpha089(self):
        low_close_volume_sma_12_corr_12 = correlation(self.low, sma(self.close, window=12, freq='h'), window=12, freq='h')
        low_close_volume_sma_12_corr_12_decay_6 = decay_linear(low_close_volume_sma_12_corr_12, window=6, freq='h')
        low_close_volume_sma_12_corr_12_decay_6_rank_4 = ts_rank(low_close_volume_sma_12_corr_12_decay_6, window=4, freq='h')
        vwap_diff_3 = delta(self.vwap, period=3)
        vwap_diff_3_decay_6 = decay_linear(vwap_diff_3, window=6, freq='h')
        vwap_diff_3_decay_6_rank_12 = ts_rank(vwap_diff_3_decay_6, window=12, freq='h')
        alpha = low_close_volume_sma_12_corr_12_decay_6_rank_4 - vwap_diff_3_decay_6_rank_12
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#90:
    # ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry), low, 5.38375), 3.21856)) * -1)
    # ((close - ts_max(close, 5))^Ts_Rank(correlation(sma(self.close,24), low, 5), 3)) * -1

    def alpha090(self):
        close_close_max_5_diff = self.close - ts_max(self.close, window=5, freq='h')
        close_volume_sma_24 = sma(self.close, window=24, freq='h')
        close_volume_sma_24_low_corr_5 = correlation(close_volume_sma_24, self.low, window=5, freq='h')
        close_volume_sma_24_low_corr_5_rank_3 = ts_rank(close_volume_sma_24_low_corr_5, window=3, freq='h')
        alpha = (close_close_max_5_diff ** close_volume_sma_24_low_corr_5_rank_3) * -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha
    # Alpha#91:
    # ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)
    # ((Ts_Rank(decay_linear(decay_linear(correlation(close, volume, 12), 12), 4), 4) - decay_linear(correlation(vwap, sma(self.close,24), 4), 4)) * -1)

    def alpha091(self):
        close_volume_corr_12 = correlation(self.close, self.volume, window=12, freq='h')
        close_volume_corr_12_decay_12 = decay_linear(close_volume_corr_12, window=12, freq='h')
        close_volume_corr_12_decay_12_decay_4 = decay_linear(close_volume_corr_12_decay_12, window=4, freq='h')
        close_volume_corr_12_decay_12_decay_4_rank_4 = ts_rank(close_volume_corr_12_decay_12_decay_4, window=4, freq='h')
        close_volume_sma_24 = sma(self.close, window=24, freq='h')
        vwap_close_volume_sma_24_corr_4 = correlation(self.vwap, close_volume_sma_24, window=4, freq='h')
        vwap_close_volume_sma_24_corr_4_decay_4 = decay_linear(vwap_close_volume_sma_24_corr_4, window=4, freq='h')
        alpha = (close_volume_corr_12_decay_12_decay_4_rank_4 - vwap_close_volume_sma_24_corr_4_decay_4) * -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#92:
    # min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))
    # min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 12), 12), Ts_Rank(decay_linear(correlation(low, sma(self.close,24), 12), 4), 4))

    def alpha092(self):
        high_low_mean = (self.high + self.low) / 2
        high_low_mean_close_sum = high_low_mean + self.close
        low_open_sum = self.low + self.open
        high_low_mean_close_sum_lt_low_open_sum = high_low_mean_close_sum < low_open_sum
        high_low_mean_close_sum_lt_low_open_sum_decay_12 = decay_linear(high_low_mean_close_sum_lt_low_open_sum, window=12, freq='h')
        high_low_mean_close_sum_lt_low_open_sum_decay_12_rank_12 = ts_rank(high_low_mean_close_sum_lt_low_open_sum_decay_12, window=12, freq='h')
        close_volume_sma_24 = sma(self.close, window=24, freq='h')
        low_close_volume_sma_24_corr_12 = correlation(self.low, close_volume_sma_24, window=12, freq='h')
        low_close_volume_sma_24_corr_12_decay_4 = decay_linear(low_close_volume_sma_24_corr_12, window=4, freq='h')
        low_close_volume_sma_24_corr_12_decay_4_rank_4 = ts_rank(low_close_volume_sma_24_corr_12_decay_4, window=4, freq='h')
        alpha = pd.concat([high_low_mean_close_sum_lt_low_open_sum_decay_12_rank_12, low_close_volume_sma_24_corr_12_decay_4_rank_4], axis=1).min(axis=1)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#93:
    # (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))
    # (Ts_Rank(decay_linear(correlation(vwap, sma(self.close,24), 12), 24), 12) / decay_linear(delta(((close * 0.52) + (vwap * 0.48)), 3), 12))

    def alpha093(self):
        close_volume_sma_24 = sma(self.close, window=24, freq='h')
        vwap_close_volume_sma_24_corr_12 = correlation(self.vwap, close_volume_sma_24, window=12, freq='h')
        vwap_close_volume_sma_24_corr_12_decay_24 = decay_linear(vwap_close_volume_sma_24_corr_12, window=24, freq='h')
        vwap_close_volume_sma_24_corr_12_decay_24_rank_12 = ts_rank(vwap_close_volume_sma_24_corr_12_decay_24, window=12, freq='h')
        close_scaled = self.close * 0.52
        vwap_scaled = self.vwap * 0.48
        close_vwap_weighted = close_scaled + vwap_scaled
        close_vwap_weighted_diff_3 = delta(close_vwap_weighted, period=3)
        close_vwap_weighted_diff_3_decay_12 = decay_linear(close_vwap_weighted_diff_3, window=12, freq='h')
        alpha = vwap_close_volume_sma_24_corr_12_decay_24_rank_12 / close_vwap_weighted_diff_3_decay_12
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#94:
    # ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
    # ((vwap - ts_min(vwap, 12))^Ts_Rank(correlation(Ts_Rank(vwap, 24), Ts_Rank(sma(self.close,24), 4), 12), 3)) * -1

    def alpha094(self):
        vwap_vwap_min_12_diff = self.vwap - ts_min(self.vwap, window=12, freq='h')
        vwap_rank_12 = ts_rank(self.vwap, window=12, freq='h')
        close_volume_sma_12 = sma(self.close, window=12, freq='h')
        close_volume_sma_12_rank_4 = ts_rank(close_volume_sma_12, window=4, freq='h')
        vwap_rank_12_close_volume_sma_12_rank_4_corr_4 = correlation(vwap_rank_12, close_volume_sma_12_rank_4, window=4, freq='h')
        vwap_rank_12_close_volume_sma_12_rank_4_corr_4_rank_3 = ts_rank(vwap_rank_12_close_volume_sma_12_rank_4_corr_4, window=3, freq='h')
        alpha = (vwap_vwap_min_12_diff ** vwap_rank_12_close_volume_sma_12_rank_4_corr_4_rank_3) * -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#95:
    # (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
    # ((open - ts_min(open, 12)) < Ts_Rank((correlation(sum(((high + low) / 2), 12), sum(sma(self.close,24), 12), 12)^4), 12))

    def alpha095(self):
        open_open_min_12_diff = self.open - ts_min(self.open, window=12, freq='h')
        high_low_mean = (self.high + self.low) / 2
        high_low_mean_sum_12 = ts_sum(high_low_mean, window=12, freq='h')
        close_volume_sma_24 = sma(self.close, window=24, freq='h')
        close_volume_sma_24_sum_12 = ts_sum(close_volume_sma_24, window=12, freq='h')
        high_low_mean_sum_12_close_volume_sma_24_sum_12_corr_12 = correlation(high_low_mean_sum_12, close_volume_sma_24_sum_12, window=12, freq='h')
        high_low_mean_sum_12_close_volume_sma_24_sum_12_corr_12_pow_4 = high_low_mean_sum_12_close_volume_sma_24_sum_12_corr_12 ** 4
        high_low_mean_sum_12_close_volume_sma_24_sum_12_corr_12_pow_4_rank_12 = ts_rank(high_low_mean_sum_12_close_volume_sma_24_sum_12_corr_12_pow_4, window=12, freq='h')
        alpha = open_open_min_12_diff < high_low_mean_sum_12_close_volume_sma_24_sum_12_corr_12_pow_4_rank_12
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha.astype(int)
    
    # Alpha#96:
    # (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
    # (max(Ts_Rank(decay_linear(correlation(vwap, volume, 4), 4), 12), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 4), Ts_Rank(sma(self.close,24), 4), 4), 12), 12), 12)) * -1)
    def alpha096(self):
        vwap_volume_corr_4 = correlation(self.vwap, self.volume, window=4, freq='h')
        vwap_volume_corr_4_decay_4 = decay_linear(vwap_volume_corr_4, window=4, freq='h')
        vwap_volume_corr_4_decay_4_rank_12 = ts_rank(vwap_volume_corr_4_decay_4, window=12, freq='h')
        close_rank_4 = ts_rank(self.close, window=4, freq='h')
        close_volume_sma_24 = sma(self.close, window=24, freq='h')
        close_volume_sma_24_rank_4 = ts_rank(close_volume_sma_24, window=4, freq='h')
        close_rank_4_close_volume_sma_24_rank_4_corr_4 = correlation(close_rank_4, close_volume_sma_24_rank_4, window=4, freq='h')
        close_rank_4_close_volume_sma_24_rank_4_corr_4_argmax_12 = ts_argmax(close_rank_4_close_volume_sma_24_rank_4_corr_4, window=12, freq='h')
        close_rank_4_close_volume_sma_24_rank_4_corr_4_argmax_12_decay_12 = decay_linear(close_rank_4_close_volume_sma_24_rank_4_corr_4_argmax_12, window=12, freq='h')
        close_rank_4_close_volume_sma_24_rank_4_corr_4_argmax_12_decay_12_rank_12 = ts_rank(close_rank_4_close_volume_sma_24_rank_4_corr_4_argmax_12_decay_12, window=12, freq='h')
        alpha = pd.concat([vwap_volume_corr_4_decay_4_rank_12, close_rank_4_close_volume_sma_24_rank_4_corr_4_argmax_12_decay_12_rank_12], axis=1).max(axis=1) * -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # # Alpha#97:
    # # ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
    # # ((decay_linear(delta(((low * 0.721001) + (vwap * (1 - 0.721001))), 3), 3) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 4), Ts_Rank(sma(self.close,12), 6), 6), 6), 6), 6)) * -1)

    # def alpha097(self):
    #     low_scaled = self.low * 0.721001
    #     vwap_scaled = self.vwap * (1 - 0.721001)
    #     low_vwap_weighted = low_scaled + vwap_scaled
    #     low_vwap_weighted_diff_3 = delta(low_vwap_weighted, period=3)
    #     low_vwap_weighted_diff_3_decay_3 = decay_linear(low_vwap_weighted_diff_3, window=3, freq='h')
    #     low_rank_4 = ts_rank(self.low, window=4, freq='h')
    #     close_volume_sma_12 = sma(self.close, window=12, freq='h')
    #     close_volume_sma_12_rank_12 = ts_rank(close_volume_sma_12, window=12, freq='h')
    #     low_rank_4_close_volume_sma_12_rank_12_corr_6 = correlation(low_rank_4, close_volume_sma_12_rank_12, window=6, freq='h')
    #     low_rank_4_close_volume_sma_12_rank_12_corr_6_rank_6 = ts_rank(low_rank_4_close_volume_sma_12_rank_12_corr_6, window=6, freq='h')
    #     low_rank_4_close_volume_sma_12_rank_12_corr_6_rank_6_decay_6 = decay_linear(low_rank_4_close_volume_sma_12_rank_12_corr_6_rank_6, window=6, freq='h')
    #     low_rank_4_close_volume_sma_12_rank_12_corr_6_rank_6_decay_6_rank_6 = ts_rank(low_rank_4_close_volume_sma_12_rank_12_corr_6_rank_6_decay_6, window=6, freq='h')
    #     alpha = (low_vwap_weighted_diff_3_decay_3 - low_rank_4_close_volume_sma_12_rank_12_corr_6_rank_6_decay_6_rank_6) * -1
    #     return alpha

    # Alpha#98:
    # (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))
    # (decay_linear(correlation(vwap, sum(sma(self.close,12), 24), 4), 6) - decay_linear(Ts_Rank(Ts_ArgMin(correlation(open, sma(self.close,12), 24), 6), 4), 3))

    def alpha098(self):
        close_volume_sma_12 = sma(self.close, window=12, freq='h')
        close_volume_sma_12_sum_24 = ts_sum(close_volume_sma_12, window=24, freq='h')
        vwap_close_volume_sma_12_sum_24_corr_4 = correlation(self.vwap, close_volume_sma_12_sum_24, window=4, freq='h')
        vwap_close_volume_sma_12_sum_24_corr_4_decay_6 = decay_linear(vwap_close_volume_sma_12_sum_24_corr_4, window=6, freq='h')
        open_close_volume_sma_12_corr_24 = correlation(self.open, close_volume_sma_12, window=24, freq='h')
        open_close_volume_sma_12_corr_24_argmin_6 = ts_argmin(open_close_volume_sma_12_corr_24, window=6, freq='h')
        open_close_volume_sma_12_corr_24_argmin_6_rank_4 = ts_rank(open_close_volume_sma_12_corr_24_argmin_6, window=4, freq='h')
        open_close_volume_sma_12_corr_24_argmin_6_rank_4_decay_3 = decay_linear(open_close_volume_sma_12_corr_24_argmin_6_rank_4, window=3, freq='h')
        alpha = vwap_close_volume_sma_12_sum_24_corr_4_decay_6 - open_close_volume_sma_12_corr_24_argmin_6_rank_4_decay_3
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha

    # Alpha#99:
    # ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1)
    # ((correlation(sum(((high + low) / 2), 24), sum(sma(self.close,24), 24), 4) < correlation(low, volume, 4)) * -1)

    def alpha099(self):
        high_low_mean = (self.high + self.low) / 2
        high_low_mean_sum_24 = ts_sum(high_low_mean, window=24, freq='h')
        close_volume_sma_24 = sma(self.close, window=24, freq='h')
        close_volume_sma_24_sum_24 = ts_sum(close_volume_sma_24, window=24, freq='h')
        high_low_mean_sum_24_close_volume_sma_24_sum_24_corr_4 = correlation(high_low_mean_sum_24, close_volume_sma_24_sum_24, window=4, freq='h')
        low_volume_corr_4 = correlation(self.low, self.volume, window=4, freq='h')
        alpha = (high_low_mean_sum_24_close_volume_sma_24_sum_24_corr_4 < low_volume_corr_4) * -1
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha.astype(int)


    # Alpha#100:
    # (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) - scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20))))
    # (0 - (1 * (((1.5 * scale(((((close - low) - (high - close)) / (high - low)) * volume))) - scale((correlation(close, sma(self.close,24), 5) - ts_argmin(close, 24)))) * (volume / sma(self.close,24)))))
    def alpha100(self):
        close_low_diff = self.close - self.low
        high_close_diff = self.high - self.close
        high_low_diff = self.high - self.low
        inner = (close_low_diff - high_close_diff) / high_low_diff
        inner_volume = inner * self.volume
        inner_volume_scale = scale(inner_volume)
        inner_volume_scale_mult = inner_volume_scale * 1.5
        close_volume_sma_24 = sma(self.close, window=24, freq='h')
        close_close_volume_sma_24_corr_5 = correlation(self.close, close_volume_sma_24, window=5, freq='h')
        close_argmin_24 = ts_argmin(self.close, window=24, freq='h')
        close_close_volume_sma_24_corr_5_close_argmin_24_diff = close_close_volume_sma_24_corr_5 - close_argmin_24
        close_close_volume_sma_24_corr_5_close_argmin_24_diff_scale = scale(close_close_volume_sma_24_corr_5_close_argmin_24_diff)
        alpha = 0 - (1 * ((inner_volume_scale_mult - close_close_volume_sma_24_corr_5_close_argmin_24_diff_scale) * (self.volume / close_volume_sma_24)))
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha


    # Alpha#101:
    # ((close - open) / ((high - low) + .001))
    # ((close - open) / ((high - low) + .001))
    def alpha101(self):
        close_open_diff = self.close - self.open
        high_low_diff = self.high - self.low
        alpha = close_open_diff / (high_low_diff + 0.001)
        alpha = alpha.fillna(method='ffill').fillna(method='bfill')
        return alpha