
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.preprocessing import RobustScaler, StandardScaler
import math
from tqdm import tqdm

class Labeler:
    def __init__(self, config):
        self.config = config
        self.dest_dir = os.path.join(self.config.data_dir, 'processed')
        self.scalers = {}
        
    def load_source_data(self, ticker):
        df = pd.read_parquet(f'{self.config.data_dir}/raw/{ticker}_1m.pqt')
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        return df
    
    def make_volume_bars_for_train(self, df):
        df = df.copy()
        vol_thresh = math.ceil(df['Volume'].mean() * self.volume_bar_multiplier)
        df = df.sort_values(by = 'Datetime', ascending = False)
        data = df.to_numpy()
        timestamps  = data[:, 0]
        opens       = data[:, 1]
        highs       = data[:, 2]
        lows        = data[:, 3]
        closes      = data[:, 4]
        volumes     = data[:, 5]
        dts = []
        bars = np.zeros(shape = (len(timestamps), 5))
        candle_counter, vol, last_i = 0, 0, 0
        for i in range(len(timestamps)):
            vol += volumes[i]
            if vol >= vol_thresh:
                # make volume bar
                dts.append((timestamps[last_i], timestamps[i]))                                                     # start and end timestamp
                bars[candle_counter][0] = closes[last_i]                                                            # open
                bars[candle_counter][1] = np.max([np.max(highs[last_i : i+1]), np.max(closes[last_i : i+1])])       # high
                bars[candle_counter][2] = np.min([np.min(lows[last_i : i+1]), np.min(closes[last_i : i+1])])        # low
                bars[candle_counter][3] = closes[i]                                                                 # close
                bars[candle_counter][4] = np.sum(volumes[last_i : i+1])                                             # volume
                candle_counter += 1
                last_i = i + 1
                vol = 0
        bars = bars[:candle_counter]
        bars = np.column_stack((np.array(dts), bars))
        df = pd.DataFrame(bars, columns = ['Start', 'End', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df = df.sort_values(by = 'End', ascending = True)
        return df, vol_thresh
    
    def make_volume_bars_for_validation(self, df, vol_thresh):
        # make volume bars with 1-timestep delay
        df = df.copy()
        df = df.sort_values(by = 'Datetime', ascending = False)
        data = df.to_numpy()
        timestamps  = data[:, 0]
        opens       = data[:, 1]
        highs       = data[:, 2]
        lows        = data[:, 3]
        closes      = data[:, 4]
        volumes     = data[:, 5]
        dts = []
        bars = np.zeros(shape = (len(timestamps), 5))
        candle_counter = 0
        last_i = 0
        while True:
            if (len(dts) > 0) & (last_i - len(dts) == 0):
                break
            vol = 0
            last_i = len(dts)
            for i in range(len(dts), len(timestamps)):
                vol += volumes[i]
                if vol >= vol_thresh:
                    # make volume bar
                    dts.append((timestamps[last_i], timestamps[i]))                                                     # start and end timestamp
                    bars[candle_counter][0] = closes[last_i]                                                            # open
                    bars[candle_counter][1] = np.max([np.max(highs[last_i : i+1]), np.max(closes[last_i : i+1])])       # high
                    bars[candle_counter][2] = np.min([np.min(lows[last_i : i+1]), np.min(closes[last_i : i+1])])        # low
                    bars[candle_counter][3] = closes[i]                                                                 # close
                    bars[candle_counter][4] = np.sum(volumes[last_i : i+1])                                             # volume
                    break
            candle_counter += 1
            
        bars = bars[:candle_counter-1]
        bars = np.column_stack((np.array(dts), bars))
        df = pd.DataFrame(bars, columns = ['Start', 'End', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df = df.sort_values(by = 'End', ascending = True)
        return df
    
    def make_volume_bars(self, ticker):
        df = self.load_source_data(ticker = ticker)
        df_train, vol_thresh = self.make_volume_bars_for_train(df = df)
        df_test = self.make_volume_bars_for_validation(df = df, vol_thresh = vol_thresh)
        return df_train, df_test
    
    def create_labels(self, df, pct_thresh, lookahead):
        df = df.copy()
        df = df.sort_values(by = 'End', ascending = True)
        # ref labels
        long_col = short_col = 'Close'
        if self.config.project_on_intrarange:
            long_col = 'High'
            short_col = 'Low'
            
        # Long labeling
        df['tgt_long'] = df['Close'] * (1 + pct_thresh)
        df['rolling'] = df[long_col].rolling(lookahead).max().shift(-lookahead)
        df['label_long'] = (df['rolling'] >= df['tgt_long']) * 1

        # Short labeling
        df['tgt_short'] = df['Close'] * (1 - pct_thresh)
        df['rolling'] = df[short_col].rolling(lookahead).min().shift(-lookahead)
        df['label_short'] = (df['rolling'] <= df['tgt_short']) * 1
        
        # Label consolidator
        def label_consolidator(row):
            lng = row['label_long']
            sht = row['label_short']
            combined = lng + sht
            if combined == 2:
                # anything can possible
                label = 'either'
            elif combined == 1:
                label = 'long' if lng > sht else 'short'
            elif combined == 0:
                label = 'neither'
            else: 
                label = None
            return label
        
        df['label'] = df.apply(lambda r: label_consolidator(r), axis = 1)
        
        df.drop(columns = ['tgt_long', 'tgt_short', 'rolling', 'label_long', 'label_short'],
                inplace = True)
        
        return df
        
    def make_dataset(self, ticker, pct_thresh, lookahead):
        train_destpath = f'{self.dest_dir}/train_{self.volume_bar_multiplier}'
        test_destpath = f'{self.dest_dir}/test_{self.volume_bar_multiplier}'
        df_train = df_test = pd.DataFrame()
        if os.path.exists(f'{train_destpath}/{ticker}.pqt'):
            df_train = pd.read_parquet(f'{train_destpath}/{ticker}.pqt')
            df_test = pd.read_parquet(f'{test_destpath}/{ticker}.pqt')
        else:
            df_train, df_test = self.make_volume_bars(ticker = ticker)
            os.makedirs(train_destpath, exist_ok = True)
            os.makedirs(test_destpath, exist_ok = True)
            df_train.to_parquet(f'{train_destpath}/{ticker}.pqt', index = False)
            df_test.to_parquet(f'{test_destpath}/{ticker}.pqt', index = False)
        
        ## Make labels
        df_train = self.create_labels(df_train, pct_thresh = pct_thresh, lookahead = lookahead)
        df_test = self.create_labels(df_test, pct_thresh = pct_thresh, lookahead = lookahead)

        return df_train, df_test
    
    def label_distributions(self, df, title):
        group_df = df.groupby(by = ['ticker', 'label'])[self.config.dt_column].count().reset_index()
        group_df.columns = ['ticker', 'label', 'count']
        
        pivot_df = group_df.pivot_table(index = 'ticker', columns = ['label'], values = 'count')
        count = pivot_df.sum(axis = 1)
        
        for c in pivot_df.columns:
            pivot_df[c] = pivot_df[c] / count
        
        plt.figure(figsize = (12, 8))
        sns.heatmap(pivot_df, cmap = 'icefire', annot = True, fmt = '.3f')
        plt.title(title)
        plt.show()
        
    def make(self, kind = 'train', verbose = True):
        ## Parse problem definition
        self.volume_bar_multiplier = self.config.horizon_def['vol_bar_multiplier']
        input_freq = self.config.horizon_def['in_freq']
        pct_thresh = self.config.horizon_def['pct_thresh']
        lookahead  = self.config.horizon_def['lookahead']
        
        # iterate over all the tickers
        dfs = []
        for ticker in tqdm(self.config.tickers, total = len(self.config.tickers), desc = 'Making dataset'):
            if kind == 'train':
                df, _ = self.make_dataset(ticker, pct_thresh, lookahead)
            if kind == 'test':
                _, df = self.make_dataset(ticker, pct_thresh, lookahead)
            
            # percentage change in close price and volume
            df['pct_chg_close'] = df['Close'].pct_change()
            df['pct_chg_volume'] = df['Volume'].pct_change()
            if self.config.project_on_intrarange:
                df['pct_chg_low'] = df['Low'].pct_change()
                df['pct_chg_high'] = df['High'].pct_change()
            df[self.config.features] = df[self.config.features].replace(np.inf, np.nan)
            df = df.fillna(method = 'bfill')
            
            # Apply scaling
            scaler = StandardScaler()
            scaler.fit(df[self.config.features].values)
            df[self.config.features] = scaler.transform(df[self.config.features].values)
            self.scalers[ticker] = scaler
            
            # append ticker
            df['ticker'] = ticker
            
            dfs.append(df)
        
        df = pd.concat(dfs, axis = 0)
        df.reset_index(drop = True, inplace = True)
        print(df.shape)
        if verbose & self.config.show_label_distributions:
            title = f'lookahead: {lookahead}volume_units, pct_threshold: {pct_thresh}'
            self.label_distributions(df = df, title = title)
            
        # encode target label
        df[self.config.target] = df[self.config.target].map(self.config.LABEL_MAPPER)
        
        return df