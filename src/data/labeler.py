import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.preprocessing import RobustScaler

class Labeler:
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        
    def load_data(self, ticker):
        df = pd.read_parquet(f'{self.config.data_dir}/raw/{ticker}_1m.pqt')
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.sort_values(by = 'Datetime', ascending = True, inplace = True)
        return df
    
    def resample(self, df, input_freq):
        ohlc_dict = {
            'Open'  :'first',
            'High'  :'max',
            'Low'   :'min',
            'Close' :'last',
            'Volume':'sum'}
        df = df.resample(f'{input_freq}T', on = 'Datetime').agg(ohlc_dict)
        df.reset_index(inplace = True, drop = False)
        return df
    
    def create_labels(self, df, pct_thresh, lookahead):
        
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
    
    def label_distributions(self, df, title):
        
        group_df = df.groupby(by = ['ticker', 'label'])['Datetime'].count().reset_index()
        group_df.columns = ['ticker', 'label', 'count']
        
        pivot_df = group_df.pivot_table(index = 'ticker', columns = ['label'], values = 'count')
        count = pivot_df.sum(axis = 1)
        
        for c in pivot_df.columns:
            pivot_df[c] = pivot_df[c] / count
        
        plt.figure(figsize = (12, 8))
        sns.heatmap(pivot_df, cmap = 'icefire', annot = True, fmt = '.3f')
        plt.title(title)
        plt.show()
    
    def make(self, verbose = True):
        
        ## Parse problem definition
        input_freq = self.config.horizon_def['in_freq']
        pct_thresh = self.config.horizon_def['pct_thresh']
        lookahead  = self.config.horizon_def['lookahead']
        
        # iterate over all the tickers
        dfs = []
        for ticker in self.config.tickers:
            df = self.load_data(ticker = ticker)
            df = self.resample(df = df, input_freq = input_freq)
            df = self.create_labels(df = df, pct_thresh = pct_thresh, lookahead = lookahead)
            
            # percentage change in close price and volume
            df['pct_chg_close'] = df['Close'].pct_change()
            df['pct_chg_volume'] = df['Volume'].pct_change()
            if self.config.project_on_intrarange:
                df['pct_chg_low'] = df['Low'].pct_change()
                df['pct_chg_high'] = df['High'].pct_change()
            df[self.config.features] = df[self.config.features].replace(np.inf, np.nan)
            df = df.fillna(method = 'bfill')
            
            # Apply scaling
            scaler = RobustScaler()
            scaler.fit(df[self.config.features].values)
            df[self.config.features] = scaler.transform(df[self.config.features].values)
            self.scalers[ticker] = scaler
            
            # append ticker
            df['ticker'] = ticker
            
            dfs.append(df)
        
        df = pd.concat(dfs, axis = 0)
        df.reset_index(drop = True, inplace = True)
        if verbose & self.config.show_label_distributions:
            title = f'input time interval: {input_freq}m, lookahead: {lookahead}time_units, pct_threshold: {pct_thresh}'
            self.label_distributions(df = df, title = title)
            
        # encode target label
        df[self.config.target] = df[self.config.target].map(self.config.LABEL_MAPPER)
            
        return df