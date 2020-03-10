"""
Created on Mon Oct  7 12:17:33 2019

@author: Vanessa
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from sklearn import metrics
import statsmodels.api as sm

class stra_pt():
    def __init__(self):
        print('Load strategy - lstm...')
        return
    
    def ewma(self,df,span):
        a0 = df[:span].mean()
        list = []
        for i in range(len(df)):
            if i < span-1:
                list.append(np.nan)
            elif i == span-1:
                list.append(a0)
            else:
                a = df[i] * (2/(span+1)) + list[-1] * (1-(2/(1+span)))
                list.append(a)
        return list
    
    def load_data(self):
        #df = pd.read_csv("dailydata.csv",index_col = 0, parse_dates = True)
        #df = df[['GBPUSD','USDCHF']].dropna()
        #df = pd.read_csv('GBPCHF.csv',index_col = 0, parse_dates = True).dropna()
        df = pd.read_csv('minuted.csv',index_col = 0, parse_dates = True)
        #df = pd.read_csv('minite.csv',index_col = 0, parse_dates = True)
        df = df[['2','1']]
        df.columns = ['x','y']
        df = df.dropna()
        df['x_n'] = df['x']/df['x'].values[0]
        df['y_n'] = df['y']/df['y'].values[0]
        df['x_'] = df['x']/df['x'].values[0]*df['y']
        #print(df)
        df['rsi_x'] = self.RSI(df['x'], 5)
        df['rsi_y'] = self.RSI(df['y'], 5)
        #df['20ewma_y'] = self.ewma(df.iloc[:,1],2)
        #df['50ewma_y'] = self.ewma(df.iloc[:,1],15)
        #df['20std_y'] = df['20ewma_y'].rolling(window = 20).std()
        #df['20up_y'] = df['20ewma_y'] + 2 * df['20std_y'] 
        #df['20dn_y'] = df['20ewma_y'] - 2 * df['20std_y']
        #df['20ewma_x'] = self.ewma(df.iloc[:,0],2)
        #df['50ewma_x'] = self.ewma(df.iloc[:,0],15)
        #df['20std_x'] = df['20ewma_x'].rolling(window = 20).std()
        #df['20up_x'] = df['20ewma_x'] + 2 * df['20std_x'] 
        #df['20dn_x'] = df['20ewma_x'] - 2 * df['20std_x']
        df['vol_x'] = df['x'].rolling(window = 5).std()
        df['vol_y'] = df['x'].rolling(window = 5).std()
        df = df.dropna()
        self.data['raw'] = df.copy()
        return
    
    def OLS_pred(self,bt_start,a,b,c,d,e):
        
        lenth = len(self.data['raw'].loc[:bt_start,:])-1100
        date_st = self.data['raw'].index.values[lenth]
        df = self.data['raw'].loc[date_st:bt_start,:].copy()
        df_t = self.data['raw'].loc[bt_start:,:].copy()
        xtrain = df.iloc[:,2]
        ytrain = df.iloc[:,3] 
        xtest = df_t.iloc[:,2]
        ytest = df_t.iloc[:,3]
        x = sm.add_constant(xtrain)
        y = ytrain
        model = sm.OLS(y,x).fit()
        df_t['beta'] = model.params[1]
        res_train = model.resid
        res_train_mean = res_train.mean()
        res_train_vol = res_train.std()
        z_train = (res_train - res_train_mean) / res_train_vol 
        
        #print(res_train_vol)
        z_train_mean = z_train.mean()
        params = model.params
        df_t['y_pred'] = params[0] + xtest * params[1]
        self.alpha.append(params[0])
        self.beta.append(params[1])
        df_t['res'] = df_t['y_pred'] - ytest
        df_t['z'] = (df_t['res'] - res_train_mean) / res_train_vol
        df_t['zup'] = z_train_mean + a
        df_t['zupup'] = z_train_mean + b
        df_t['zupupup'] = z_train_mean + c
        df_t['zdn'] = z_train_mean - a
        df_t['zdndn'] = z_train_mean - b
        df_t['zdndndn'] = z_train_mean - c
        df_t['mid'] = d
        df_t['uppermid'] = e
        self.data['cooked_bt'] = df_t.copy()

        return

    def RSI(self,close, period):
        delta = close.diff().dropna()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
		# Calculate the SMA
        roll_up2 = up.rolling(window = period).mean()
        roll_down2 = down.abs().rolling(window = period).mean()
		# Calculate the RSI based on SMA
        RS2 = roll_up2 / roll_down2
        RSI2 = 100.0 - (100.0 / (1.0 + RS2))
        return RSI2
    
    
    def turtle(self):
        self.now.update(self.data['cooked_bt'].T[self.now['time']].to_dict())
        if self.i == 0:
            self.now['con_1'] = 0.
            self.now['con_2'] = 0.
            self.now['p&l'      ] = 0.
            self.now['interest' ] = 0.
            self.now['comm'     ] = 0.
            self.now['interest' ] = 0.
            self.now['signal'   ] = None
            self.now['type'] = 0.
            return

        above_n_down = self.now['z'] <= self.record['zup'].values[-1] and self.record['z'].values[-1] >= self.record['zup'].values[-1]
        above_n_up = self.now['z'] >= self.record['zup'].values[-1] and self.record['z'].values[-1] >= self.record['zup'].values[-1]
        below_n_down = self.now['z'] < self.record['zdn'].values[-1] and self.record['z'].values[-1] < self.record['zdn'].values[-1]
        below_n_up = self.now['z'] >= self.record['zdn'].values[-1] and self.record['z'].values[-1] < self.record['zdn'].values[-1]
        above_mid = self.now['z'] >= self.record['uppermid'].values[-1]
        below_mid = self.now['z'] <= -self.record['uppermid'].values[-1]
        blw_upper = self.now['z'] <= self.record['zupup'].values[-1]
        abv_lower = self.now['z'] >= self.record['zdndn'].values[-1]
        around_mean = self.now['z'] <= self.record['mid'].values[-1] and self.now['z'] >= -self.record['mid'].values[-1]
        stop_loss1 = self.now['z'] >= self.record['zupupup'].values[-1] 
        stop_loss2 = self.now['z'] <= self.record['zdndndn'].values[-1]
        change_sign = self.now['z'] * self.record['z'].values[-1] < 0

        l = 4
        
        self.now['p&l'] = self.now['con_1'] * (self.now['y'] - self.record['y'].values[-1]) +\
                          self.now['con_2'] * (self.now['x'] - self.record['x'].values[-1])

        if self.record['type'].values[-1] == 0.:
            if above_n_down:
                if above_mid:
                    self.now['signal'] = 'sell y, buy x'
                    #buy usd sell other currencies
                    self.now['con_1'] = - np.floor(self.now['cash']/1000*0.5)*l*1000
                    self.now['con_2'] =  np.floor(self.now['cash']/1000*0.5)*l*1000
                    self.now['comm'] = 2
                    self.now['interest'] = 0.
                    self.now['type'] = -1
                else:
                    self.now['signal'] = 'None1'
                    self.now['con_1'] = self.record['con_1'].values[-1]
                    self.now['con_2'] = self.record['con_2'].values[-1]
                    self.now['comm'] = 0.
                    self.now['interest'] = 0.
                    self.now['type'] = self.record['type'].values[-1]
            elif above_n_up:
                if blw_upper:
                    self.now['signal'] = 'buy y, sell x'
                    self.now['con_1'] = + np.floor(self.now['cash']/1000*0.5)*l*1000
                    self.now['con_2'] = - np.floor(self.now['cash']/1000*0.5)*l*1000
                    self.now['comm'] = 2
                    self.now['interest'] = 0.
                    self.now['type'] = 1
                else:
                    self.now['signal'] = 'None1'
                    self.now['con_1'] = self.record['con_1'].values[-1]
                    self.now['con_2'] = self.record['con_2'].values[-1]
                    self.now['comm'] = 0.
                    self.now['interest'] = 0.
                    self.now['type'] = self.record['type'].values[-1]
            elif below_n_up:
                if below_mid:
                    self.now['signal'] = 'buy y, sell x'
                    self.now['con_1'] =  np.floor(self.now['cash']/1000*0.5)*l*1000
                    self.now['con_2'] = - np.floor(self.now['cash']/1000*0.5)*l*1000
                    self.now['comm'] = 2
                    self.now['interest'] = 0.
                    self.now['type'] = 1
                else:
                    self.now['signal'] = 'None1'
                    self.now['con_1'] = self.record['con_1'].values[-1]
                    self.now['con_2'] = self.record['con_2'].values[-1]
                    self.now['comm'] = 0.
                    self.now['interest'] = 0.
                    self.now['type'] = self.record['type'].values[-1]
            elif below_n_down:
                if abv_lower:
                    self.now['signal'] = 'sell y, buy x'
                    self.now['con_1'] = - np.floor(self.now['cash']/1000*0.5)*l*1000
                    self.now['con_2'] =  np.floor(self.now['cash']/1000*0.5)*l*1000
                    self.now['comm'] = 2
                    self.now['interest'] = 0.
                    self.now['type'] = -1
                else:
                    self.now['signal'] = 'None1'
                    self.now['con_1'] = self.record['con_1'].values[-1]
                    self.now['con_2'] = self.record['con_2'].values[-1]
                    self.now['comm'] = 0.
                    self.now['interest'] = 0.
                    self.now['type'] = self.record['type'].values[-1]
            else:
                self.now['signal'] = 'None1'
                self.now['con_1'] = self.record['con_1'].values[-1]
                self.now['con_2'] = self.record['con_2'].values[-1]
                self.now['comm'] = 0.
                self.now['interest'] = 0.
                self.now['type'] = self.record['type'].values[-1]
            
        elif self.record['type'].values[-1] == -1:
            if above_n_up:
                if blw_upper:
                    self.now['signal'] = 'buy y, sell x'
                    self.now['con_1'] = + np.floor(self.now['cash']/1000*0.5)*l*1000
                    self.now['con_2'] = - np.floor(self.now['cash']/1000*0.5)*l*1000
                    self.now['comm'] = 2.
                    self.now['interest'] = 0.
                    self.now['type'] = 1.
                else:
                    self.now['signal'] = 'None1'
                    self.now['con_1'] = self.record['con_1'].values[-1]
                    self.now['con_2'] = self.record['con_2'].values[-1]
                    self.now['comm'] = 0.
                    self.now['interest'] = 0.
                    self.now['type'] = self.record['type'].values[-1]
                
            elif below_n_up:
                if below_mid:
                    self.now['signal'] = 'buy y, sell x'
                    self.now['con_1'] = + np.floor(self.now['cash']/1000*0.5)*l*1000
                    self.now['con_2'] = - np.floor(self.now['cash']/1000*0.5)*l*1000
                    self.now['comm'] = 2.
                    self.now['interest'] = 0.
                    self.now['type'] = 1.
                else:
                    self.now['signal'] = 'None1'
                    self.now['con_1'] = self.record['con_1'].values[-1]
                    self.now['con_2'] = self.record['con_2'].values[-1]
                    self.now['comm'] = 0.
                    self.now['interest'] = 0.
                    self.now['type'] = self.record['type'].values[-1]
                
            elif around_mean or change_sign or stop_loss1:
                self.now['signal'] = 'unwind'
                self.now['con_1'] = 0.
                self.now['con_2'] = 0.
                self.now['interest'] = 0.
                self.now['comm'] = 2.
                self.now['type'] = 0.
                
            else:
                self.now['signal'] = 'None2'
                self.now['con_1'] = self.record['con_1'].values[-1]
                self.now['con_2'] = self.record['con_2'].values[-1]
                self.now['comm'] = 0.
                #self.now['interest'] = (abs(self.now['con_1']) * (0.01 + 0.015)/360) + (abs(self.now['con_2']) * (0.01 + 0.015)/360)
                self.now['type'] = self.record['type'].values[-1]
                
        else:
            if below_n_down:
                if abv_lower:
                    self.now['signal'] = 'sell y, buy x'
                    self.now['con_1'] = - np.floor(self.now['cash']/1000*0.5)*l*1000
                    self.now['con_2'] = + np.floor(self.now['cash']/1000*0.5)*l*1000
                    self.now['comm'] = 2.
                    self.now['interest'] = 0.
                    self.now['type'] = -1.
                else:
                    self.now['signal'] = 'None1'
                    self.now['con_1'] = self.record['con_1'].values[-1]
                    self.now['con_2'] = self.record['con_2'].values[-1]
                    self.now['comm'] = 0.
                    self.now['interest'] = 0.
                    self.now['type'] = self.record['type'].values[-1]
                    
            elif above_n_down:
                if above_mid:
                    self.now['signal'] = 'sell y, buy x'
                    self.now['con_1'] = - np.floor(self.now['cash']/1000*0.5)*l*1000
                    self.now['con_2'] = + np.floor(self.now['cash']/1000*0.5)*l*1000
                    self.now['comm'] = 2.
                    self.now['interest'] = 0.
                    self.now['type'] = -1.
                else:
                    self.now['signal'] = 'None1'
                    self.now['con_1'] = self.record['con_1'].values[-1]
                    self.now['con_2'] = self.record['con_2'].values[-1]
                    self.now['comm'] = 0.
                    self.now['interest'] = 0.
                    self.now['type'] = self.record['type'].values[-1]
                    
            elif around_mean or stop_loss2 or change_sign :
                self.now['signal'] = 'unwind'
                self.now['con_1'] = 0.
                self.now['con_2'] = 0.
                self.now['interest'] = 0.
                self.now['comm'] = 2.
                self.now['type'] = 0.
                
            else:
                self.now['signal'] = 'None2'
                self.now['con_1'] = self.record['con_1'].values[-1]
                self.now['con_2'] = self.record['con_2'].values[-1]
                self.now['comm'] = 0.
                #self.now['interest'] = (abs(self.now['con_1']) * (0.01 + 0.015)/360) + (abs(self.now['con_2']) * (0.01 + 0.015)/360)
                self.now['type'] = self.record['type'].values[-1]
            
        self.now['cash'] = self.now['cash'] + self.now['p&l'] - self.now['comm'] - self.now['interest']
        self.now['mdd'] = (self.record['cash'].values.max() - self.now['cash']) / self.record['cash'].values.max() * 100
        if self.now['mdd'] < 0.:
            self.now['mdd'] = 0.
        return

