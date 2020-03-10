# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 12:25:43 2019

@author: Vanessa
"""

import numpy as np
import pandas as pd
import my_str_pt
import matplotlib.pyplot as plt

class my_main( my_str_pt.stra_pt ):
    
    
    ###########################################################################
    def __init__(self):
        my_str_pt.stra_pt.__init__(self) 
        print('Load my_backtesting class...')
        #################################
        self.interest = []
        self.now = {}
        self.data = {}
        self.record = pd.DataFrame()
        self.blist = []
        self.alist = []
        self.alpha = []
        self.beta = []
        self.clist = []
        self.count1 = 0.
        self.count2 = 0.
        self.load_data()
        self.buy = 0.
        self.sell = 0.
        self.buy1 = 0.
        self.sell1 = 0.
        return
    
    def run(self, params):
        date_all = self.data['raw'].index
        date_bt  = date_all[ ( date_all >= params['begin_date'] ) &\
                             ( date_all <= params['end_date'  ] )   ]
        
        self.now['cash'] = params['initialE']
        self.i = 0
        while self.i < len(date_bt):
            self.now['time'] = date_bt[ self.i ]
            self.OLS_pred(self.now['time'],a,b,c,d,e)
            self.turtle()
            self.record = self.record.append(self.now, ignore_index = True)
            self.i += 1        
        return
    def graph(self,record,params):
        l = np.arange(len(record.index))
        plt.figure(figsize=(15,5))
        plt.plot(l,record['p&l'])
        plt.grid()
        plt.title(params['begin_date'] + ' - ' + params['end_date'] +' '+ 'P&L')
        plt.xlabel('Backtesting Minutes')
        plt.ylabel('Dollar(4x)')
        plt.show()
        
        plt.figure(figsize=(15,5))
        plt.plot(l,record['mdd'])
        plt.grid()
        plt.title(params['begin_date'] + ' - ' + params['end_date'] +' '+ 'Maximum Drawdown')
        plt.xlabel('Backtesting Minutes')
        plt.ylabel('percentage(%)')
        plt.show()     
        
        
        fig1 = plt.figure(figsize=(60,20))
        ax1 = fig1.add_subplot(222)
        ax1.set_title(params['begin_date'] + ' to ' + params['end_date'] + ' Forex x Movement')
        ax1.set_xlabel('Backtesting Minutes')
        ax1.set_ylabel('Voliatility')
        ax2 = ax1.twinx()
        ax2.set_xlabel('Backtesting Minutes')
        ax2.set_ylabel('FX Rate')
        A1 = record[record['signal']=='buy y, sell x']['x']
        Aid1 = record[record['signal']=='buy y, sell x'].index
        
        A2 = record[record['signal']=='buy y, sell x']['y']
        Aid2 = record[record['signal']=='buy y, sell x'].index
        
        B1 = record[record['signal']=='sell y, buy x']['x']
        Bid1 = record[record['signal']=='sell y, buy x'].index
        
        B2 = record[record['signal']=='sell y, buy x']['y']
        Bid2 = record[record['signal']=='sell y, buy x'].index
        
        C1 = record[record['signal']=='unwind']['x']
        Cid1 = record[record['signal']=='unwind'].index
        
        C2 = record[record['signal']=='unwind']['y']
        Cid2 = record[record['signal']=='unwind'].index
        
        D1 = record[record['signal']=='buy y, buy x']['x']
        Did1 = record[record['signal']=='buy y, buy x'].index
        
        D2 = record[record['signal']=='buy y, buy x']['y']
        Did2 = record[record['signal']=='buy y, buy x'].index
        
        E1 = record[record['signal']=='sell y, sell x']['x']
        Eid1 = record[record['signal']=='sell y, sell x'].index
        
        E2 = record[record['signal']=='sell y, sell x']['y']
        Eid2 = record[record['signal']=='sell y, sell x'].index
        
        ax1.plot(l,record['vol_x'],color = 'grey')
        ax1.plot(l,record['rsi_x']/100000, color = 'grey')
        ax1.grid()
        
        ax2.scatter(Aid1,A1,label = 'buy y, sell x',color = 'g', zorder=2)
        ax2.scatter(Bid1,B1,label = 'sell y, buy x',color = 'orange', zorder=2)
        ax2.scatter(Cid1,C1,label = 'unwind',color = 'r', zorder=2)
        #ax2.scatter(Did1,D1,label = 'buy y, buy x',color = 'g', marker = 's', zorder=2)
        #ax2.scatter(Eid1,E1,label = 'sell y, sell x',color = 'black',marker = 's', zorder=2)
        ax2.plot(l,record['x'] ,label = 'x', zorder=1)
        #ax2.plot(l,record['20up_x'] ,label = 'x + 2 std', zorder=1)
        #ax2.plot(l,record['20dn_x'] ,label = 'x - 2 std',color = 'pink', zorder=1)
        #ax2.plot(l,record['50ewma_x'] ,label = '50ewma', zorder=1)
        #ax2.set_ylim((,))
        #ax2.set_ylim((,))
        ax2.legend()
        ax2.grid()
        
        A1_ = record[record['signal']=='buy y, sell x']['x_n']
        Aid1_ = record[record['signal']=='buy y, sell x'].index
        
        A2_ = record[record['signal']=='buy y, sell x']['y_n']
        Aid2_ = record[record['signal']=='buy y, sell x'].index
        
        B1_ = record[record['signal']=='sell y, buy x']['x_n']
        Bid1_ = record[record['signal']=='sell y, buy x'].index
        
        B2_ = record[record['signal']=='sell y, buy x']['y_n']
        Bid2_ = record[record['signal']=='sell y, buy x'].index
        
        C1_ = record[record['signal']=='unwind']['x_n']
        Cid1_ = record[record['signal']=='unwind'].index
        
        C2_ = record[record['signal']=='unwind']['y_n']
        Cid2_ = record[record['signal']=='unwind'].index

        
        plt.figure(figsize=(30,10))
        plt.scatter(Aid2_,A2_,label = 'buy y, sell x',color = 'g')
        plt.scatter(Bid2_,B2_,label = 'sell y, buy x',color = 'orange')
        plt.scatter(Cid2_,C2_,label = 'unwind',color = 'r')
        plt.scatter(Aid1_,A1_,label = 'buy y, sell x',color = 'g')
        plt.scatter(Bid1_,B1_,label = 'sell y, buy x',color = 'orange')
        plt.scatter(Cid1_,C1_,label = 'unwind',color = 'r')
        #plt.scatter(Did2,D2,label = 'buy y, buy x',color = 'g', marker = 's')
        #plt.scatter(Eid2,E2,label = 'sell y, sell x',color = 'orange',marker = 's')
        plt.plot(l,record['y_n'],label = 'normalized y')
        plt.plot(l,record['x_n'] ,label = 'normalized x')
        #plt.plot(l,record['20up_y'] ,label = 'y + 2 std')
        #plt.plot(l,record['20dn_y'] ,label = 'y - 2 std')
        #plt.plot(l,record['50ewma_y'] ,label = '50ewma')
        plt.title(params['begin_date'] + ' - ' + params['end_date'] +' '+ 'Forex y Movement')
        plt.xlabel('Backtesting Minutes')
        plt.ylabel('Forex Rate')
        plt.legend(loc = 4)
        #plt.ylim((,))
        plt.grid()
        plt.show()

        plt.figure(figsize=(30,10))
        plt.plot(l,record['cash'],'b',label = 'Acc. Dollar')


        plt.title(params['begin_date'] + ' - ' + params['end_date'] +' '+ 'Profit')
        plt.xlabel('Backtesting Minutes')
        plt.ylabel('Dollar')
        plt.legend(loc = 4)
        #plt.ylim((0.635, 0.645))
        plt.grid()
        plt.show()
        
        
        
        fig = plt.figure(figsize=(30,10))
        ax1 = fig.add_subplot(111)
        ax1.set_title(params['begin_date'] + ' to ' + params['end_date'] + ' P&L and Z')
        #ax1.plot(l,record['cash'],'b')
        #plt.title(params['begin_date'] + ' - ' + params['end_date'] +' '+ 'Cash')
        ax1.set_xlabel('Backtesting Minutes')
        ax1.set_ylabel('Accumulated Dollar')
        ax2 = ax1.twinx()
        A = record[record['signal']=='buy y, sell x']['z']
        Aid = record[record['signal']=='buy y, sell x'].index
        B = record[record['signal']=='sell y, buy x']['z']
        Bid = record[record['signal']=='sell y, buy x'].index
        C = record[record['signal']=='unwind']['z']
        Cid = record[record['signal']=='unwind'].index
        #D = record[record['signal']=='buy y, buy x']['z']
        #Did = record[record['signal']=='buy y, buy x'].index
        #E = record[record['signal']=='sell y, sell x']['z']
        #Eid = record[record['signal']=='sell y, sell x'].index
        ax2.scatter(Aid,A,label = 'buy y, sell x',color = 'g', zorder=2)
        ax2.scatter(Bid,B,label = 'sell y, buy x',color = 'orange', zorder=2)
        #ax2.scatter(Did,D,label = 'buy y, buy x',color = 'g',marker = 's', zorder=2)
        #ax2.scatter(Eid,E,label = 'sell y, sell x',color = 'orange',marker = 's', zorder=2)
        ax2.scatter(Cid,C,label = 'unwind',color = 'r', zorder=2)
        ax2.plot(l,record['z'],color = 'grey', zorder=1)
        ax2.plot(l,record['zup'], color = 'grey', zorder=1)
        ax2.plot(l,record['zupup'], color = 'grey', zorder=1)
        ax2.plot(l,record['zupupup'], color = 'grey')
        ax2.plot(l,record['zdndndn'] , color = 'grey')
        ax2.plot(l,record['zdn'] , color = 'grey',label = 'signal bnd', zorder=1)
        ax2.plot(l,record['zdndn'] , color = 'grey', label = 'stoploss bnd', zorder=1)
        ax2.hlines(0,l, 0, color = 'grey', zorder=1)
        
        ax2.grid()
        #ax1.title(params['begin_date'] + ' - ' + params['end_date'] +' '+ 'Trading Signals')
        ax2.set_xlabel('Backtesting Minutes')
        ax2.set_ylabel('z score')

if __name__ == '__main__':
    self = my_main()
    params = {'begin_date': '2020-02-18  06:15:00',
              'end_date'  : '2020-02-20  06:15:00',
              'initialE'  : 1e5        }
    a = 
    b = 
    c = 
    d = 
    e = 
    self.run(params)
    record = self.record.copy()
    raw =  self.data['raw'].copy()
    self.graph(record,params)