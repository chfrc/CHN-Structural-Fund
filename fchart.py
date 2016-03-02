#%matplotlib inline
import time
import datetime
import DataAPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.finance import candlestick
import matplotlib
matplotlib.rcParams.update({'font.size':9})

def getData(sid, start = '20121201', end = '20150501'):
    '''
    ## getData ##
    grab price/date data using DataAPI.
    '''
    df = DataAPI.MktEqudGet(ticker='{}'.format(sid), 
                            beginDate = start, endDate = end,
                            field = ['tradeDate', 'closePrice',
                                    'highestPrice','lowestPrice',
                                    'openPrice','dealAmount'])
    df.columns = ['date','closep','highp','lowp','openp','volume']
    df.date = [mdates.datestr2num(df.date[t]) for t in df.index]
    for t in df[df.volume==0].index:
        if t==0: continue
        df.openp[t],df.closep[t],df.highp[t],df.lowp[t] = df.closep[t-1],df.closep[t-1],df.closep[t-1],df.closep[t-1]
    #if save:
    #    df.to_csv('tempdata.csv')
    return df

def rsiFunc(prices, n=14):
    '''
    ##[RSI] Relative Strength Index ##
    incicator of whether a stock is overbought or oversold
    signal: <30, >70
    relative stength = avg.gain/avg.loss (based upon some period
    '''

    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)
    for i in range(n, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta     
        up = (up*(n-1)+upval)/n
        down = (down*(n-1)+downval)/n
        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)    
    return rsi

def movingAverage(values, window):
    '''
    ##[SMA] Simple Moving Average ##
    '''
    weights = np.repeat(1.0, window)/window
    smas = np.convolve(values, weights, 'valid')
    return smas

def expMovingAverage(values, window):
    '''
    ##[SMA] Exponential moving average ##
    '''
    weights = np.exp(np.linspace(-1.,0.,window))
    weights /= weights.sum()
    emas = np.convolve(values, weights, mode='full')[:len(values)]
    emas[:window] = emas[window]
    return emas

def computeMACD(x, slow=26, fast=12):
    '''
    ##[MACD] Moving Average Convergence/Divergence ##
    MACD Line = 12EMA - 26EMA
    signal_line = MACD_9EMA 
    histogram = MACD - signal
    '''
    emaslow = expMovingAverage(x, slow)
    emafast = expMovingAverage(x, fast)
    return emaslow, emafast, emafast-emaslow


def graphData(stock, MA1=12, MA2=26, online=0, figsave=1):
    '''
    ###### Plot Data ######
    Parameters: {'stock' : stockticker, 
                 'MA1': MA window(short,default=12), 'MA2': MA window(long,default=26),
                 'online': 1 if use 通联's dataAPI, 'figsave': 1 then save figure as pdf }
    Loaded indicators: RSI, SMAs, MACD, Volumes
    '''
    dataGrabFlag = True
    #############################################
    
    if online:
        try:
            df = getData(stock)
        except Exception, e:
            dataGrabFlag = False
            print 'failed main loop',str(e)
    if not online:
        try:
            df = pd.read_csv('data1.csv')
        except Exception, e:
            dataGrabFlag = False
            print 'failed main loop',str(e)
     
    #############################################
    if dataGrabFlag:
        t = 0
        candleAr = []
        while t < len(df):
            appendLine = df.date[t],df.openp[t],df.closep[t],df.highp[t],df.lowp[t],df.volume[t]
            candleAr.append(appendLine)
            t += 1
            
        Av1 = movingAverage(df.closep,MA1)
        Av2 = movingAverage(df.closep,MA2)
        SP = len(df[MA2-1:])
        label1 = str(MA1)+' SMA'
        label2 = str(MA2)+' SMA'
        
        fig = plt.figure(figsize=(10,10), facecolor='#07000d')
        
        ### AX1: Candlesticks, SMAs, Embed Volumes
        ax1 = plt.subplot2grid((6,4), (0,0), rowspan=4, colspan=4, axisbg = '#07000d')
        candlestick(ax1, candleAr[-SP:], width=0.8, colorup='#53c156', colordown='#ff1717')
        ax1.plot(df.date[-SP:], Av1[-SP:], '#e1edf9', label=label1, linewidth=1.5)
        ax1.plot(df.date[-SP:], Av2[-SP:], '#4ee6fd', label=label2, linewidth=1.5)
        
        ax1.grid(True, color='w')
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='lower'))
        ax1.yaxis.label.set_color('w')
        ax1.spines['top'].set_color('#5998ff')
        ax1.spines['bottom'].set_color('#5998ff')
        ax1.spines['left'].set_color('#5998ff')
        ax1.spines['right'].set_color('#5998ff')
        ax1.tick_params(axis='y', colors='w')
        ax1.tick_params(axis='x', colors='w')
        plt.ylabel('Stock Price & Volume')
        
        ######## Exclude Legend, pylab not supported. #########
        #maLeg = plt.legend(loc=9, ncol=2, prop={'size':9}, fancybox=True, borderaxespad=0)
        #maLeg.get_frame().set_alpha(0.4)
        #textEd = pylab.gca().get_legend().get_texts()
        #pylab.setp(textEd[0:5], color='w')

        ### AX1v volumes
        volumeMin = 0
        ax1v = ax1.twinx()
        ax1v.fill_between(df.date[-SP:],volumeMin,df.volume[-SP:],
                          facecolor='#00ffe8',alpha=.5)
        ax1v.grid(False)
        ax1v.axes.yaxis.set_ticklabels([])
        ax1v.spines['top'].set_color('#5998ff')
        ax1v.spines['bottom'].set_color('#5998ff')
        ax1v.spines['left'].set_color('#5998ff')
        ax1v.spines['right'].set_color('#5998ff')
        ax1v.set_ylim(0,2.7*df.volume.max())
        ax1v.tick_params(axis='x', colors='w')
        ax1v.tick_params(axis='y', colors='w')

        ### AX2: RSI
        ax2 = plt.subplot2grid((6,4),(4,0),sharex=ax1,
                                rowspan=1,colspan=4,axisbg = '#07000d')
        rsi = rsiFunc(df.closep)
        rsiCol = '#1a8782'
        poiCol = '#386d13'
        negCol = '#8f2020'
        
        ax2.plot(df.date[-SP:], rsi[-SP:], rsiCol, linewidth=1.5)
        ax2.axhline(70, color=negCol)
        ax2.axhline(30, color=poiCol)
        ax2.fill_between(df.date[-SP:], rsi[-SP:], 70, where=(rsi[-SP:]>=70),
                         facecolor = negCol, edgecolor = negCol)
        ax2.fill_between(df.date[-SP:], rsi[-SP:], 30, where=(rsi[-SP:]<30),
                         facecolor = poiCol, edgecolor = poiCol)
        
        ax2.spines['top'].set_color('#5998ff')
        ax2.spines['bottom'].set_color('#5998ff')
        ax2.spines['left'].set_color('#5998ff')
        ax2.spines['right'].set_color('#5998ff')
        ax2.tick_params(axis='x', colors='w')
        ax2.tick_params(axis='y', colors='w')
        ax2.set_yticks([30,70])
        ax2.yaxis.label.set_color('w')
        #plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='lower'))
        ax2.text(0.015, 0.95, 'RSI (14)', va='top',color='w',transform=ax2.transAxes)
        
        ### AX3: MACD
        ax3 = plt.subplot2grid((6,4), (5,0), sharex = ax1, 
                                rowspan=1, colspan=4, axisbg='#07000d')
        macdCol = '#00ffe8'
        nslow = 26
        nfast = 12
        nema = 9
        emaslow, emafast, macd = computeMACD(df.closep)
        ema9 = expMovingAverage(macd, nema)
        ax3.plot(df.date[-SP:], macd[-SP:], color='#86bbec', lw=1.5)
        ax3.plot(df.date[-SP:], ema9[-SP:], color='#e1edf9', lw=1.5)
        ax3.fill_between(df.date[-SP:], macd[-SP:]-ema9[-SP:], 0, alpha=.5, 
                        facecolor=macdCol, edgecolor=macdCol)
        ax3.text(0.015, 0.95, 'MACD 12,26,9', va='top',color='w',transform=ax3.transAxes)

        ax3.spines['top'].set_color('#5998ff')
        ax3.spines['bottom'].set_color('#5998ff')
        ax3.spines['left'].set_color('#5998ff')
        ax3.spines['right'].set_color('#5998ff')
        ax3.tick_params(axis='x', colors='w')
        ax3.tick_params(axis='y', colors='w')
        #plt.ylabel('MACD', color='w')
        plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
        ax3.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))
        for label in ax3.xaxis.get_ticklabels():
            label.set_rotation(30)
        
        
        plt.suptitle(stock, color='w')
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.subplots_adjust(left=.10, bottom=.18, right=.94, top=.95,
                            wspace=.20, hspace=.0)
        plt.show()
        #############################################
        
        if figsave:
            fig.savefig('tempFchart.pdf',facecolor=fig.get_facecolor())