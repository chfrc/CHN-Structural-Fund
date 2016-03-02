#%matplotlib inline
from bs4 import BeautifulSoup
from collections import OrderedDict
import requests
import json
import sys
import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

labels = ['a_id','a_name','a_price','a_rate',
          'a_vol','a_new','a_turnover',
          'b_id','b_name','b_price','b_rate',
          'b_vol','b_new','b_turnover',
          'a_to_b','merge_price','discount_rate',
          'base_id','base_name','base_nav',
          'est_nav','index','index_chg','apply_fee',
          'redeem_fee']
labelsMapping = {'a_id':'fundA_id',
           'a_name':'fundA_nm',
           'a_price':'priceA',
           'a_rate':'increase_rtA',
           'a_vol':'fundA_volume',
           'a_new':'fundA_amount_increase',
           'a_turnover' :'fundA_turnover_rt',
           'b_id':'fundB_id',
           'b_name':'fundB_nm',
           'b_price':'priceB',
           'b_rate':'increase_rtB',
           'b_vol':'fundB_volume',
           'b_new':'fundB_amount_increase',
           'b_turnover' :'fundB_turnover_rt',
           'a_to_b':'abrate',
           'merge_price':'merge_price',
           'discount_rate':'est_dis_rt',
           'base_id':'base_fund_id',
           'base_name':'base_fund_nm',
           'base_nav':'base_nav',
           'est_nav':'base_est_val',
           'index':'index_nm',
           'index_chg':'idx_incr_rt',
           'apply_fee':'apply_fee',
           'redeem_fee':'redeem_fee'
          }

timeStamp = lambda: int(str(int(time.time()))+'000')

###################################################################
###################################################################
###################################################################

class Spider:
    """
    This spider object scrapes quotes panel or historical data
    of structred funds. 

    Methods: 

    * Spider.getPanel(self, clean=1, time=0) gets newest quotes of 
    all funds.
    * Spider.getHistory(self, baseId, time=0) gets historical 
    data of specific fund, by the base fund's ID.

    """
    def __init__(self):
        self.s = requests.session()
        self.urlList = []
        self.header = {
              'Host': 'www.jisilu.cn',
              'Connection': 'keep-alive',
              'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.81 Safari/537.36',
              'Cookie': 'kbz__Session=ghrfl50ar0b06eb2bsqtt2hoi1; kbz_newcookie=1; kbz_r_uname=steliada; kbz__user_login=1ubd08_P1ebax9aX6tbbzdzZxdGCr6blyuzf7tHoxdHVjKmSqNqnotqiq5PZl6vZxanGrNylm9-d2ManltSxk9TGlbSi3uLQ1b-hl6mrk6iCr6bKqtfJoq_l29zkzdGQqaeliaG52MXfz-fn1NiclMLj3d7Yl6_XxJPHtJC5lKqlyKqc1pWfgbTo0dzGy97XtOLgppepmKGrl5CJv8HJtsWYl87fzNiYqNXE3-ieibzU6dHjxqKslJ6RoquonrCar5OWtNTewuLKo66ooKefrQ..; Hm_lvt_164fe01b1433a19b507595a43bf58262=1433506372; Hm_lpvt_164fe01b1433a19b507595a43bf58262={}'.format(timeStamp())
        }
        
    def getPanel(self, clean=1, time=0):
        """
        getPanel gets json data file of newest quotes of funds from web and
        reframes it into pd.dataframe style. if clean flag is set, getPanel
        will read a label list defined in global environment and drop columns
        other than what is in this list.

        args = {
            <boolean> clean: whether or not return only columns defined in global environment.
            <int> time: specifies the timestamp that the spider sends to host, default is 0,
                  (unset), if unset, will generate current timestamp using time.time()
        }
        """
        if not time:
            ts = timeStamp()
        else: ts = time
        url = 'http://www.jisilu.cn/data/sfnew/arbitrage_vip_list/?___t={}'.format(ts)
        parameters = [('___t', ts)]
        self.header = {
              'Host': 'www.jisilu.cn',
              'Connection': 'keep-alive',
              'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.81 Safari/537.36',
              'Cookie': 'kbz_newcookie=1; kbzw__Session=ppg4kj7gfkfbemcvmkqcabea75; kbzw__user_login=7Obd08_P1ebax9aX8cfa2tjl2pmcndHV7Ojg6N7bwNOMqainsZrSx9Wy2Zmy0NuS2JqssNjal6SZrNumza6j3JmXnKTs3Ny_zYylrayisqSYnaO2uNXQo67f293l4cqooaWSlonE2Nbhz-TQ5-GwicLa68figcTY1piww4HMmaaZ2J2owaiKl7jj6M3VuNnbwNLtm6yVrY-qrZOgrLi1wcWhieXV4seWqNza3ueKkKTc6-TW3purmqSRpamorpWekqWvlbza0tjU35CsqqqmlKY.; Hm_lvt_164fe01b1433a19b507595a43bf58262=1456149689,1456150132,1456152058,1456752309; Hm_lpvt_164fe01b1433a19b507595a43bf58262={}'.format(ts)
        }
        try:
            response = requests.get(url, headers=self.header)
            Flist = response.json()['rows']
            df = pd.DataFrame(columns = Flist[0]['cell'].keys())
            for i in range(len(Flist)):
                df.loc[i] = Flist[i]['cell'].values()
            if clean:
                global labels
                global labelsMapping
                df_cleaned = pd.DataFrame(columns=labels)
                for label in labels:
                    df_cleaned[label] = df[labelsMapping[label]]
                df_cleaned.set_index('base_id', inplace=1)
            xlsWriter = pd.ExcelWriter('./tempData.xls', encoding='utf-8')
            df_cleaned.to_excel(xlsWriter) if clean else df.to_excel(xlsWriter)
            xlsWriter.save()
            if clean: return df_cleaned
            else: return df        
        except Exception,e:
            print 'Failed main loop:',e
            
    def getHistory(self, baseId, time=0):
        """
        getHistory gets json data file of historical infomation of selected fund from,
        and reframes it into pd.dataframe style. It uses base fund ID as key.

        args = {
            <str> baseId: the ID of base fund, which specifies data that will be got.
            <int> time: specifies the timestamp that the spider sends to host, default is 0,
                  (unset), if unset, will generate current timestamp using time.time()
        }
        """
        if not time:
            ts = timeStamp()
        else: ts = time
        url = 'http://www.jisilu.cn/jisiludata/StockFenJiDetail.php?qtype=hist&display=table&fund_id={}&___t={}'.format(baseId,ts)
        params = {
            'qtype':'hist',
            'display':'table',
            'fund_id':baseId,
            '___t':ts
        }
        self.header = {
              'Host': 'www.jisilu.cn',
              'Connection': 'keep-alive',
              'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.81 Safari/537.36',
              'Cookie': 'kbz__Session=ghrfl50ar0b06eb2bsqtt2hoi1; kbz_newcookie=1; kbz_r_uname=steliada; kbz__user_login=1ubd08_P1ebax9aX6tbbzdzZxdGCr6blyuzf7tHoxdHVjKmSqNqnotqiq5PZl6vZxanGrNylm9-d2ManltSxk9TGlbSi3uLQ1b-hl6mrk6iCr6bKqtfJoq_l29zkzdGQqaeliaG52MXfz-fn1NiclMLj3d7Yl6_XxJPHtJC5lKqlyKqc1pWfgbTo0dzGy97XtOLgppepmKGrl5CJv8HJtsWYl87fzNiYqNXE3-ieibzU6dHjxqKslJ6RoquonrCar5OWtNTewuLKo66ooKefrQ..; Hm_lvt_164fe01b1433a19b507595a43bf58262=1433506372; Hm_lpvt_164fe01b1433a19b507595a43bf58262={}'.format(ts)
        }
        try:
            response = requests.get(url, headers=self.header, data=params)
            Flist = response.json()['rows']
            df = pd.DataFrame(columns = Flist[0]['cell'].keys())
            for i in range(len(Flist)):
                df.loc[i] = Flist[i]['cell'].values()
                
            df.replace(to_replace='-', value=np.nan, inplace=1)
            df.price_dt = [mdates.datestr2num(df.price_dt[t]) for t in df.index]
            
            xlsWriter = pd.ExcelWriter('./histoData{}.xls'.format(baseId), encoding='utf-8')
            df.to_excel(xlsWriter)
            xlsWriter.save()
            return df
        except Exception,e:
            print 'Failed main loop:',e

    def getETF(self, clean=1, time=0):
        if not time:
            ts = timeStamp()
        else: ts = time
        url = 'http://www.jisilu.cn/jisiludata/etf.php?___t={}'.format(ts)
        parameters = [('___t', ts)]
        self.header = {
            'Host': 'www.jisilu.cn',
            'Connection': 'keep-alive',
            'Cache-Control':'max-age=0',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.124 Safari/537.36',
            'Cookie':'kbz_newcookie=1; kbz__Session=kvt21kbhpurri7628gmkibui07; kbz_r_uname=steliada; kbz__user_login=1ubd08_P1ebax9aX6tbbzdzZxdGCr6blyuzf7tHoxdHVjKmSqNqnotqiq5PZl6vZxanGrNylm9-d2ManltSxk9TGlbSi3uLQ1b-hl6mrk6iCr6bKqtfJoq_l29zkzdGQqaeliaG52MXfz-fn1NiclMLj3d7Yl6_XxJPHtJC5lKqlyKqc1pWfgbTo0dzGy97XtOLgppepmKGrl5CJv8HJtsWYl87fzNiYqNXE3-ieibzU6dHjxqKslJ6RoquonrCaqJSqgcbZx9HT3aiqnLCaqpg.; Hm_lvt_164fe01b1433a19b507595a43bf58262=1433506372,1433745046,1434091595; Hm_lpvt_164fe01b1433a19b507595a43bf58262={}'.format(ts)
        }
        try:
            response = requests.get(url, headers=self.header)
            js = response.json()['rows']
            df = pd.DataFrame(columns = js[0]['cell'].keys())
            for i in range(len(js)):
                df.loc[i] = js[i]['cell'].values()
            xlsWriter = pd.ExcelWriter('./tempData.xls', encoding='utf-8')
            df.to_excel(xlsWriter) if clean else df.to_excel(xlsWriter)
            xlsWriter.save()
            return df
        except Exception,e:
            print 'Failed mainloop:',e


###################################################################
###################################################################
###################################################################
    
def cleanData(data, hist=0):
    """
    This function Cleans data for either history or current panel:
    1) If history data, drops the first(newest transaction day)
    row. Drops 'a_amount' and 'a_amount_increase' where values
    are missing.
    2) Converts str with '%' and '.' to float. For example, it converts
    '2.5%' to 0.025.
    
    args = {
        <pd.DataFrame> data: Dataframe, history or panel
        <boolean> hist: Histroy data Flag
    }
    """
    if hist:
        data.dropna(subset=['a_profit_rt'], inplace=1)
    indices = [i for i in data.index]
    columns = list(data.columns)
    pctColumns = []
    floatColumns = []
    intColumns = []
    for col in columns:
        if '%' in str(data.loc[[indices[0]]][col]) and col!='apply_fee':
            pctColumns.append(col)
        elif '.' in str(data.loc[[indices[0]]][col]):
            floatColumns.append(col)
        elif str(data.loc[[indices[0]]][col]).isdigit():
            intColumns.append(col)
    pctConvert = lambda s: round(float(s.split('%')[0])/100.0, 4)
    floatConvert = lambda s: round(float(s), 4)
    for col in pctColumns:
        data[col].fillna('0%', inplace=1)
        data[col] = map(pctConvert, data[col])
    for col in floatColumns:
        data[col].fillna('0.0', inplace=1)
        data[col] = map(floatConvert, data[col])
    if hist:
        for col in ['a_amount', 'a_amount_increase']:
            data[col].fillna('0.0', inplace=1)
            data[col] = map(floatConvert, data[col])
    return data

###################################################################
###################################################################
###################################################################

class Agent:
    def __init__(self, hist=0):
        self.data = pd.DataFrame()
        self.ts = time.time()
        self.newSession(hist)
 
    def newSession(self, hist=0):
        self.mySpider = Spider()
        if not hist:
            self.data = self.mySpider.getPanel()
            self.data = cleanData(self.data)
        else:
            self.data = self.mySpider.getHistory('161026')
            self.data = cleanData(self.data, hist=1)        
    
    def viewData(self, rows=10, sort='discount_rate', ascending=1):
        return self.data.sort(
            columns=sort,ascending=ascending).head(rows)
    
    def viewFund(self, BaseId):
        keys = list(self.data.loc[[BaseId]].columns)
        vals = self.data.loc[[BaseId]].values.tolist()[0]
        return dict(zip(keys,vals))
    
    def viewHistory(self, BaseId):
        self.data = self.mySpider.getHistory(BaseId)
        self.data = cleanData(self.data, hist=1)
        return self.data
    
    def calcABShares(self, baseId, baseShares=100000):
        dic = self.viewFund(baseId)
        abRatio = [float(j) for j in dic['a_to_b'].split(':')]
        aRatio = abRatio[0]/sum(abRatio)
        bRatio = abRatio[1]/sum(abRatio)
        aShares = int(baseShares*aRatio)
        bShares = int(baseShares*bRatio)
        return aShares, bShares
    
    def calcSplitBase(self, baseId, applySum=500000):
        dic = self.viewFund(baseId)
        baseShares = int(applySum/(dic['base_nav'] * (1+dic['apply_fee'])))
        splitShares = baseShares/10*10
        aShares, bShares = self.calcABShares(baseId, splitShares)
        return {
            'base_id':baseId,
            'base_shares':baseShares,
            'splitable_shares':splitShares,
            'a_shares':aShares,
            'b_shares':bShares,
            'apply_fee':dic['apply_fee'],
            'payment':applySum
        }
    
    def calcDiscAbt(self, baseId, baseShares=100000):
        aShares, bShares = self.calcABShares(baseId, baseShares)
        dic = self.viewFund(baseId)
        buy_fee = 0.0003
        rev = baseShares*dic['est_nav']*(1-dic['redeem_fee'])
        exp = (dic['a_price']*aShares + dic['b_price']*bShares)*(1+buy_fee)
        return {
            'base_shares':baseShares,
            'a_shares':aShares,
            'b_shares':bShares,
            'a_price':dic['a_price'],
            'b_price':dic['b_price'],
            'redeem_fee':dic['redeem_fee'],
            'purchase_fee':buy_fee,
            'expense':round(exp,2),
            'revenue':round(rev,2),
            'current_discount':dic['discount_rate'],
            'est_profit':round(rev-exp,2)
        }
    
    def calcPremAbt(self, baseId, applySum=500000):
        dic_split = self.calcSplitBase(baseId=baseId, applySum=applySum)
        dic = self.viewFund(baseId)
        exp = dic_split['payment']
        rev = dic['a_price']*dic_split['a_shares']+dic['b_price']*dic_split['b_shares']+\
        (dic_split['base_shares']-dic_split['splitable_shares'])*dic['est_nav']*(1-dic['redeem_fee'])
        return {
            'base_shares':dic_split['base_shares'],
            'a_shares':dic_split['a_shares'],
            'b_shares':dic_split['b_shares'],
            'a_price':dic['a_price'],
            'b_price':dic['b_price'],
            'apply_fee':dic_split['apply_fee'],
            'expense':round(exp,2),
            'revenue':round(rev,2),
            'current_discount':dic['discount_rate'],
            'est_profit':round(rev-exp,2)
        }    

###################################################################
###################################################################
###################################################################
    
class Fchart:
    def __init__(self, data):
        self.data = data
    
    def graphData(self, back=20):
        n = min(len(self.data), back)
        df = self.data.head(n)
        
        fig = plt.figure(figsize=(10,10), facecolor='#07000d')
        ax1 = plt.subplot2grid((4,4), (0,0), rowspan=2, colspan=4, axisbg = '#07000d')
        ax1.plot(df.price_dt, df.a_discount_rt, color='#4ee6fd',
                 linewidth=1.5)
        ax1.plot(df.price_dt, df.b_discount_rt, color='#e1edf9',
                 linewidth=1.5)
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
        plt.ylabel('A/B Discount Rate')
        
        ### AX2 a_volume increment
        ax2 = plt.subplot2grid((4,4),(2,0),sharex=ax1,
                                rowspan=1,colspan=4,axisbg = '#07000d')
        volumeMin = 0
        #ax2.grid(True, color='w')
        ax2.plot(df.price_dt, df.a_amount_increase, color='#1a8782', linewidth=1.5)
        ax2.fill_between(df.price_dt,volumeMin,df.a_amount_increase,
                          facecolor='#00ffe8',alpha=.3)
        ax2.spines['top'].set_color('#5998ff')
        ax2.spines['bottom'].set_color('#5998ff')
        ax2.spines['left'].set_color('#5998ff')
        ax2.spines['right'].set_color('#5998ff')
        ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune='lower'))
        ax2.tick_params(axis='x', colors='w')
        ax2.tick_params(axis='y', colors='w')
        ax2.text(0.015, 0.95, 'A vol_incr', va='top',color='w',transform=ax2.transAxes)

        
        ### AX3 base discount
        poiCol = '#386d13'
        negCol = '#8f2020'
        ax3 = plt.subplot2grid((4,4),(3,0),sharex=ax1,
                                rowspan=1,colspan=4,axisbg = '#07000d')
        
        ax3.plot(df.price_dt, df.base_discount_rt, color='#1a8782', linewidth=1.5)
        ax3.axhline(0.02, color=negCol)
        ax3.axhline(-0.02, color=poiCol)
        
        #ax3.grid(True, color='w')
        ax3.spines['top'].set_color('#5998ff')
        ax3.spines['bottom'].set_color('#5998ff')
        ax3.spines['left'].set_color('#5998ff')
        ax3.spines['right'].set_color('#5998ff')
        ax3.tick_params(axis='x', colors='w')
        ax3.tick_params(axis='y', colors='w')
        ax3.yaxis.label.set_color('w')
        ax3.set_yticks([-0.02,0.02])
        for label in ax3.xaxis.get_ticklabels():
            label.set_rotation(30)
        ax3.text(0.015, 0.95, 'Base Disc', va='top',color='w',transform=ax3.transAxes)
        
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.subplots_adjust(left=.10, bottom=.18, right=.94, top=.95,
                            wspace=.20, hspace=.0)
        fig.savefig('tempFchart.pdf',facecolor=fig.get_facecolor())
        plt.suptitle('BaseFund '+df['base_fund_id'][1], color='w')
        plt.show()

###################################################################
###################################################################
###################################################################

def test1():
    ag = Agent(hist=0)
    print ag.viewData()

def test2():
    ag = Agent(hist=0)
    print ag.calcPremAbt(baseId='163113')

def test3():
    ag = Agent(hist=0)
    df = ag.viewHistory('502030')
    myplot = Fchart(df)
    myplot.graphData(back=100)

def test4():
    help(cleanData)

def test5():
    mySp = Spider()
    df = mySp.getETF()
    print df

test3()
