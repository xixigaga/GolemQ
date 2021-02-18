import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from analysis.machinelearning import *
from analysis.emd import *
import QUANTAXIS as QA
from QUANTAXIS.QAIndicator.base import *

import pandas as pd
import numpy as np
import numba as nb
# import peakutils
# import scipy.signal as signal
# from scipy.signal import lfilter, lfilter_zi, filtfilt, butter, savgol_filter
import matplotlib.pyplot as plt
import QUANTAXIS as QA
from QUANTAXIS.QAIndicator.base import *
from QUANTAXIS import CROSS,find_peak_vextors,Timeline_Integral_with_cross_before,find_peak_vextors_eagerly,CROSS_STATUS,Timeline_Integral
import mplfinance
from scipy.signal import find_peaks
import talib as ta


def QA_indicator_MACD(DataFrame, short=12, long=26, mid=9):
    """
    MACD CALC
    """
    CLOSE = DataFrame['close']

    DIF = EMA(CLOSE, short)-EMA(CLOSE, long)
    DIF_nor = (EMA(CLOSE, short)-EMA(CLOSE, long))/EMA(CLOSE, long)
    DEA = EMA(DIF, mid)
    DEA_nor = EMA(DIF_nor, mid)
    MACD = (DIF-DEA)*2
    MACD_nor = (DIF_nor-DEA_nor)*2

    return pd.DataFrame({'DIF': DIF, 'DEA': DEA, 'MACD': MACD,
                        'DIF_nor': DIF_nor, 'DEA_nor': DEA_nor, 'MACD_nor': MACD_nor})

class DataFinanceDraw(object):
    """
    获取数据，并按照 mplfinanace 需求的格式格式化，然后绘图
    """
    def __init__(self,data):
        self.data = data.droplevel(level=1)
        self.data=self.data[-300:]
    def add_p_plot(self,val,typename='atrsuper'):
        val=val[-300:]
        if typename=='atrsuper':
            up=np.full([len(val[1])], np.nan)
            down=np.full([len(val[1])], np.nan)
            for i,a in enumerate(val[1]):
                if a>0:
                    up[i]=val[0][i]
                else:
                    down[i]=val[0][i]
            self.add_plot = [
                mplfinance.make_addplot(val[1],panel=2, color='y'),
                mplfinance.make_addplot(up, panel=1, color='r',secondary_y=False),
                mplfinance.make_addplot(down, panel=1, color='g',secondary_y=False),
            ]
        elif typename=='vhma':
            up=val[0]
            down=val[1]
            self.add_plot = [
            mplfinance.make_addplot(up, panel=1, color='r',secondary_y=False),
            mplfinance.make_addplot(down, panel=2, color='g',secondary_y=False),
            # mplfinance.make_addplot(hclose, panel=1, color='b',secondary_y=False),
            ]
        elif typename=='macd':
            self.add_plot = [
            mplfinance.make_addplot(val['DIF'], panel=2, color='g',secondary_y=False),
            mplfinance.make_addplot(val['DEA'], panel=2, color='b',secondary_y=False),
            mplfinance.make_addplot(val['MACD'], panel=2, color='r',type='bar',secondary_y=False),
            # mplfinance.make_addplot(val['DIF_nor'], panel=3, color='r',secondary_y=False), #nor标准化数据过程
            # mplfinance.make_addplot(val['DEA_nor'], panel=3, color='g',secondary_y=False),
            # mplfinance.make_addplot(val['MACD_nor'], panel=3, color='b',type='bar',secondary_y=False),
            #nor各类金叉死叉
            mplfinance.make_addplot(val['DEA_CROSS'], panel=3, color='b',secondary_y=False),
            mplfinance.make_addplot(val['DIF_CROSS'], panel=3, color='g',secondary_y=False),
            mplfinance.make_addplot(val['DIF_CROSS_MAX'], panel=3, type='bar',color='r',secondary_y=False),
            mplfinance.make_addplot(val['DIF_PEAK_BF'], panel=4, color='r',secondary_y=False),
            # mplfinance.make_addplot(val['DIF_PEAK_BF_c'], panel=4, color='b',secondary_y=True),
            mplfinance.make_addplot(val['ZLMACD'], panel=5, color='r',secondary_y=False),
            mplfinance.make_addplot(val['ZLMACDS'], panel=5, color='b',secondary_y=True),

            mplfinance.make_addplot(val['mltrend'], panel=6, color='r',secondary_y=False),

            ]
        elif typename=='Schaf':
            self.add_plot = [
            mplfinance.make_addplot(val['sd'], panel=2, color='g',secondary_y=False),
            mplfinance.make_addplot(val['sd_short'], panel=2,type='bar', color='g',secondary_y=True),
            # mplfinance.make_addplot(val['sd_long'], panel=2, color='g',secondary_y=True),
            ]
        elif typename=='fea':
            self.add_plot = [
                # BOLL_UB\BOLL_LB\BOLL_WIDTH\MA_VOL
                # ATR_UB\ATR_LB\ZSCORE_21\\\\\
            mplfinance.make_addplot(val['BOLL_UB'], panel=1,color='b', secondary_y=False),
            mplfinance.make_addplot(val['BOLL_LB'], panel=1,color='b',secondary_y=False),
            mplfinance.make_addplot(val['ATR_UB'], panel=1, color='g',secondary_y=False),
            mplfinance.make_addplot(val['ATR_LB'], panel=1, color='g',secondary_y=False),
            mplfinance.make_addplot(val['ZSCORE_21'], panel=2, color='b',secondary_y=False),
            mplfinance.make_addplot(val['BOLL_WIDTH'], panel=3,color='g',secondary_y=False),
            # mplfinance.make_addplot(val['sd_long'], panel=2, color='g',secondary_y=True),
            ]
    def panel_draw(self):
        """
        make_addplot 绘制多个子图
        """
        data = self.data
        mplfinance.plot(data, type='candle',main_panel=1,volume_panel=0,
                        # mav=(5, 10),
                        addplot=self.add_plot,
                        volume=True,
                        figscale=1.5,
                        xrotation=15,
                        title='Candle',  ylabel='price', #ylabel_lower='volume',
                        )
        plt.show()  # 显示
        plt.close()  # 关闭plt，释放内存

def CROSS_STATUS_both(A, B):
    """
    A 穿过 B 产生+1 -1 序列信号
    """
    return np.where(A > B, 1, -1)

def ind_macd(data):
    """
    计算MACD相关指标
    """
    MACD = QA.TA_MACD(data.close)

    PRICE_PREDICT = pd.DataFrame(columns=['DIF_CROSS_MAX', 'MACD_CROSS_JX', 'MACD_CROSS_SX','DIF_PEAK_BF'], index=data.index)
    PRICE_PREDICT = PRICE_PREDICT.assign(DIF=MACD[:,0])
    PRICE_PREDICT = PRICE_PREDICT.assign(DEA=MACD[:,1])
    PRICE_PREDICT = PRICE_PREDICT.assign(MACD=MACD[:,2])
    PRICE_PREDICT = PRICE_PREDICT.assign(DELTA=MACD[:,3])
    # PRICE_PREDICT = PRICE_PREDICT.assign(DIF_nor=MACD[:,4]) #标准化数据
    # PRICE_PREDICT = PRICE_PREDICT.assign(DEA_nor=MACD[:,5])
    # PRICE_PREDICT = PRICE_PREDICT.assign(MACD_nor=MACD[:,6])
    # PRICE_PREDICT = PRICE_PREDICT.assign(DELTA_nor=MACD[:,7])

    # macd_cross_np(PRICE_PREDICT['DIF'].values,PRICE_PREDICT['DEA'].values,PRICE_PREDICT['MACD'].values,)

    dif_tp_max, _ = find_peaks(PRICE_PREDICT['DIF'].values, width=4)
    dif_tp_min, _ = find_peaks(-PRICE_PREDICT['DIF'].values, width=4)
    PRICE_PREDICT.iloc[dif_tp_max, PRICE_PREDICT.columns.get_loc('DIF_CROSS_MAX')] = 1 
    PRICE_PREDICT.iloc[dif_tp_min, PRICE_PREDICT.columns.get_loc('DIF_CROSS_MAX')] = -1
    PRICE_PREDICT['DIF_SIDE'] =CROSS_STATUS_both(PRICE_PREDICT['DIF'], 0)
    PRICE_PREDICT['DIF_CROSS_MAX']= PRICE_PREDICT['DIF_SIDE']+PRICE_PREDICT['DIF_CROSS_MAX'] #peak标志位
    
    PRICE_PREDICT['DIF_PEAK_BF'] =PRICE_PREDICT['DIF'][dif_tp_max]/PRICE_PREDICT['DIF'][dif_tp_max].shift(1)#与前一极值比值
    PRICE_PREDICT['DIF_PEAK_BF'] =PRICE_PREDICT['DIF_PEAK_BF'].fillna(method='ffill')

    #极值点close比值
    PRICE_PREDICT['HIGHER_SETTLE_PRICE'] =data.close[dif_tp_max]
    PRICE_PREDICT['HIGHER_SETTLE_PRICE_BEFORE'] =data.close[dif_tp_max]/data.close[dif_tp_max].shift(1)
    PRICE_PREDICT['HIGHER_SETTLE_PRICE_BEFORE'] =PRICE_PREDICT['HIGHER_SETTLE_PRICE_BEFORE'].fillna(method='ffill')
    PRICE_PREDICT['NEGATIVE_LOWER_PRICE'] =data.close[dif_tp_min]
    PRICE_PREDICT['NEGATIVE_LOWER_PRICE_BEFORE'] =data.close[dif_tp_min]/data.close[dif_tp_min].shift(1)
    PRICE_PREDICT['NEGATIVE_LOWER_PRICE_BEFORE'] =PRICE_PREDICT['NEGATIVE_LOWER_PRICE_BEFORE'].fillna(method='ffill')

    # PRICE_PREDICT['DIF_PEAK_BF'] =PRICE_PREDICT['DIF_PEAK_BF']/PRICE_PREDICT['DIF_PEAK_BF'].shift(1)#求比值
    # PRICE_PREDICT['DIF_PEAK_BF'][PRICE_PREDICT['DIF_PEAK_BF']==1]=np.nan
    # PRICE_PREDICT['DIF_PEAK_BF'] =PRICE_PREDICT['DIF_PEAK_BF'].fillna(method='ffill')#再填充
    # print(PRICE_PREDICT['DIF'][dif_tp_max]/PRICE_PREDICT['DIF'][dif_tp_max].shift(1))

    PRICE_PREDICT['DEA_CROSS_JX'] = CROSS(PRICE_PREDICT['DEA'], 0)
    PRICE_PREDICT['DEA_CROSS_SX'] = CROSS(0, PRICE_PREDICT['DEA'])
    PRICE_PREDICT['DEA_CROSS']=PRICE_PREDICT.apply(lambda x: 1 if (x['DEA_CROSS_JX'] == 1) else (-1 if (x['DEA_CROSS_SX'] == 1) else 0), axis=1) 
    # ta.LINEARREG(close, timeperiod=14)
    # LINEARREG_SLOPE : Linear Regression Slope 线性回归斜率：
    # LINEARREG_INTERCEPT : Linear Regression Angle 线性回归截距:
    PRICE_PREDICT['DEA_SLOPE']=ta.LINEARREG_SLOPE(PRICE_PREDICT['DEA'],5)
    PRICE_PREDICT['DEA_INTERCEPT']=ta.LINEARREG_INTERCEPT(PRICE_PREDICT['DEA'],5)
    PRICE_PREDICT['DEA_SLOPE_TIMING_LAG'] = Timeline_Integral_with_cross_before(np.nan_to_num(PRICE_PREDICT['DEA_SLOPE'].values.tolist()))
    PRICE_PREDICT['DEA_SLOPE_CHANGE']=PRICE_PREDICT['DEA_SLOPE'].diff(1)
    PRICE_PREDICT['DEA_INTERCEPT_TIMING_LAG']= Timeline_Integral_with_cross_before(np.nan_to_num(PRICE_PREDICT['DEA_INTERCEPT'].values.tolist()))

    PRICE_PREDICT['DEA_CROSS_JX_BF'] = Timeline_Integral(np.nan_to_num(PRICE_PREDICT['DEA_CROSS_JX'].values.tolist()))
    PRICE_PREDICT['DEA_CROSS_SX_BF'] = Timeline_Integral(np.nan_to_num(PRICE_PREDICT['DEA_CROSS_SX'].values.tolist()))
    PRICE_PREDICT['DEA_ZERO_TIMING_LAG'] = Timeline_Integral_with_cross_before(np.nan_to_num(PRICE_PREDICT['DEA_CROSS'].values.tolist()))

    PRICE_PREDICT['DIF_CROSS_JX'] = CROSS(PRICE_PREDICT['DIF'], 0)
    PRICE_PREDICT['DIF_CROSS_SX'] = CROSS(0, PRICE_PREDICT['DIF'])
    PRICE_PREDICT['DIF_CROSS_JX_BF'] = Timeline_Integral(np.nan_to_num(PRICE_PREDICT['DIF_CROSS_JX'].values.tolist()))
    PRICE_PREDICT['DIF_CROSS_SX_BF'] = Timeline_Integral(np.nan_to_num(PRICE_PREDICT['DIF_CROSS_SX'].values.tolist()))
    PRICE_PREDICT['DIF_CROSS']=PRICE_PREDICT.apply(lambda x: 1 if (x['DIF_CROSS_JX'] == 1) else (-1 if (x['DIF_CROSS_SX'] == 1) else 0), axis=1) 

    PRICE_PREDICT['MACD_CROSS_JX'] = CROSS(PRICE_PREDICT['DIF'], PRICE_PREDICT['DEA'])
    PRICE_PREDICT['MACD_CROSS_SX'] = CROSS(PRICE_PREDICT['DEA'], PRICE_PREDICT['DIF'])
    PRICE_PREDICT['MACD_ZERO'] = CROSS(PRICE_PREDICT['MACD'], 0) +CROSS(0, PRICE_PREDICT['MACD'])
    PRICE_PREDICT['MACD_ZERO_TIMING_LAG']=Timeline_Integral_with_cross_before(np.nan_to_num(PRICE_PREDICT['MACD_CROSS_JX'].values.tolist()))


    PRICE_PREDICT['MACD_CROSS_JX_BF']=Timeline_Integral(np.nan_to_num(PRICE_PREDICT['MACD_CROSS_JX'].values.tolist()))
    PRICE_PREDICT['MACD_CROSS_SX_BF']=Timeline_Integral(np.nan_to_num(PRICE_PREDICT['MACD_CROSS_SX'].values.tolist()))
    PRICE_PREDICT['mltrend']=dpgmm_predict(data, '000001')
    SN=12
    LP=26
    M=9
    close=data.close
    PRICE_PREDICT['ZLMACD']=(2*EMA(close,SN)-EMA(EMA(close,SN),SN))-(2*EMA(close,LP)-EMA(EMA(close,LP),LP))
    PRICE_PREDICT['ZLMACDS']=2*EMA(PRICE_PREDICT['ZLMACD'],M)-EMA(EMA(PRICE_PREDICT['ZLMACD'],M),M)

    # close=data.close
    # PRICE_PREDICT['ZLMACD']=(2*EMA(close,SN)-EMA(EMA(close,SN),SN))-(2*EMA(close,LP)-EMA(EMA(close,LP),LP))
    # PRICE_PREDICT['ZLMACDS']=2*EMA(PRICE_PREDICT['ZLMACD'],M)-EMA(EMA(PRICE_PREDICT['ZLMACD'],M),M)

    if (len(PRICE_PREDICT.index.names) > 2):
        return PRICE_PREDICT.reset_index([1,2])
    elif (len(PRICE_PREDICT.index.names) > 1):
        return PRICE_PREDICT.reset_index([1])
    else:
        return PRICE_PREDICT


def ind_price(data):
    indFrame = pd.DataFrame(columns=['MA5'], index=data.index)
    indFrame['PCT_CHANGE']=data.close.pct_change(5)
    indFrame['PCT_CHANGE5']=data.close.pct_change(20)
    indFrame['MA5']=QA.TA_HMA(data.close,5) #注意此处的类型！
    indFrame['MA10']=ta.MA(data.close,10)
    indFrame['MA20']=ta.MA(data.close,20)
    indFrame['MA30']=ta.MA(data.close,30)
    indFrame['MA30_CROSS_JX']=CROSS(indFrame['MA5'], indFrame['MA30'])
    indFrame['MA30_CROSS_SX']=CROSS(indFrame['MA30'], indFrame['MA5'])
    indFrame['MA30_CROSS_JX_BEFORE']=Timeline_Integral(np.nan_to_num(indFrame['MA30_CROSS_JX'].values.tolist()))
    indFrame['MA30_CROSS_SX_BEFORE']=Timeline_Integral(np.nan_to_num(indFrame['MA30_CROSS_SX'].values.tolist()))
    indFrame['MA30_CROSS'] = 1 if (indFrame['MA30_CROSS_JX'] == 1) else (-1 if (indFrame['MA30_CROSS_SX'] == 1) else 0)
    indFrame['MA30_SLOPE']=ta.LINEARREG_SLOPE(indFrame['MA30'],5)
    indFrame['MA30_SLOPE_CHANGE']=indFrame['MA30_SLOPE'].diff()
    indFrame['MA60']=ta.MA(data.close,60)
    indFrame['MA60_SLOPE']=ta.LINEARREG_SLOPE(indFrame['MA60'],5)

    indFrame['MA90']=ta.MA(data.close,90)
    indFrame['MA90_CROSS_JX']=CROSS(indFrame['MA60'], indFrame['MA90'])
    indFrame['MA90_CROSS_SX']=CROSS(indFrame['MA90'], indFrame['MA60'])
    indFrame['MA90_CROSS_JX_BEFORE']=Timeline_Integral(np.nan_to_num(indFrame['MA90_CROSS_JX'].values.tolist()))
    indFrame['MA90_CROSS_SX_BEFORE']=Timeline_Integral(np.nan_to_num(indFrame['MA90_CROSS_SX'].values.tolist()))
    indFrame['MA90_CROSS']=1 if (indFrame['MA90_CROSS_JX'] == 1) else (-1 if (indFrame['MA90_CROSS_SX'] == 1) else 0)
    indFrame['MA90_SLOPE']=ta.LINEARREG_SLOPE(indFrame['MA90'],5)
    indFrame['MA90_RETURNS']=np.log(indFrame['MA90']/indFrame['MA90'].shift(1))

    indFrame['MA120']=ta.MA(data.close,120)
    indFrame['MA120_SLOPE']=ta.LINEARREG_SLOPE(indFrame['MA120'],5)
    indFrame['MA120_RETURNS']=np.log(indFrame['MA120']/indFrame['MA120'].shift(1))

    indFrame['MAPOWER30']=ma_power_np_func2(data.close,30)
    indFrame['MAPOWER30_RETURNS']=np.log(indFrame['MAPOWER30']/indFrame['MAPOWER30'].shift(1))
    indFrame['MAPOWER30_TIMING_LAG']=Timeline_Integral_with_cross_before(np.nan_to_num([PRICE_PREDICT['MAPOWER30']-0.5].values.tolist()))
    # indFrame['MAPOWER30_SORTINO']=ma_power_np_func2(data.close,120)
    indFrame['MAPOWER120']=ma_power_np_func2(data.close,120)
    indFrame['MAPOWER120_TIMING_LAG']=Timeline_Integral_with_cross_before(np.nan_to_num([PRICE_PREDICT['MAPOWER30']-0.5].values.tolist()))

    indFrame['HMAPOWER30']=ma_power_np_func2(indFrame['MA5'],30)
    indFrame['HMAPOWER120']=ma_power_np_func2(indFrame['MA5'],120)
    ###########VOL
    indFrame['VOL_MA5']=ta.MA(data.volume,5)
    indFrame['VOL_MA10']=ta.MA(data.volume,10)
    indFrame['VOL_CROSS_JX']=CROSS(indFrame['VOL_MA5'], indFrame['VOL_MA10'])
    indFrame['VOL_CROSS_SX']=CROSS(indFrame['VOL_MA10'], indFrame['VOL_MA5'])
    indFrame['VOL_CROSS'] = 1 if (indFrame['VOL_CROSS_JX'] == 1) else (-1 if (indFrame['VOL_CROSS_SX'] == 1) else 0)
    indFrame['VOL_TREND']=Trend_fun(indFrame['VOL_MA5'])
    indFrame['VOL_TREND_TIMING_LAG']=Timeline_Integral_with_cross_before(np.nan_to_num(PRICE_PREDICT['VOL_TREND'].values.tolist()))

    ############RSI
    indFrame['RSI'],indFrame['RSI_DELTA']=QA.TA_RSI(data.volume,12)
    indFrame['RSI_NORM']=indFrame['RSI']/100
    indFrame['RSI_TREND']=Trend_fun(indFrame['RSI_NORM'])
    indFrame['RSI_CROSS_JX']=CROSS(indFrame['RSI'], 25)
    indFrame['RSI_CROSS_SX']=CROSS(75, indFrame['RSI'])
    indFrame['RSI_CROSS'] = 1 if (indFrame['VOL_CROSS_JX'] == 1) else (-1 if (indFrame['VOL_CROSS_SX'] == 1) else 0)
    indFrame['RSI_TREND_TIMING_LAG']=Timeline_Integral_with_cross_before(np.nan_to_num(PRICE_PREDICT['RSI_TREND'].values.tolist()))

    return indFrame

def zeroMACD(data):
    ''' Zero Lag MACD
    上穿０轴或是信号线，买入；下穿０轴或是信号线，卖出
    ZLMACD=(2*EMA(C,SN)-EMA(EMA(C,SN),SN))-(2*EMA(C,LP)-EMA(EMA(C,LP),LP))
    ZLMACDS=2*EMA(ZLMACD,M)-EMA(EMA(ZLMACD,M),M)

    EMA是指数移动平均；
    SN  12；
    LP  26；
    M 9；
    ZLMACDS是信号线。
    '''
    indFrame = pd.DataFrame(columns=['ZLMACD'], index=data.index)
    SN=12
    LP=26
    M=9
    close=data.close
    indFrame['ZLMACD']=(2*EMA(close,SN)-EMA(EMA(close,SN),SN))-(2*EMA(close,LP)-EMA(EMA(close,LP),LP))
    indFrame['ZLMACDS']=2*EMA(indFrame['ZLMACD'],M)-EMA(EMA(indFrame['ZLMACD'],M),M)
    indFrame['ZLMACDS_hist']=2*(indFrame['ZLMACD']-indFrame['ZLMACDS'])
    return indFrame

def SchaffTrendCycle(data):
    '''
    macdx=ema(c,n1)-ema(c,n2)
    v1=llv(macdx,n)
    v2=hhv(macdx,n)-llv(macdx,n)
    fk=iff(v2>0,(macdx-v1)/v2*100,ref(fk,1))
    fd=sma(fk,n,1)#无factor
    #有factor
    # fd=IFF(bar<=1,fk,fd(1)+factor*(fk-fd(1)))
    v3=llv(fd,n)
    v4=HHV(fd,n)-llv(fd,n)
    sk=iff(v4>0,(fd-v3)/v4*100,ref(sk,1))
    sd=sma(sk,n,1)

    '''
    indFrame = pd.DataFrame(columns=['sd_short','sd_long'], index=data.index)
    fast=23
    slow=50
    cycle=20
    factor=0.5
    close=data.close
    indFrame['macdx']=EMA(close,fast)-EMA(close,slow)
    indFrame['v1']=LLV(indFrame['macdx'],cycle)
    indFrame['v2']=HHV(indFrame['macdx'],cycle)-indFrame['v1']
    indFrame['fk']=indFrame.apply(lambda x:(x['macdx']-x['v1'])/x['v2']*100 if x['v2']>0 else 0,axis=1)
    indFrame['fk_1']=indFrame['fk'].shift(1)
    indFrame['fk']=indFrame.apply(lambda x:(x['macdx']-x['v1'])/x['v2']*100 if x['v2']>0 else x['fk_1'],axis=1)
    # indFrame['fk2']=indFrame.apply(lambda x: x['fk_1']+factor*(x[]),axis=1)
    # fd=IFF(bar<=1,fk,fd(1)+factor*(fk-fd(1)))
    indFrame['fd']=EMA(indFrame['fk'],cycle)
    indFrame['v3']=LLV(indFrame['fd'],cycle)
    indFrame['v4']=HHV(indFrame['fd'],cycle)-indFrame['v3']
    
    indFrame['sk']=indFrame.apply(lambda x:(x['fd']-x['v1'])/x['v2']*100 if x['v2']>0 else 0,axis=1)
    indFrame['sk_1']=indFrame['sk'].shift(1)
    indFrame['sk']=indFrame.apply(lambda x:(x['fd']-x['v3'])/x['v4']*100 if x['v4']>0 else x['sk_1'],axis=1)
    indFrame['sd']=EMA(indFrame['sk'],cycle)
    dif_tp_max, _ = find_peaks(indFrame['sd'].values)
    dif_tp_min, _ = find_peaks(-indFrame['sd'].values)
    indFrame.iloc[dif_tp_max, indFrame.columns.get_loc('sd_short')] = -1 
    indFrame.iloc[dif_tp_min, indFrame.columns.get_loc('sd_short')] = 1
    # print(indFrame[indFrame.sd_short.isin([1,-1])])
    print(indFrame.sd_short.values)
    return indFrame

def schaff_np(closep:np.ndarray,) -> np.ndarray:
    """
    自创指标：MAXFACTOR
    """
    fast=23
    slow=50
    cycle=20
    factor=0.5
    close=closep
    macdx=talib.EMA(close,fast)-talib.EMA(close,slow)
    
    v1=LLV(macdx,cycle)
    v2=HHV(macdx,cycle)-v1
    fk = np.where(v2>0, (macdx-v1)*100/v2, 0)
    fk_1=np.r_[0,fk[:-1]]
    fk = np.where(v2>0, fk, fk_1)
    # fk = np.where(v2>0, (macdx-v1)*100/v2, fk_1) #test
    # fd=IFF(bar<=1,fk,fd(1)+factor*(fk-fd(1)))
    fd=talib.EMA(fk,cycle)
    v3=LLV(fd,cycle)
    v4=HHV(fd,cycle)-v3
    sk = np.where(v2>0, (fd-v1)*100/v2, 0)
    # sk=indFrame.apply(lambda x:(x['fd']-x['v1'])/x['v2']*100 if x['v2']>0 else 0,axis=1)
    sk_1=np.r_[0,sk[:-1]]
    sk = np.where(v4>0, (fd-v3)*100/v4, sk_1)
    # sk=indFrame.apply(lambda x:(x['fd']-x['v3'])/x['v4']*100 if x['v4']>0 else x['sk_1'],axis=1)
    sd=talib.EMA(sk,cycle)
    sd_cross_dif = np.nan_to_num(np.r_[0, np.diff(sd)], nan=0)
    sd_cross_dif_1=np.r_[0,sd_cross_dif[:-1]]
    mft_cross = np.where((sd_cross_dif < 0) & (sd_cross_dif_1 > 0), 1,\
                        (np.where((sd_cross_dif > 0) & (sd_cross_dif_1 < 0), -1,0)))
    print(mft_cross)


if __name__ == "__main__":
    data = QA.QA_fetch_stock_day_adv(
    ['600519'], '2016-09-01', '2018-04-23')

    data = data.to_qfq()
    schaff_np(data.data.close)
    # macd=ind_macd(data.data)
    Schaf=SchaffTrendCycle(data.data)
    # fea=ml_trend_func(data.data)
    # imfs, imfNo=calc_eemd_func(fea)
    # fea=calc_best_imf_periods(fea,imfs,imfNo)

    # macd=QA_indicator_MACD(data.data)
    # atrsuper=QA.ATR_SuperTrend_cross(data.data) 
    # adx=QA.ADX_MA(data.data)
    # vhma=QA.Volume_HMA(data.data)
    # hclose=QA.TA_HMA(data.data.close,5)
    
    candle = DataFinanceDraw(data.data)
    typename='Schaf'
    switch={
        'atrsuper':lambda x:candle.add_p_plot(atrsuper,x),
        'vhma':lambda x:candle.add_p_plot(vhma,x),
        'macd':lambda x:candle.add_p_plot(macd,x),
        'Schaf':lambda x:candle.add_p_plot(Schaf,x),
        'fea':lambda x:candle.add_p_plot(fea,x),
    }

    switch[typename](typename)
    candle.panel_draw()