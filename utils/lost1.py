#
# The MIT License (MIT)
#
# Copyright (c) 2018-2020 azai/Rgveda/GolemQuant
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import numpy as np
import numba as nb
from numba import vectorize, float64, jit as njit
import scipy.stats as scs
from datetime import datetime as dt, timezone, timedelta

from GolemQ.analysis.timeseries import *
from QUANTAXIS.QAUtil.QADate_Adv import (QA_util_print_timestamp,)
import pandas as pd
import empyrical
from GolemQ.utils.parameter import (
    AKA, 
    INDICATOR_FIELD as FLD, 
    TREND_STATUS as ST,
    FEATURES as FTR,
    )

"""
这里定义的是一些 fractal, strategy, portfolio 的绩效统计相关工具函数
"""

def calc_fractal_stats(symbol, display_name, fractal_uid, fractal_triggers,
                  ref_features=None, rsk_fre=0.04, annual=252, taxfee=0.0003, 
                  long=1, mode='lrc', format='pd'):
    """
    explanation:
        分型触发点(卖点or买点，取决于交易方向)绩效分析

        什么是分型触发点呢？对大众认知的交易里，最通俗的含义就是特定交易策略的金叉/死叉点。
        例如MACD金叉/死叉点，双MA金叉、死叉点，KDJ金叉、死叉点等。
        它们全部都可以纳入这套 fractal_stats 评估体系中衡量。

        它的功能用法有点像 alphalens，但是 alphalens 使用的框架体系是传统经济学量化金融
        的那一套理论，跟我朴素的衡量标准并不一致。比如 alpha_beta 只计算当前 bar。
        在我的交易系统内，并没有发现我认知的“趋势买点”和传统金融量化理论计算的 IC alpha值有
        什么显著的相关性特征。
        所以我只能另起炉灶，用统计学习的方式来量化衡量一个分型触发点的优劣。
        这套分型系统的数学涵义上更类似深度学习中Attention模型的注意力关注点，能否与机器学习
        结合有待后续研究。

    params:
        symbol: str, 交易标的代码
        display_name: str, 交易标的显示名称
        fractal_uid: str, 分型唯一标识编码
        fractal_triggers : np.array, 分型触发信号
        ref_features: np.array, 参考指标特征
        rsk_fre: float32, 无风险利率
        annual: int32, 年化周期
        taxfee: float32, 税费
        long: int32, 交易方向
        mode: str, 'lrc' or 'zen' 趋势判断模式为 zen趋势 或者 lrc回归线趋势，
                   'hmapower'为追踪hmapower120的MA逆序趋势，
                   'raw'不进行趋势判断，完全跟随 fractal_triggers 状态信号
        format : string, 返回格式

    return:
        pd.Series or np.array or string
    """
    # 这里严格定义应该是考虑交易方向，但是暂时先偷懒简化了计算，以后做双向策略出现问题再完善
    if (long > 0):
        # 做多方向
        fractal_cross_before = Timeline_duration(np.where(fractal_triggers > 0, 1, 0))
    else:
        # 做空方向
        fractal_cross_before = Timeline_duration(np.where(fractal_triggers < 0, 1, 0))

    if (annual > 125) and (annual < 366):
        # 推断为日线级别的数据周期
        fractal_forcast_position = np.where(fractal_cross_before < 3, 1, 0)
        fractal_limited = 3
    elif ((annual > 1680) and (annual < 2560)):
        # 推断为数字币 4小时先级别的数据周期
        fractal_limited = 24
        fractal_forcast_position = np.where(fractal_cross_before < 24, 1, 0)
    elif ((annual > 512) and (annual < 1280)):
        # 推断为股票/证券1小时先级别的数据周期
        fractal_limited = 12
        fractal_forcast_position = np.where(fractal_cross_before < 12, 1, 0)
    elif ((annual > 6180) and (annual < 9600)):
        # 推断为股票/证券1小时先级别的数据周期
        fractal_limited = 72
        fractal_forcast_position = np.where(fractal_cross_before < 72, 1, 0)

    # 固定统计3交易日内收益
    fractal_forcast_3d_lag = calc_event_timing_lag(np.where(fractal_forcast_position > 0, 1, -1))
    fractal_forcast_3d_lag = np.where(fractal_forcast_3d_lag <= fractal_limited, fractal_forcast_3d_lag, 0)
    closep = ref_features[AKA.CLOSE].values

    if (mode == 'lrc'):
        # 统计到下一次 lineareg_band / 死叉等对等交易信号结束时的 收益，时间长度不固定
        if (long > 0):
            # 做多方向
            lineareg_endpoint_before = Timeline_duration(np.where(ref_features[FLD.LINEAREG_BAND_TIMING_LAG] == -1, 1, 0))
        else:
            # 做空方向
            lineareg_endpoint_before = Timeline_duration(np.where(ref_features[FLD.LINEAREG_BAND_TIMING_LAG] == 1, 1, 0))
        fractal_lineareg_position = np.where(fractal_cross_before < lineareg_endpoint_before, 1, 0)
        fractal_lineareg_lag = calc_event_timing_lag(np.where(fractal_lineareg_position > 0, 1, -1))

        transcation_stats = calc_transcation_stats(fractal_triggers,
                                                   closep,
                                                   fractal_forcast_3d_lag,
                                                   fractal_lineareg_lag,
                                                   ref_features[FLD.LINEAREG_BAND_TIMING_LAG].values,
                                                   taxfee=taxfee,
                                                   long=long)

        transcation_stats_df = pd.DataFrame(transcation_stats, columns=['trans_stats',
                                                                        'trans_start',
                                                                        'trans_act',
                                                                        'trans_end', 
                                                                        'start_principle', 
                                                                        'ret_3d',
                                                                        'ret_fractal',
                                                                        'pric_settle',
                                                                        'trans_3d',
                                                                        'price_end_3d',
                                                                        'price_end_fractal',
                                                                        'ret_fractal_sim',
                                                                        'long',
                                                                        'duration_time',])
    elif (mode == 'zen') or \
        (mode == 'mapower'):
        if (long > 0):
            # 做多方向
            zen_wavelet_endpoint_before = Timeline_duration(np.where(ref_features[FLD.ZEN_WAVELET_TIMING_LAG] == -1, 1, 0))
        else:
            # 做空方向
            zen_wavelet_endpoint_before = Timeline_duration(np.where(ref_features[FLD.ZEN_WAVELET_TIMING_LAG] == 1, 1, 0))
        fractal_zen_wavelet_position = np.where(fractal_cross_before < zen_wavelet_endpoint_before, 1, 0)
        fractal_zen_wavelet_lag = calc_event_timing_lag(np.where(fractal_zen_wavelet_position > 0, 1, -1))
        transcation_stats = calc_transcation_stats_np(fractal_triggers,
                                                   closep,
                                                   fractal_forcast_3d_lag,
                                                   fractal_zen_wavelet_lag,
                                                   taxfee=taxfee,
                                                   long=long)

        transcation_stats_df = pd.DataFrame(transcation_stats, columns=['trans_stats',
                                                                        'trans_start',
                                                                        'trans_act',
                                                                        'trans_end', 
                                                                        'start_principle', 
                                                                        'ret_3d',
                                                                        'ret_fractal',
                                                                        'pric_settle',
                                                                        'trans_3d',
                                                                        'price_end_3d',
                                                                        'price_end_fractal',
                                                                        'ret_fractal_sim',
                                                                        'long',
                                                                        'duration_time',])
    elif (mode == 'hmapower') or \
        (mode == 'hmapower120') or \
        (mode == 'hmapower30'):
        if (long > 0):
            # 做多方向
            hmapower120_endpoint_before = Timeline_duration(np.where(ref_features[FLD.HMAPOWER120_TIMING_LAG] == -1, 1, 0))
        else:
            # 做空方向
            hmapower120_endpoint_before = Timeline_duration(np.where(ref_features[FLD.HMAPOWER120_TIMING_LAG] == 1, 1, 0))
        fractal_hmapower120_position = np.where(fractal_cross_before < hmapower120_endpoint_before, 1, 0)
        fractal_hmapower120_lag = calc_event_timing_lag(np.where(fractal_hmapower120_position > 0, 1, -1))
        transcation_stats = calc_transcation_stats_np(fractal_triggers,
                                                   closep,
                                                   fractal_forcast_3d_lag,
                                                   fractal_hmapower120_lag,
                                                   taxfee=taxfee,
                                                   long=long)

        transcation_stats_df = pd.DataFrame(transcation_stats, columns=['trans_stats',
                                                                        'trans_start',
                                                                        'trans_act',
                                                                        'trans_end', 
                                                                        'start_principle', 
                                                                        'ret_3d',
                                                                        'ret_fractal',
                                                                        'pric_settle',
                                                                        'trans_3d',
                                                                        'price_end_3d',
                                                                        'price_end_fractal',
                                                                        'ret_fractal_sim',
                                                                        'long',
                                                                        'duration_time',])
    elif (mode == 'raw'):
        fractal_position = np.where(fractal_triggers > 0, 1, 0)
        fractal_timing_lag = calc_event_timing_lag(np.where(fractal_position > 0, 1, -1))
        if (np.max(fractal_timing_lag) < 12):
            #print('A spot Fractal, not a Complete Cycle Fractal')
            pass
        transcation_stats = calc_transcation_stats_np(fractal_triggers,
                                                      closep,
                                                      fractal_forcast_3d_lag,
                                                      fractal_timing_lag,
                                                      taxfee=taxfee,
                                                      long=long)

        transcation_stats_df = pd.DataFrame(transcation_stats, columns=['trans_stats',
                                                                        'trans_start',
                                                                        'trans_act',
                                                                        'trans_end', 
                                                                        'start_principle', 
                                                                        'ret_3d',
                                                                        'ret_fractal',
                                                                        'pric_settle',
                                                                        'trans_3d',
                                                                        'price_end_3d',
                                                                        'price_end_fractal',
                                                                        'ret_fractal_sim',
                                                                        'long',
                                                                        'duration_time',])

    transcation_stats_df[AKA.CODE] = symbol
    transcation_stats_df['fractal_uid'] = fractal_uid

    # bar ID索引 转换成交易时间戳
    selected_trans_start = ref_features.iloc[transcation_stats[:, 1], :]
    transcation_stats_df['trans_start'] = pd.to_datetime(selected_trans_start.index.get_level_values(level=0))
    selected_trans_action = ref_features.iloc[transcation_stats[:, 2], :]
    transcation_stats_df['trans_act'] = pd.to_datetime(selected_trans_action.index.get_level_values(level=0))
    selected_trans_end = ref_features.iloc[transcation_stats[:, 3], :]
    transcation_stats_df['trans_end'] = pd.to_datetime(selected_trans_end.index.get_level_values(level=0))

    transcation_stats_df = transcation_stats_df.assign(datetime=pd.to_datetime(selected_trans_start.index.get_level_values(level=0))).drop_duplicates((['datetime',
                                'code'])).set_index(['datetime',
                                'code'],
                                    drop=True)

    return transcation_stats_df


@nb.jit(nopython=True)
def calc_transcation_stats(fractal_triggers: np.ndarray, 
                           closep: np.ndarray,
                           fractal_forcast_position: np.ndarray,
                           fractal_sim_position: np.ndarray,
                           principle_timing_lag: np.ndarray,
                           taxfee: float=0.0003, 
                           long: int=1):

    """
    explanation:
        在“大方向”（规则）引导下，计算当前交易盈亏状况
        np.ndarray 实现，编码规范支持JIT和Cython加速

    params:
        fractal_triggers : np.array, 分型触发信号
        closep: np.array, 参考指标特征
        fractal_forcast_position: np.ndarray,
        fractal_principle_position: np.ndarray,
        principle_timing_lag:np.ndarray,
        taxfee: float32, 税费
        long: int32, 交易方向

    return:
        np.array
    """
    # 交易状态，状态机规则，低状态可以向高状态迁移
    stats_nop = 0            # 无状态
    stats_onhold = 1         # 执行交易并持有
    stats_suspended = 2      # 挂起，不执行交易，观察走势
    stats_closed = 3         # 结束交易
    stats_teminated = 4      # 趋势走势不对，终止交易

    idx_transcation = -1
    idx_transcation_stats = 0
    idx_transcation_start = 1
    idx_transcation_action = 2
    idx_transcation_endpoint = 3
    idx_start_in_principle = 4
    idx_forcast_returns = 5
    idx_principle_returns = 6
    idx_settle_price = 7
    idx_transcation_3d = 8
    idx_endpoint_price_3d = 9
    idx_endpoint_price_principle = 10
    idx_fractal_sim_returns = 11
    idx_long = 12
    idx_duration_time = 13
    #idx_lineareg_band_lag = 12
 
    ret_transcation_stats = np.zeros((len(closep), 14))
    onhold_price = onhold_returns = 0.0
    onhold_position_3d = onhold_position_lineareg = False
    assert long == 1 or long == -1
    ret_transcation_stats[:, idx_long] = long
    for i in range(0, len(closep)):
        # 开启交易判断
        if (fractal_triggers[i] > 0) and \
            (not onhold_position_3d) and \
            (not onhold_position_lineareg):
            onhold_position_3d = True
            onhold_position_lineareg = True
            idx_transcation = idx_transcation + 1
            ret_transcation_stats[idx_transcation, idx_transcation_start] = i

            if (principle_timing_lag[i] * long > 0):
                ret_transcation_stats[idx_transcation, 
                                      idx_start_in_principle] = principle_timing_lag[i]
                if (ret_transcation_stats[idx_transcation, 
                                          idx_start_in_principle] * long == -1):
                    ret_transcation_stats[idx_transcation, 
                                          idx_transcation_stats] = stats_suspended
                elif (ret_transcation_stats[idx_transcation, 
                                            idx_transcation_stats] < stats_onhold):
                    ret_transcation_stats[idx_transcation, 
                                          idx_transcation_stats] = stats_onhold
                    ret_transcation_stats[idx_transcation, 
                                          idx_transcation_action] = i
            else:
                ret_transcation_stats[idx_transcation, 
                                      idx_start_in_principle] = principle_timing_lag[i]
                if (ret_transcation_stats[idx_transcation, 
                                          idx_transcation_stats] < stats_suspended):
                    ret_transcation_stats[idx_transcation, 
                                          idx_transcation_stats] = stats_suspended

            if (principle_timing_lag[i] * long > 0):
                if (int(ret_transcation_stats[idx_transcation, 
                                              idx_transcation_stats]) == stats_onhold):
                    onhold_price = closep[i]
                    ret_transcation_stats[idx_transcation, 
                                          idx_settle_price] = onhold_price
            elif (i != len(closep)):
                if (int(ret_transcation_stats[idx_transcation, 
                                              idx_transcation_stats]) == stats_onhold):
                    onhold_price = closep[i + 1]
                    ret_transcation_stats[idx_transcation, 
                                          idx_settle_price] = onhold_price

        if (onhold_position_lineareg) and (fractal_forcast_position[i] > 0):
            if (principle_timing_lag[i] * long > 0):
                if (int(ret_transcation_stats[idx_transcation, 
                                              idx_transcation_stats]) == stats_suspended):
                    ret_transcation_stats[idx_transcation, 
                                          idx_transcation_stats] = stats_onhold
                    ret_transcation_stats[idx_transcation, 
                                          idx_transcation_action] = i
                    onhold_price = closep[i]
                    ret_transcation_stats[idx_transcation, 
                                          idx_settle_price] = onhold_price
            else:
                ret_transcation_stats[idx_transcation, 
                                      idx_transcation_stats] = stats_suspended

        # 结束交易判断
        if (onhold_position_lineareg) and (fractal_sim_position[i] <= 0):
            onhold_position_lineareg = False
            onhold_position_3d = False
            ret_transcation_stats[idx_transcation, 
                                  idx_transcation_endpoint] = i
            onhold_sim_price = closep[int(ret_transcation_stats[idx_transcation, idx_transcation_start])]
            ret_transcation_stats[idx_transcation, 
                                  idx_fractal_sim_returns] = (closep[i] - onhold_sim_price) / onhold_sim_price * long
            if (int(ret_transcation_stats[idx_transcation, 
                                          idx_transcation_stats]) == stats_onhold):
                ret_transcation_stats[idx_transcation, 
                                      idx_principle_returns] = (closep[i] - onhold_price) / onhold_price * long
                ret_transcation_stats[idx_transcation, 
                                      idx_endpoint_price_principle] = closep[i]
                ret_transcation_stats[idx_transcation, 
                                      idx_transcation_stats] = stats_closed
                onhold_price = 0.0
            elif (int(ret_transcation_stats[idx_transcation, 
                                            idx_transcation_stats]) == stats_suspended):
                ret_transcation_stats[idx_transcation, 
                                      idx_transcation_stats] = stats_teminated
                onhold_price = 0.0

        if (onhold_position_3d) and (fractal_forcast_position[i] <= 0):
            onhold_position_3d = False
            ret_transcation_stats[idx_transcation, 
                                  idx_transcation_3d] = principle_timing_lag[i]
            if (int(ret_transcation_stats[idx_transcation, 
                                          idx_transcation_stats]) == stats_onhold):
                ret_transcation_stats[idx_transcation, 
                                      idx_forcast_returns] = (closep[i] - onhold_price) / onhold_price * long
                ret_transcation_stats[idx_transcation, 
                                      idx_endpoint_price_3d] = closep[i]
            elif (int(ret_transcation_stats[idx_transcation, 
                                            idx_transcation_stats]) == stats_suspended):
                ret_transcation_stats[idx_transcation, 
                                      idx_transcation_stats] = stats_teminated
                onhold_price = 0.0
            else:
                pass

        if (onhold_position_lineareg) and (i == len(closep)):
            # 交易当前处于未结束状态
            if (int(ret_transcation_stats[idx_transcation, 
                                          idx_transcation_stats]) == stats_onhold):
                ret_transcation_stats[idx_transcation, 
                                      idx_principle_returns] = (closep[i] - onhold_price) / onhold_price * long
            pass
        ret_transcation_stats[idx_transcation, 
                              idx_duration_time] = ret_transcation_stats[idx_transcation, 
                                                                         idx_transcation_endpoint] - ret_transcation_stats[idx_transcation, 
                                                                                                                           idx_transcation_action]

    return ret_transcation_stats[:idx_transcation + 1, :]


@nb.jit(nopython=True)
def calc_transcation_stats_np(fractal_triggers:np.ndarray, 
                              closep:np.ndarray,
                              fractal_forcast_position:np.ndarray,
                              fractal_timing_lag:np.ndarray,
                              taxfee:float=0.0003, 
                              long:int=1):

    """
    计算当前交易盈亏状况
    np.ndarray 实现，编码规范支持JIT和Cython加速
    """
    # 交易状态，状态机规则，低状态可以向高状态迁移
    stats_nop = 0            # 无状态
    stats_onhold = 1         # 执行交易并持有
    stats_suspended = 2      # 挂起，不执行交易，观察走势
    stats_closed = 3         # 结束交易
    stats_teminated = 4      # 趋势走势不对，终止交易

    idx_transcation = -1
    idx_transcation_stats = 0
    idx_transcation_start = 1
    idx_transcation_action = 2
    idx_transcation_endpoint = 3
    idx_start_zen_wavelet = 4
    idx_forcast_returns = 5
    idx_fractal_returns = 6
    idx_settle_price = 7
    idx_transcation_3d = 8
    idx_endpoint_price_3d = 9
    idx_endpoint_price_fractal = 10
    idx_fractal_sim_returns = 11
    idx_long = 12
    idx_duration_time = 13
    #idx_lineareg_band_lag = 12
 
    ret_transcation_stats = np.zeros((len(closep), 14))
    onhold_price = onhold_returns = 0.0
    onhold_position_3d = onhold_position_lineareg = False
    assert long == 1 or long == -1
    ret_transcation_stats[:, idx_long] = long
    for i in range(0, len(closep)):
        # 开启交易判断
        if (fractal_triggers[i] > 0) and \
            (not onhold_position_3d) and \
            (not onhold_position_lineareg):
            onhold_position_3d = True
            onhold_position_lineareg = True
            idx_transcation = idx_transcation + 1
            ret_transcation_stats[idx_transcation, idx_transcation_start] = i
            ret_transcation_stats[idx_transcation, 
                                  idx_transcation_stats] = stats_onhold
            ret_transcation_stats[idx_transcation, idx_transcation_action] = i
            onhold_price = closep[i]
            ret_transcation_stats[idx_transcation, 
                                  idx_settle_price] = onhold_price

        # 结束交易判断
        if (onhold_position_lineareg) and (fractal_timing_lag[i] <= 0):
            onhold_position_lineareg = False
            onhold_position_3d = False
            ret_transcation_stats[idx_transcation, 
                                  idx_transcation_endpoint] = i
            ret_transcation_stats[idx_transcation, 
                                  idx_fractal_sim_returns] = (closep[i] - onhold_price) / onhold_price * long
            ret_transcation_stats[idx_transcation, 
                                  idx_fractal_returns] = (closep[i] - onhold_price) / onhold_price * long
            ret_transcation_stats[idx_transcation, 
                                  idx_endpoint_price_fractal] = closep[i]
            ret_transcation_stats[idx_transcation, 
                                  idx_transcation_stats] = stats_closed
            onhold_price = 0.0

        if (onhold_position_3d) and (fractal_forcast_position[i] <= 0):
            onhold_position_3d = False
            ret_transcation_stats[idx_transcation, 
                                  idx_transcation_3d] = fractal_timing_lag[i]
            if (onhold_position_lineareg):
                ret_transcation_stats[idx_transcation, 
                                      idx_forcast_returns] = (closep[i] - onhold_price) / onhold_price * long
                ret_transcation_stats[idx_transcation, 
                                      idx_endpoint_price_3d] = closep[i]
            else:
                ret_transcation_stats[idx_transcation, 
                                      idx_transcation_stats] = stats_teminated
                onhold_price = 0.0

        if (onhold_position_lineareg) and (i == len(closep)):
            # 交易当前处于未结束状态
            if (int(ret_transcation_stats[idx_transcation, 
                                          idx_transcation_stats]) == stats_onhold):
                ret_transcation_stats[idx_transcation, 
                                      idx_fractal_returns] = (closep[i] - onhold_price) / onhold_price * long
            pass
        ret_transcation_stats[idx_transcation, 
                              idx_duration_time] = ret_transcation_stats[idx_transcation, 
                                                                         idx_transcation_endpoint] - ret_transcation_stats[idx_transcation, 
                                                                                                                         idx_transcation_action]

    return ret_transcation_stats[:idx_transcation + 1, :]