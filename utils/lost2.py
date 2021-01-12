print(u'分析和展示计算结果...')
ret_codelist = featurs_printable_formatter(ret_codelist)
codelist_saltedfish, ret_codelist_saltedfish, ret_codelist = find_fratcal_saltfish(ret_codelist)
codelist_combo, ret_codelist_combo, ret_codelist = find_fratcal_bootstrap_combo(ret_codelist)
codelist_action, ret_codelist_action, ret_codelist_combo = find_action_long(ret_codelist_combo)
codelist_weak, ret_codelist_weak, ret_codelist = find_fratcal_weakshort(ret_codelist)
codelist_short, ret_codelist_short, ret_codelist_unknown = find_action_sellshort(ret_codelist)
#codelist_hyper_punch, ret_stocklist_hyper_punch, ret_codelist =
#find_action_hyper_punch(ret_codelist)
codelist_unknown = [index[1] for index, symbol in ret_codelist_unknown.iterrows()]

# 将计算的标的分成四类 —— 买入判断'buy'，持有判断'long'，做空判断'short',
# 'slatfish'是垃圾咸鱼票，既没有价格波动性也没有想象空间
if (eval_range in ['etf', 'csindex']):
    full_etf_list = perpar_symbol_range('etf')
    full_csindex_list = perpar_symbol_range('csindex')
    ret_codelist = ret_features_pd.loc[(each_day[-1], slice(None)), :].copy()
    symbol_list = ret_codelist.index.get_level_values(level=1).unique()
    symbol_list_grouped = [(symbol, 'etf') for symbol in list(set(symbol_list).intersection(set(full_etf_list)))] + \
                            [(symbol, 'csindex') for symbol in list(set(symbol_list).intersection(set(full_csindex_list)))]
    if (eval_range in ['etf']):
        symbol_list_grouped = [(symbol, 'etf') for symbol in symbol_list]
    elif (eval_range in ['csindex']):
        symbol_list_grouped = [(symbol, 'csindex') for symbol in symbol_list]
    symbol_list_grouped = list(set(symbol_list_grouped))
else:
    symbol_list_grouped = [(symbol, 'buy') for symbol in codelist_action] + \
                          [(symbol, 'long') for symbol in list(set(codelist_combo).difference(set(codelist_action)))] + \
                          [(symbol, 'weak') for symbol in codelist_weak] + \
                          [(symbol, 'short') for symbol in codelist_short] + \
                          [(symbol, 'saltedfish') for symbol in codelist_saltedfish] + \
                          [(symbol, 'unknown') for symbol in codelist_unknown]