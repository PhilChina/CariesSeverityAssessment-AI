from utils.io import get_json

label_dict= {'GAO_TE_m_46': 0, 'HONG_BING_JUN_m_15': 0, 'HUANG_JIN_DI_m_46': 0, 'HUANG_YA_PING_m_47': 0, 'LIANG_JIA_XIN_m_15': 0, 'BAO_YU_YING_d_46': 1, 'CHEN_RONG_m_16': 0, 'CHEN_FANG_m_27': 0, 'JI_YUN_d_27': 1, 'HUANG_CAI_YUAN_d_17': 1, 'AN_RAN_d_27': 1, 'CHEN_CHEN_m_47': 0, 'BIAN_YU_d_16': 1, 'HONG_BING_JUN_m_14': 0, 'CHEN_JUN_MING_m_28': 0, 'CHEN_XIN_YU_m_47': 0, 'GUO_BO_WEN_m_26': 0, 'JI_JING_m_46': 0, 'BAO_PENG_YU_d_16': 1, 'HUANG_QI_HUA_m_15': 0, 'CHEN_FANG_m_24': 0, 'CHEN_FEI_LONG_d_17': 1, 'CHEN_HAN_YING_m_37': 0, 'MA_JIE_d_26': 1, 'HUANG_AN_WEN_d_27': 1, 'HUANG_GUO_JING_m_27': 0, 'CHEN_FANG_LI_d_47': 1, 'AN_RAN_m_26': 0, 'HE_XIAO_HUI_d_27': 1, 'HUANG_HAN_TONG_d_47': 1, 'CHEN_JUN_MING_d_36': 1, 'CHEN_CHEN_m_46': 0, 'MA_JIE_m_35': 0, 'WANG_LIU_m_26': 0, 'CHEN_HONG_FENG_m_35': 0, 'CHE_LEI_d_16': 1, 'CHEN_HUI_XIA_d_47': 1, 'BAO_YUAN_d_17': 1, 'CHEN_JIE_m_28': 0, 'CHEN_RONG_FA_d_14': 1, 'CHEN_CHEN_m_36': 0, 'GU_WEI_LIANG_d_37': 1, 'CEN_ZHONG_QIAN_m_16': 0, 'CHEN_AI_DI_m_47': 0, 'XU_MENG_d_38': 1, 'WANG_KE_XUAN_m_16': 0, 'CHEN_CHEN_d_26': 1, 'CHEN_XIN_YU_m_25': 0, 'HUANG_HAO_BO_m_17': 0, 'CHEN_FANG_LI_m_35': 0, 'CHEN_FEI_FAN_d_25': 1, 'CHEN_XU_DONG_d_35': 1, 'HUANG_MEI_JUN_d_17': 1, 'BAO_YUAN_d_16': 1, 'GU_HUI_LIN_d_17': 1, 'BAO_DONG_YAN_d_46': 1, 'HU_LAN_YING_d_16': 1, 'CHEN_TIAN_YANG_d_16': 1, 'HUANG_YI_d_25': 1, 'HUANG_JIAN_XIN_d_45': 1, 'CHEN_YAN_MEI_d_36': 1, 'RAN_LONG_XIU_d_46': 1, 'CHEN_RONG_FA_d_16': 1, 'HANG_YAN_d_26': 1, 'BAO_PENG_YU_m_18': 0, 'BAO_PENG_YU_m_47': 0, 'TANG_TAN_m_13': 0, 'CEN_ZHONG_QIAN_m_17': 0, 'CHEN_SI_YUN_d_25': 1, 'HUANG_JIE_QING_d_17': 1, 'CHEN_LAN_d_48': 1, 'HANG_YAN_d_27': 1, 'CHEN_FANG_LI_m_46': 0, 'CHEN_DONG_DONG_m_47': 0, 'CHEN_HAN_YING_m_46': 0, 'WU_FENG_YA_m_27': 0, 'HOU_YONG_CEN_m_28': 0, 'LI_JIANG_d_27': 1, 'CEN_ZHONG_QIAN_m_36': 0, 'CHEN_JIAN_d_46': 1, 'HU_SHU_HUI_d_14': 1, 'CHEN_SI_YUN_m_17': 0, 'CHEN_ROU_FAN_d_37': 1, 'CHEN_HONG_FENG_m_27': 0, 'CHEN_LI_MING_m_24': 0, 'CAI_YI_SHENG_m_38': 0, 'CHEN_CHEN_d_17': 1, 'HUANG_CHEN_XIN_m_25': 0, 'CHEN_HONG_FENG_m_45': 0, 'YU_FEI_m_15': 0, 'HUANG_AN_WEN_d_17': 1, 'CAO_YAN_QING_m_35': 0, 'HUANG_BO_SONG_d_36': 1, 'CHEN_LIAN_FENG_m_14': 0, 'CHEN_XIN_YU_d_16': 1, 'TANG_ZHAO_HUI_m_47': 0, 'ZHOU_QIAN_HUI_m_16': 0, 'HOU_YONG_CEN_d_26': 1, 'HE_WEI_d_14': 1, 'CHEN_HONG_d_37': 1, 'WANG_TAN_m_35': 0, 'CHEN_FANG_LI_m_26': 0, 'CAI_YI_SHENG_m_37': 0, 'CHEN_HAI_HONG_d_26': 1, 'HUANG_HAN_TONG_m_27': 0, 'CHEN_QING_QUAN_d_28': 1, 'CHEN_YI_LIN_d_47': 1, 'YANG_XIAO_LV_m_44': 0, 'HUANG_YA_BING_d_46': 1, 'CHEN_JIAN_d_36': 1, 'CHE_LEI_m_46': 0, 'CHEN_FEI_LONG_d_24': 1, 'HUANG_HAN_TONG_m_46': 0, 'CHEN_YE_d_47': 1, 'CHEN_DONG_DONG_m_16': 0, 'HUANG_JIAN_XIN_d_37': 1, 'GU_JING_YAN_m_37': 0, 'YU_XIAN_d_46': 1, 'CHEN_HAI_HONG_m_34': 0, 'CHEN_FANG_d_36': 1, 'GUO_BING_YANG_d_47': 1, 'CHEN_HONG_FENG_d_26': 1, 'LEI_SU_YING_m_36': 0, 'CHEN_FEI_FAN_m_24': 0, 'CHEN_LI_MING_d_36': 1, 'HUANG_JIE_LI_d_17': 1, 'JI_YUN_d_47': 1, 'CHEN_JIE_d_24': 1, 'CHEN_XU_DONG_m_45': 0, 'GONG_JUN_YING_d_26': 1, 'LEI_SU_YING_m_13': 0, 'WANG_DI_m_47': 0, 'CHEN_AI_DI_m_46': 0, 'BIAN_YU_d_17': 1, 'XU_MENG_m_25': 0, 'BAO_DONG_YAN_m_47': 0, 'CHEN_HONG_FENG_d_44': 1, 'HE_WEI_m_37': 0, 'HUANG_GONG_YAN_m_27': 0, 'BAO_YU_YING_d_17': 1, 'CHEN_CHEN_m_37': 0, 'CHEN_FEI_FAN_m_36': 0, 'CHEN_DONG_DONG_m_45': 0, 'LU_HAN_BING_m_15': 0, 'CHEN_AI_DI_d_37': 1, 'HUANG_BO_SONG_m_47': 0, 'CHEN_JIE_m_17': 0, 'CHEN_LIAN_FENG_m_17': 0, 'GU_WEI_LIANG_m_38': 0, 'CHEN_HUI_XIA_d_48': 1, 'XU_MENG_m_24': 0, 'HUANG_HAO_BO_m_16': 0, 'BAO_YA_LAN_d_27': 1, 'ZHANG_JIAN_YOU_d_47': 1, 'HUANG_GONG_YAN_m_34': 0, 'CAO_YING_YING_m_37': 0, 'HUA_JING_YU_d_15': 1, 'CHEN_QING_QUAN_d_26': 1, 'CHEN_SI_YUN_m_27': 0, 'PAN_RUI_m_26': 0, 'GU_FENG_d_47': 1, 'CHEN_XU_DONG_m_47': 0, 'XIE_AI_LAN_d_47': 1, 'JI_YUN_d_26': 1, 'CHEN_HAN_YING_m_27': 0, 'CAO_YING_YING_m_47': 0, 'QIN_WEI_m_15': 0, 'Al_niwaerwusiman_d_15': 1, 'CHEN_YING_m_35': 0, 'HE_RUI_m_27': 0, 'GUO_LI_XIA_d_18': 1, 'BAO_JIN_XIA_d_16': 1, 'HE_RUI_m_16': 0, 'HONG_JI_YAO_m_47': 0, 'CHEN_FEI_FAN_m_45': 0, 'HU_JING_m_27': 0, 'HUANG_QI_HUA_m_14': 0, 'CEN_ZHONG_QIAN_d_14': 1, 'CHEN_LAN_m_47': 0, 'HUI_XIAO_HAI_d_28': 1, 'PAN_RUI_m_36': 0, 'JI_JING_d_25': 1, 'HU_YUAN_YUAN_d_37': 1, 'CHEN_YONG_FANG_m_26': 0, 'BAO_PENG_YU_m_48': 0, 'CHEN_HAI_HONG_d_36': 1, 'HONG_JI_YAO_d_16': 1, 'HE_XIAO_HUI_m_47': 0, 'ZHANG_JIAN_YOU_m_46': 0, 'BIAN_YU_d_27': 1, 'HU_WEN_QIAN_m_35': 0, 'GU_SHUN_LEI_m_17': 0, 'CHEN_PEI_WEN_m_24': 0, 'HUA_JING_YU_d_14': 1, 'WANG_GANG_m_15': 0, 'HUA_JING_YU_m_17': 0, 'CHEN_RONG_FA_d_47': 1, 'CHEN_FEI_LONG_m_47': 0, 'CHEN_DONG_DONG_d_14': 1, 'CHEN_JUN_m_17': 0, 'CAO_YING_YING_m_17': 0, 'CHEN_HAI_HONG_m_24': 0, 'ZHAO_GUO_TAO_m_47': 0, 'GU_YU_MIN_d_28': 1, 'CHEN_LIN_JIA_d_25': 1, 'CAI_YI_SHENG_m_36': 0, 'SHAZ_ZHOU_d_27': 1, 'CHEN_XU_DONG_m_15': 0, 'HUANG_XIANG_RU_m_14': 0, 'HUA_JING_YU_m_47': 0, 'GUAN_PEI_LAN_d_25': 1, 'ZHOU_JIA_YI_m_45': 0, 'CAI_LI_PING_d_37': 1, 'GUO_BING_YANG_m_27': 0, 'HUANG_XIANG_RU_d_37': 1, 'JI_JING_d_24': 1, 'CHEN_LIN_JIA_d_15': 1, 'CHEN_RONG_FA_d_46': 1, 'HUANG_CHUN_MING_d_16': 1, 'CHEN_FANG_m_26': 0, 'CHEN_DONG_DONG_m_44': 0, 'XIE_AI_LAN_m_15': 0, 'CHEN_FEI_FAN_m_47': 0, 'LIANG_ZHI_JUN_d_47': 1, 'BAO_YU_YING_d_44': 1, 'CHEN_FEI_FAN_m_16': 0, 'GAO_TE_m_47': 0, 'CAO_YAN_QING_m_26': 0, 'HE_WEI_m_36': 0, 'JI_FEI_d_46': 1, 'HE_XIAO_HUI_m_35': 0, 'CHEN_FANG_LI_m_16': 0, 'HU_LAN_YING_m_45': 0, 'CHEN_RONG_FA_d_17': 1, 'CHEN_RONG_FA_d_36': 1, 'GU_YI_MING_d_27': 1, 'CHEN_DONG_DONG_m_46': 0, 'HE_WEI_m_47': 0, 'HE_QIN_d_27': 1, 'GUO_BING_YANG_d_26': 1, 'CHEN_DONG_DONG_m_13': 0, 'HUANG_XIANG_RU_m_26': 0, 'AN_RAN_m_46': 0, 'CEN_ZHONG_QIAN_m_26': 0, 'CHEN_QING_QUAN_d_17': 1, 'HU_LAN_YING_m_14': 0, 'HUANG_QIAN_d_25': 1, 'CHEN_LIAN_FENG_m_16': 0, 'CHEN_YING_d_25': 1, 'CHEN_RONG_FA_d_25': 1, 'JI_YUN_d_38': 1, 'XIE_AI_LAN_d_46': 1, 'CHEN_JIE_d_18': 1, 'GONG_LIANG_d_26': 1, 'LEI_SU_YING_d_37': 1, 'HUANG_XIANG_RU_d_46': 1, 'CHEN_FANG_m_25': 0, 'HUANG_YA_PING_m_44': 0, 'GU_SHUN_LEI_d_26': 1, 'CHE_QI_MIMNG_d_15': 1, 'Al_niwaerwusiman_d_26': 1, 'CHEN_RONG_FA_m_24': 0, 'BAO_JIN_XIA_d_46': 1, 'CHEN_RONG_FA_m_37': 0, 'HU_ZHI_LING_m_24': 0, 'JI_FEI_d_26': 1, 'CHEN_TIAN_YANG_d_37': 1, 'CHEN_TIAN_YANG_m_26': 0, 'JI_FEI_d_47': 1, 'CHEN_XIAO_WEN_d_26': 1, 'CHEN_TIAN_YANG_m_47': 0, 'JI_QING_d_17': 1, 'CHEN_XIN_YU_m_46': 0, 'CAI_LI_PING_m_24': 0, 'CHEN_XU_DONG_d_16': 1, 'CAI_YI_SHENG_d_26': 1, 'LE_JIANG_YUN_d_15': 1, 'CAO_YING_YING_d_27': 1, 'LIU_CHENG_m_24': 0, 'CAO_YING_YING_m_35': 0, 'LUO_HUA_m_27': 0, 'CHEN_YI_LIN_m_15': 0, 'CHEN_YI_LIN_m_46': 0, 'CHEN_YONG_FANG_d_17': 1, 'CHEN_YE_m_14': 0, 'CHE_LEI_m_17': 0, 'CHEN_AI_DI_d_25': 1, 'CHE_LEI_m_47': 0, 'QIAO_YANG_KAI_d_46': 1, 'CHEN_AI_DI_d_27': 1, 'PAN_SI_CHENG_d_15': 1, 'CHEN_AI_DI_m_38': 0, 'WANG_DI_m_48': 0, 'CHEN_CHEN_m_25': 0, 'GONG_LIANG_d_46': 1, 'GUAN_PEI_LAN_d_26': 1, 'XU_MENG_d_26': 1, 'CHEN_DONG_DONG_m_17': 0, 'GU_FENG_m_17': 0, 'CHEN_FANG_LI_d_37': 1, 'GU_YU_MIN_d_46': 1, 'CHEN_FANG_LI_m_45': 0, 'HAN_YA_LI_m_15': 0, 'CHEN_FEI_FAN_m_35': 0, 'HE_XIAO_HUI_m_45': 0, 'CHEN_FEI_FAN_m_46': 0, 'CHEN_HAI_HONG_m_25': 0, 'HUANG_AN_WEN_d_35': 1, 'CHEN_HAN_YING_d_15': 1, 'HUANG_AN_WEN_d_37': 1, 'HUANG_CHUN_MING_d_14': 1, 'CHEN_HONG_FENG_m_15': 0, 'HUANG_HAN_TONG_d_24': 1, 'HUANG_HAN_TONG_d_25': 1, 'HUANG_JIAN_XIN_d_35': 1, 'CHEN_JUN_d_48': 1, 'HUANG_JIN_DI_d_15': 1, 'HUANG_MEI_JUN_d_26': 1, 'CHEN_LIAN_FENG_d_18': 1, 'HUANG_MIN_JUAN_d_18': 1, 'HUANG_QI_HUA_d_17': 1, 'CHEN_LIAN_FENG_m_46': 0, 'CHEN_PEI_WEN_d_26': 1, 'HUANG_YI_HUA_d_16': 1}


if __name__ == "__main__":
    split_dir = './split/caries_classifer_split_12_30.json'
    split_ = get_json(split_dir)
    print(split_['0']['train'])
    print(split_['0']['test'])
    all_samples = split_['0']['train'] + split_['0']['test']
    print("all_samples：", len(all_samples))
    split_data = [s for s in all_samples]

    label = {}
    for el_data in split_data:
        if "_m_" in el_data:
            label[el_data] = 0
        elif "_d_" in el_data:
            label[el_data] = 1
        else:
            print('{} label error!!'.format(el_data))
    print(label)


