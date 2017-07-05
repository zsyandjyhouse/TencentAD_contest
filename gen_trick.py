#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 抽取trick特征

import math
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import threading
import gc

FilePath = '../final/'
FeaFilePath = '../../gen/'
TrainFile = 'train.csv'
CleanedTrainFile = 'cleaned_train.csv'
TestFile = 'test.csv'
AdFile = 'ad.csv'
User_basic_File = 'user.csv'
AppCategoriesFile = 'app_categories.csv'
UserInstallAppsFile = 'user_installedapps.csv'
UserAppActionsFile = 'user_app_actions.csv'
PositionFile = 'position.csv'

result_list = []
delete_list = []
user_install_count_previous15_list = []
app_install_count_previous15_list = []
app_install_count_yestoday_list = []

#计算要删除的广告主id以及最后最后一次回流时间
def find_delete_advertiser():
    advertiser_conversion_Data = pd.read_csv(FilePath+'advertiser_analy.csv')
    advertiser_conversion_Data = advertiser_conversion_Data[advertiser_conversion_Data['late_clickTime']>=26000000]
    advertiser_conversion_Data = advertiser_conversion_Data[['advertiserID','late_conversionTime']]
    advertiser_conversion_Data = advertiser_conversion_Data.dropna()
    advertiser_conversion_Data = advertiser_conversion_Data.astype(int)
    advertiser_conversion_data = np.array(advertiser_conversion_Data).tolist()
    return advertiser_conversion_data

def get_index_to_delete(data_all,item):
    advertiserID = item[0]
    time = item[1]
    delete_data = data_all.loc[(data_all['advertiserID'] == advertiserID) & (data_all['clickTime'] > time)].index
    delete_index = delete_data.tolist()
    delete_list.extend(delete_index)
    result_list.append(1)

def delete_conversion_data():
    train_data = pd.read_hdf(FilePath + 'train_0613_nodelconvert')
    print 'read finish'
    advertiser_conversion_list = find_delete_advertiser()
    print len(advertiser_conversion_list)

    for item in advertiser_conversion_list:
        t = threading.Thread(target=get_index_to_delete,args=(train_data,item))
        t.start()
    while len(result_list)<len(advertiser_conversion_list):
        pass
    train_data.drop(delete_list, axis=0, inplace=True)
    train_data = train_data.reset_index()
    del train_data['index']
    print 'train write begin'
    train_data.to_hdf(FilePath + 'train_0613', 'all')
    delete_list = Series(delete_list)
    delete_list.to_csv(FilePath + 'delete_negsample_index_oftrain0613.csv', mode='a', index=False)

def fun1(x):
    if x==1:
        return 2
    else:
        return 0
def fun2(x):
    if x==1:
        return 3
    else:
        return 0
# 获取用户操作跟上一条的时间差
def get_min_click_ID_time_diff(x):
    x['user'] = x['userID'].shift(1)
    x['time_delta_user'] = x['clickTime'] - x['clickTime'].shift(1)  # 减去上一条的时间
    x.loc[x.user != x.userID, 'time_delta_user'] = -1  # 第一条的时间差是-1
    return x.drop('user', axis=1)

# 获取用户，creative跟上一条的时间差
def get_min_click_IDS_time_diff(x):
    x['user'] = x['userID'].shift(1)
    x['creative'] = x['creativeID'].shift(1)
    x['time_delta_user_creative'] = x['clickTime'] - x['clickTime'].shift(1)  # 减去上一条的时间
    x.loc[
        ((x.user != x.userID) | (x.creative != x.creativeID)), 'time_delta_user_creative'] = -1  # 第一条的时间差是-1
    return x.drop(['user', 'creative'], axis=1)

# 获取用户，appID跟上一条的时间差
def get_min_click_IDS_app_time_diff(x):
    x['user'] = x['userID'].shift(1)
    x['app'] = x['appID'].shift(1)
    x['time_delta_user_app'] = x['clickTime'] - x['clickTime'].shift(1)  # 减去上一条的时间
    x.loc[
        ((x.user != x.userID) | (x.app != x.appID)), 'time_delta_user_app'] = -1  # 第一条的时间差是-1
    return x.drop(['user', 'app'], axis=1)

# 获取用户，appID跟上一条的时间差
def get_min_click_IDS_position_time_diff(x):
    x['user'] = x['userID'].shift(1)
    x['position'] = x['positionID'].shift(1)
    x['time_delta_user_position'] = x['clickTime'] - x['clickTime'].shift(1)  # 减去上一条的时间
    x.loc[
        ((x.user != x.userID) | (x.position != x.positionID)), 'time_delta_user_position'] = -1  # 第一条的时间差是-1
    return x.drop(['user', 'position'], axis=1)

# 获取用户操作跟下一条的时间差
def get_min_next_click_ID_time_diff(x):
    x['user'] = x['userID'].shift(-1)
    x['time_delta_user_next'] = x['clickTime'].shift(-1) - x['clickTime']
    x.loc[x.user != x.userID, 'time_delta_user_next'] = -1
    return x.drop('user', axis=1)

# 获取用户，creative跟下一条的时间差
def get_min_next_click_IDS_time_diff(x):
    x['user'] = x['userID'].shift(-1)
    x['creative'] = x['creativeID'].shift(-1)
    x['time_delta_user_creative_next'] = x['clickTime'].shift(-1) - x['clickTime']
    x.loc[((x.user != x.userID) | (x.creative != x.creativeID)), 'time_delta_user_creative_next'] = -1
    return x.drop(['user', 'creative'], axis=1)

# 获取用户，appID跟下一条的时间差
def get_min_next_click_IDS_app_time_diff(x):
    x['user'] = x['userID'].shift(-1)
    x['app'] = x['appID'].shift(-1)
    x['time_delta_user_app_next'] = x['clickTime'].shift(-1) - x['clickTime']
    x.loc[((x.user != x.userID) | (x.app != x.appID)), 'time_delta_user_app_next'] = -1
    return x.drop(['user', 'app'], axis=1)

# 获取用户，appID跟下一条的时间差
def get_min_next_click_IDS_position_time_diff(x):
    x['user'] = x['userID'].shift(-1)
    x['position'] = x['positionID'].shift(-1)
    x['time_delta_user_position_next'] = x['clickTime'].shift(-1) - x['clickTime']
    x.loc[((x.user != x.userID) | (x.position != x.positionID)), 'time_delta_user_position_next'] = -1
    return x.drop(['user', 'position'], axis=1)


def get_userID_delta(x):
    x['user'] = x['userID'].shift(1)
    x['user_delta_up'] = x['userID'] - x['user']  # 减去上一条的userID
    x.loc[x.user_delta_up != 0, 'user_delta_up'] = 2
    return x.drop('user', axis=1)
def get_userID_delta_next(x):
    x['user'] = x['userID'].shift(-1)
    x['user_delta_next'] = x['userID'] - x['user']  # 减去xia一条的userID
    x.loc[x.user_delta_next != 0, 'user_delta_next'] = 3  # 第一条的时间差是-1
    return x.drop('user', axis=1)
def get_userID_creativeID_delta(x):
    x['user'] = x['userID'].shift(1)
    x['creative'] = x['creativeID'].shift(1)
    x['user_delta_up'] = x['userID'] - x['user']  # 减去上一条的userID
    x['creative_delta_up'] = x['creativeID'] - x['creative']  # 减去上一条的creativeID
    x.loc[(x.user_delta_up != 0)|(x.creative_delta_up != 0), 'user_creative_delta_up'] = 2  # 第一条的rank是2
    x.fillna(0,inplace=True)
    return x.drop(['user','creative','user_delta_up','creative_delta_up'], axis=1)
def get_userID_creativeID_delta_next(x):
    x['user'] = x['userID'].shift(-1)
    x['creative'] = x['creativeID'].shift(-1)
    x['user_delta_next'] = x['userID'] - x['user']  # 减去xia一条的userID
    x['creative_delta_next'] = x['creativeID'] - x['creative']  # 减去xia一条的userID
    x.loc[(x.user_delta_next != 0)|(x.creative_delta_next != 0), 'user_creative_delta_next'] = 3  # 第一条的时间差是-1
    x.fillna(0, inplace=True)
    return x.drop(['user','creative','user_delta_next','creative_delta_next'], axis=1)
def get_userID_appID_delta(x):
    x['user'] = x['userID'].shift(1)
    x['app'] = x['appID'].shift(1)
    x['user_delta_up'] = x['userID'] - x['user']  # 减去上一条的userID
    x['app_delta_up'] = x['appID'] - x['app']  # 减去上一条的creativeID
    x.loc[(x.user_delta_up != 0)|(x.app_delta_up != 0), 'user_app_delta_up'] = 2  # 第一条的rank是2
    x.fillna(0,inplace=True)
    return x.drop(['user','app','user_delta_up','app_delta_up'], axis=1)
def get_userID_appID_delta_next(x):
    x['user'] = x['userID'].shift(-1)
    x['app'] = x['appID'].shift(-1)
    x['user_delta_next'] = x['userID'] - x['user']  # 减去xia一条的userID
    x['app_delta_next'] = x['appID'] - x['app']  # 减去xia一条的userID
    x.loc[(x.user_delta_next != 0)|(x.app_delta_next != 0), 'user_app_delta_next'] = 3  # 第一条的时间差是-1
    x.fillna(0, inplace=True)
    return x.drop(['user','app','user_delta_next','app_delta_next'], axis=1)

def get_userID_positionID_delta(x):
    x['user'] = x['userID'].shift(1)
    x['position'] = x['positionID'].shift(1)
    x['user_delta_up'] = x['userID'] - x['user']  # 减去上一条的userID
    x['position_delta_up'] = x['positionID'] - x['position']  # 减去上一条的creativeID
    x.loc[(x.user_delta_up != 0)|(x.position_delta_up != 0), 'user_position_delta_up'] = 2  # 第一条的rank是2
    x.fillna(0,inplace=True)
    return x.drop(['user','position','user_delta_up','position_delta_up'], axis=1)
def get_userID_positionID_delta_next(x):
    x['user'] = x['userID'].shift(-1)
    x['position'] = x['positionID'].shift(-1)
    x['user_delta_next'] = x['userID'] - x['user']  # 减去xia一条的userID
    x['position_delta_next'] = x['positionID'] - x['position']  # 减去xia一条的userID
    x.loc[(x.user_delta_next != 0)|(x.position_delta_next != 0), 'user_position_delta_next'] = 3  # 第一条的时间差是-1
    x.fillna(0, inplace=True)
    return x.drop(['user','position','user_delta_next','position_delta_next'], axis=1)


#计算时间差
def cal_time_delta():
    train_data = pd.read_hdf(FilePath+'train_cleaned')
    test_data = pd.read_hdf(FilePath+'test_cleaned')
    print 'read finish'
    train_data1 = train_data[['label','userID', 'creativeID', 'clickTime','connectionType','telecomsOperator']]
    test_data1 = test_data[['label','userID', 'creativeID', 'clickTime','connectionType','telecomsOperator']]

    train_test_data = pd.concat([train_data1,test_data1],axis=0)
    print 'concat finish'
    actions = train_test_data.sort_values(by=['userID', 'creativeID', 'clickTime'])
    actions.reset_index(inplace=True)
    del actions['index']
    # 获取user_creative操作跟上一条、下一条的时间差
    actions = get_min_click_IDS_time_diff(actions)
    actions = get_min_next_click_IDS_time_diff(actions)

    actions = actions.sort_values(by=['userID', 'clickTime'])
    actions.reset_index(inplace=True)
    del actions['index']
    # 获取用户操作跟上一条、下一条的时间差
    actions = get_min_click_ID_time_diff(actions)
    actions = get_min_next_click_ID_time_diff(actions)
    print 'time_delta finish'
    print actions

    train_fea = actions[actions['clickTime'] < 31000000]
    print train_fea.shape
    test_fea = actions[actions['clickTime'] >= 31000000]
    print test_fea.shape
    train_data = pd.merge(train_data, train_fea,
                          on=['label', 'userID', 'creativeID', 'clickTime', 'connectionType', 'telecomsOperator'],
                          how='left')
    test_data = pd.merge(test_data, test_fea,
                         on=['label', 'userID', 'creativeID', 'clickTime', 'connectionType', 'telecomsOperator'],
                         how='left')
    print 'train大小', train_data.shape
    print 'test大小', test_data.shape
    print 'test write begin'
    test_data.to_hdf(FilePath + 'test_timedelta_0612', 'all')
    train_data.to_hdf(FilePath + 'train_timedelta_0612_nodelconvert', 'all')

#标注相同userID, creativeID非重复数据0，重复但不是第一条和最后一条1，重复且是第一条2，最后一条3
def if_user_creative_first_last():
    train_data = pd.read_hdf(FilePath+'train_cleaned')
    test_data = pd.read_hdf(FilePath+'test_cleaned')
    print 'read finish'
    train_data1 = train_data[['label','userID', 'creativeID', 'clickTime','connectionType','telecomsOperator']].drop_duplicates()
    test_data1 = test_data[['label','userID', 'creativeID', 'clickTime','connectionType','telecomsOperator']].drop_duplicates()
    train_test_data = pd.concat([train_data1,test_data1],axis=0)
    print 'concat finish'

    actions = train_test_data.sort_values(by=['userID', 'creativeID', 'clickTime'])
    actions.reset_index(inplace=True)
    del actions['index']
    # 获取user_creative操作跟上一条、下一条的时间差
    actions = get_min_click_IDS_time_diff(actions)
    actions = get_min_next_click_IDS_time_diff(actions)
    print 'time_delta user_creative finish'
    actions = get_userID_creativeID_delta(actions)
    actions = get_userID_creativeID_delta_next(actions)
    actions['rank_user_creative_click'] = actions['user_creative_delta_up'] + actions['user_creative_delta_next']
    actions['rank_user_creative_click'].replace([5], 1, inplace=True)
    actions.drop(['user_creative_delta_up', 'user_creative_delta_next'], axis=1, inplace=True)
    print 'rank user_creative finish'

    actions = actions.sort_values(by=['userID', 'clickTime'])
    actions.reset_index(inplace=True)
    del actions['index']
    # 获取用户操作跟上一条、下一条的时间差
    actions = get_min_click_ID_time_diff(actions)
    actions = get_min_next_click_ID_time_diff(actions)
    print 'time_delta user finish'

    actions = get_userID_delta(actions)
    actions = get_userID_delta_next(actions)
    actions['rank_user_click'] = actions['user_delta_up'] + actions['user_delta_next']
    actions['rank_user_click'].replace([5], 1, inplace=True)
    actions.drop(['user_delta_up', 'user_delta_next'], axis=1, inplace=True)
    print 'rank user finish'
    print actions[['time_delta_user','time_delta_user_creative','time_delta_user_next','time_delta_user_creative_next','rank_user_creative_click','rank_user_click']]


    train_fea = actions[actions['clickTime'] < 31000000]
    print train_fea.shape
    test_fea = actions[actions['clickTime'] >= 31000000]
    print test_fea.shape
    train_data = pd.merge(train_data, train_fea,
                          on=['label', 'userID', 'creativeID', 'clickTime', 'connectionType', 'telecomsOperator'],
                          how='left')
    test_data = pd.merge(test_data, test_fea,
                         on=['label', 'userID', 'creativeID', 'clickTime', 'connectionType', 'telecomsOperator'],
                         how='left')
    print 'train大小', train_data.shape
    print 'test大小', test_data.shape

    print 'write begin'
    test_data.to_hdf(FeaFilePath + 'test_0615_delta', 'all')
    train_data.to_hdf(FeaFilePath + 'train_0615_delta', 'all')


    # 清回流
    # print 'delete conversion delay begin'
    # advertiser_conversion_list = find_delete_advertiser()
    # print len(advertiser_conversion_list)
    # for item in advertiser_conversion_list:
    #     t = threading.Thread(target=get_index_to_delete, args=(train_data, item))
    #     t.start()
    # while len(result_list) < len(advertiser_conversion_list):
    #     pass
    # train_data.drop(delete_list, axis=0, inplace=True)
    # train_data = train_data.reset_index()
    # del train_data['index']
    # print 'train write begin'
    # train_data.to_hdf(FilePath + 'train_0613', 'all')
    # delete_list = Series(delete_list)
    # delete_list.to_csv(FilePath + 'delete_negsample_index_oftrain0613.csv', mode='a', index=False)
    #
    # advertiser_conversion_list = find_delete_advertiser()
    # for item in advertiser_conversion_list:
    #     advertiserID = item[0]
    #     time = item[1]
    #     delete_data = train_data[(train_data['advertiserID'] == advertiserID) & (train_data['clickTime'] > time)]
    #     delete_index = Series(delete_data.index)
    #     delete_index.to_hdf(FilePath + 'delete_negsample_index', 'all',mode='a')
    #     delete_index2 = list(delete_index)
    #     train_data.drop(delete_index2, axis=0, inplace=True)
    # train_data = train_data.reset_index()
    # del train_data['index']
    # print 'train write begin'
    # train_data.to_hdf(FilePath + 'train_0613', 'all')

#获取installed.csv中用户和app的安装次数
def gen_user_app_install_num_in_installed():
    data = pd.read_csv(FilePath+UserInstallAppsFile)
    user_install_app_num = data.groupby('userID').size().to_frame().reset_index().rename(columns={0: 'user_install_app_count_before'})
    app_install_user_num = data.groupby('appID').size().to_frame().reset_index().rename(
        columns={0: 'app_install_user_count_before'})
    user_install_app_num.to_hdf(FilePath+'user_install_app_num','all')
    app_install_user_num.to_hdf(FilePath+'app_install_user_num','all')

#多线程计算前十五天user和app的安装次数
def cal_user_app_install_count_previous15(data,day):
    end_time = day*1000000
    start_time = (day-15)*1000000
    data_previous = data[(data['installTime']>=start_time)&(data['installTime'] < end_time)]
    user_install_count_previous15 = data_previous.groupby('userID').size().to_frame().reset_index().rename(columns={0: 'user_install_count_previous15'})
    app_install_count_previous15 = data_previous.groupby('appID').size().to_frame().reset_index().rename(
        columns={0: 'app_install_count_previous15'})
    user_install_count_previous15['day'] = day
    app_install_count_previous15['day'] = day
    # print user_install_count_previous15
    user_install_count_previous15_list.append(user_install_count_previous15)
    app_install_count_previous15_list.append(app_install_count_previous15)

#计算前十五天user和app的安装次数
def gen_user_install_count_previous_15_day():
    action_data = pd.read_csv(FilePath+UserAppActionsFile)
    for i in range(17,32):
        t = threading.Thread(target=cal_user_app_install_count_previous15,args=(action_data,i))
        t.start()
    while (len(user_install_count_previous15_list)<15 or len(app_install_count_previous15_list)<15):
        pass
    flag=0
    for item in user_install_count_previous15_list:
        print flag
        if flag == 0:
            item.to_csv(FilePath+'user_install_count_previous15.csv',index=False)
        else:
            item.to_csv(FilePath + 'user_install_count_previous15.csv',mode='a',index=False,header=None)
        flag+=1
    flag = 0
    for item in app_install_count_previous15_list:
        if flag == 0:
            item.to_csv(FilePath + 'app_install_count_previous15.csv', index=False)
        else:
            item.to_csv(FilePath + 'app_install_count_previous15.csv', index=False, header=None, mode='a')
        flag += 1

#计算user之前有没有安装过这个app
def has_user_installed_app():
    train_data = pd.read_csv(FilePath+TrainFile)
    action_data = pd.read_csv(FilePath + UserAppActionsFile)
    ad_data = pd.read_csv(FilePath+AdFile)
    train_data.drop(['label'],axis=1,inplace=True)
    train_data = train_data.drop_duplicates()
    test_data = pd.read_csv(FilePath+TestFile)
    test_data.drop(['instanceID'],axis=1,inplace=True)
    test_data = test_data.drop_duplicates()
    train_data = pd.concat([train_data,test_data],axis=0)
    train_data = pd.merge(train_data, ad_data, on=['creativeID'], how='left')
    train_data = pd.merge(train_data,action_data,on=['userID','appID'],how='left').fillna(32000000)
    train_data['day'] = train_data['clickTime']/1000000
    train_data['day'] = train_data['day'].astype(int)
    train_data['installday'] = train_data['installTime'] / 1000000
    train_data['installday'] = train_data['installday'].astype(int)
    train_data['user_app_before_has_installed'] = train_data.installday < train_data.day
    train_data = train_data.astype(int)
    train_data = train_data.groupby(['userID', 'appID', 'day'], as_index=False)['user_app_before_has_installed'].sum()
    train_data = train_data[['userID', 'appID', 'day', 'user_app_before_has_installed']]
    train_data.to_hdf(FilePath+'user_app_before_has_installed','all')

#多线程计算前一天app的安装次数
def cal_user_app_install_count_yestoday(data,day):
    end_time = day*1000000
    start_time = (day-1)*1000000
    data_previous = data[(data['installTime']>=start_time)&(data['installTime'] < end_time)]
    app_install_count_previous15 = data_previous.groupby('appID').size().to_frame().reset_index().rename(
        columns={0: 'app_install_count_yestoday'})
    app_install_count_previous15['app_install_count_yestoday'].fillna(0,inplace=True)
    app_install_count_previous15['app_install_count_yestoday'].replace(0, 0.5, inplace=True)
    app_install_count_previous15['app_install_count_yestoday'] = np.log2(app_install_count_previous15['app_install_count_yestoday'])
    app_install_count_previous15['day'] = day
    app_install_count_yestoday_list.append(app_install_count_previous15)

#计算user和app前一天安装量
def gen_user_app_install_yestoday():
    action_data = pd.read_csv(FilePath + UserAppActionsFile)
    for i in range(17, 32):
        t = threading.Thread(target=cal_user_app_install_count_yestoday, args=(action_data, i))
        t.start()
    while (len(app_install_count_yestoday_list) < 15 or len(app_install_count_yestoday_list) < 15):
        pass
    flag = 0
    print flag
    for item in app_install_count_yestoday_list:
        if flag == 0:
            item.to_csv(FilePath + 'app_install_count_yestoday.csv', index=False)
        else:
            item.to_csv(FilePath + 'app_install_count_yestoday.csv', index=False, header=None, mode='a')
        flag += 1

#merge0614生成的特征
def merge_fea_0614(train_data):
    train_data['day'] = train_data['clickTime']/1000000
    train_data['day'] = train_data['day'].astype(int)
    user_install_app_num = pd.read_hdf(FilePath + 'user_install_app_num')
    train_data = pd.merge(train_data,user_install_app_num,on='userID',how='left')
    print '11111'
    print train_data.shape
    del user_install_app_num
    gc.collect()
    app_install_user_num= pd.read_hdf(FilePath + 'app_install_user_num')
    train_data = pd.merge(train_data, app_install_user_num, on='appID', how='left')
    print '22222'
    print train_data.shape
    del app_install_user_num
    gc.collect()
    user_instal_count_previous_15 = pd.read_csv(FilePath+'user_install_count_previous15.csv')
    train_data = pd.merge(train_data,user_instal_count_previous_15,on=['userID','day'],how='left')
    print '33333'
    print train_data.shape
    del user_instal_count_previous_15
    gc.collect()
    app_install_count_previous15 = pd.read_csv(FilePath+'app_install_count_previous15.csv')
    train_data = pd.merge(train_data,app_install_count_previous15,on=['appID','day'],how='left')
    print '44444'
    print train_data.shape
    del app_install_count_previous15
    gc.collect()
    has_installed_data = pd.read_hdf(FilePath+'user_app_before_has_installed')
    train_data = pd.merge(train_data, has_installed_data, on=['userID','appID', 'day'], how='left')
    print '55555'
    print train_data.shape
    del has_installed_data
    gc.collect()
    # user_install_count_yestoday_data = pd.read_csv(FilePath+'user_install_count_yestoday.csv')
    # train_data = pd.merge(train_data, user_install_count_yestoday_data, on=['userID', 'day'], how='left')
    # print '66666'
    # del user_install_count_yestoday_data
    # gc.collect()
    app_install_count_yestoday = pd.read_csv(FilePath + 'app_install_count_yestoday.csv')
    train_data = pd.merge(train_data, app_install_count_yestoday, on=['appID', 'day'], how='left')
    print '77777'
    print train_data.shape
    del app_install_count_yestoday
    gc.collect()
    return train_data.drop('day',axis=1)

#分别将train和test输入上面的函数，merge特征，存储
def run_merge_fea_0614():
    train_data = pd.read_hdf(FeaFilePath + 'train_0615_delta')
    print 'trian read finish'
    print train_data.shape
    train_data = merge_fea_0614(train_data)
    train_data.to_hdf(FeaFilePath+'train_0616','all')
    del train_data
    gc.collect()
    test_data = pd.read_hdf(FeaFilePath + 'test_0615_delta')
    print 'test read finish'
    print test_data.shape
    test_data = merge_fea_0614(test_data)
    test_data.to_hdf(FeaFilePath+ 'test_0616', 'all')
    
#计算转化时间
def cal_conversion_time(x):
    clicktime = x.clickTime
    conversiontime = x.conversionTime
    clickdd = int(str(clicktime)[0:2])
    clickhh = int(str(clicktime)[2:4])
    clickmm = int(str(clicktime)[4:6])
    clickss = int(str(clicktime)[6:8])
    conversiondd = int(str(conversiontime)[0:2])
    conversionhh = int(str(conversiontime)[2:4])
    conversionmm = int(str(conversiontime)[4:6])
    conversionss = int(str(conversiontime)[6:8])
    conversion_time = (conversiondd-clickdd)*86400+(conversionhh-clickhh)*3600+(conversionmm-clickmm)*60+conversionss-clickss
    return conversion_time
    
#统计app平均回流时间
def app_conversion_time_average():
    train_data = pd.read_csv(FilePath+TrainFile)
    ad_data = pd.read_csv(FilePath+AdFile)
    train_data = pd.merge(train_data,ad_data,on='creativeID',how='left')
    train_data = train_data[train_data['label']==1]
    train_data['app_conversion_time_average'] = train_data.apply(cal_conversion_time,axis=1)
    data = train_data[['appID', 'app_conversion_time_average']]
    data = data.groupby('appID').agg('median').reset_index()
    data.to_hdf(FilePath+'app_conversion_time_average','all')

#统计advertiseriD平均回流时间
def adevertiser_conversion_time_average():
    train_data = pd.read_csv(FilePath+TrainFile)
    ad_data = pd.read_csv(FilePath+AdFile)
    train_data = pd.merge(train_data,ad_data,on='creativeID',how='left')
    train_data = train_data[train_data['label']==1]
    train_data['acvertiser_conversion_time_average'] = train_data.apply(cal_conversion_time,axis=1)
    data = train_data[['advertiserID', 'acvertiser_conversion_time_average']]
    data = data.groupby('advertiserID').agg('median').reset_index()
    data.to_hdf(FilePath+'adevertiser_conversion_time_average','all')
#
# #把appID回流时间加进来
def merge_fea_0616():
    train_data = pd.read_hdf(FeaFilePath+'train_0616')
    app_conversion_time_average = pd.read_hdf(FilePath+'app_conversion_time_average')
    adevertiser_conversion_time_average = pd.read_hdf(FilePath + 'adevertiser_conversion_time_average')
    train_data = pd.merge(train_data,app_conversion_time_average,on='appID',how='left')
    train_data = pd.merge(train_data, adevertiser_conversion_time_average, on='advertiserID', how='left')
    train_data.to_hdf(FeaFilePath+'train_0616_c','all')
    del train_data
    gc.collect()
    test_data = pd.read_hdf(FeaFilePath + 'test_0616')
    test_data = pd.merge(test_data, app_conversion_time_average, on='appID', how='left')
    test_data = pd.merge(test_data, adevertiser_conversion_time_average, on='advertiserID', how='left')
    test_data.to_hdf(FeaFilePath + 'test_0616_c', 'all')

#tfidf特征merge
def merge_tfidf_fea(train_data):
    print train_data.shape
    userID_appCategory_1_tfidf = pd.read_hdf(FilePath+'userID_appCategory_1_tfidf')
    train_data = pd.merge(train_data,userID_appCategory_1_tfidf,on=['userID','appCategory_1'],how='left')
    del userID_appCategory_1_tfidf
    gc.collect()
    print '111'
    print train_data.shape
    userID_appCategory_tfidf = pd.read_hdf(FilePath+'userID_appCategory_tfidf')
    train_data = pd.merge(train_data, userID_appCategory_tfidf, on=['userID', 'appCategory_1', 'appCategory_2'], how='left')
    del userID_appCategory_tfidf
    gc.collect()
    print '222'
    print train_data.shape
    appID_age_tfidf = pd.read_hdf(FilePath+'appID_age_tfidf')
    train_data = pd.merge(train_data, appID_age_tfidf, on=['appID', 'age'], how='left')
    del appID_age_tfidf
    gc.collect()
    print '333'
    print train_data.shape
    appID_gender_tfidf = pd.read_hdf(FilePath + 'appID_gender_tfidf')
    train_data = pd.merge(train_data, appID_gender_tfidf, on=['appID', 'gender'], how='left')
    del appID_gender_tfidf
    gc.collect()
    print '444'
    print train_data.shape
    appID_haveBaby_tfidf = pd.read_hdf(FilePath + 'appID_haveBaby_tfidf')
    train_data = pd.merge(train_data, appID_haveBaby_tfidf, on=['appID', 'haveBaby'], how='left')
    del appID_haveBaby_tfidf
    gc.collect()
    print '666'
    print train_data.shape
    appID_marriageStatus_tfidf = pd.read_hdf(FilePath + 'appID_marriageStatus_tfidf')
    train_data = pd.merge(train_data, appID_marriageStatus_tfidf, on=['appID', 'marriageStatus'], how='left')
    del appID_marriageStatus_tfidf
    gc.collect()
    print '777'
    print train_data.shape
    appID_residence_1_tfidf = pd.read_hdf(FilePath + 'appID_residence_1_tfidf')
    train_data = pd.merge(train_data, appID_residence_1_tfidf, on=['appID', 'residence_1'], how='left')
    del appID_residence_1_tfidf
    gc.collect()
    print '999'
    print train_data.shape
    return train_data


#这次补上APPID的rank!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#标注相同userID, creativeID非重复数据0，重复但不是第一条和最后一条1，重复且是第一条2，最后一条3
def if_user_app_first_last():
    train_data = pd.read_hdf('../../gen/train_0622')
    test_data = pd.read_hdf(FilePath+'../../gen/test_0622')
    print 'read finish'
    train_data1 = train_data[['label','userID', 'appID', 'clickTime','connectionType','telecomsOperator']].drop_duplicates()
    test_data1 = test_data[['label','userID', 'appID', 'clickTime','connectionType','telecomsOperator']].drop_duplicates()
    train_test_data = pd.concat([train_data1,test_data1],axis=0)
    # train_test_data = pd.concat([train_data,test_data],axis=0)
    print 'concat finish'

    actions = train_test_data.sort_values(by=['userID', 'appID', 'clickTime'])
    actions.reset_index(inplace=True)
    del actions['index']
    # 获取user_creative操作跟上一条、下一条的时间差
    actions = get_min_click_IDS_app_time_diff(actions)
    actions = get_min_next_click_IDS_app_time_diff(actions)
    print 'time_delta user_app finish'
    actions = get_userID_appID_delta(actions)
    actions = get_userID_appID_delta_next(actions)
    actions['rank_user_app_click'] = actions['user_app_delta_up'] + actions['user_app_delta_next']
    actions['rank_user_app_click'].replace([5], 1, inplace=True)
    actions.drop(['user_app_delta_up', 'user_app_delta_next'], axis=1, inplace=True)
    print 'rank user_app finish'

    # print actions[['rank_user_app_click','time_delta_user_app','time_delta_user_app_next']]
    print actions

    train_fea = actions[actions['clickTime'] < 31000000]
    print train_fea.shape
    test_fea = actions[actions['clickTime'] >= 31000000]
    print test_fea.shape
    train_data = pd.merge(train_data, train_fea,
                          on=['label', 'userID', 'appID', 'clickTime', 'connectionType', 'telecomsOperator'],
                          how='left')
    test_data = pd.merge(test_data, test_fea,
                         on=['label', 'userID', 'appID', 'clickTime', 'connectionType', 'telecomsOperator'],
                         how='left')
    print 'train大小', train_data.shape
    print 'test大小', test_data.shape

    print 'write begin'
    test_data.to_hdf(FeaFilePath + 'test_0626', 'all')
    train_data.to_hdf(FeaFilePath + 'train_0626', 'all')


#这次补上positionID的rank!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#标注相同userID, positionID非重复数据0，重复但不是第一条和最后一条1，重复且是第一条2，最后一条3
def if_user_position_first_last():
    train_data = pd.read_hdf('../../gen/train_0626')
    test_data = pd.read_hdf(FilePath+'../../gen/test_0626')
    print 'read finish'
    train_data1 = train_data[['label','userID', 'positionID', 'clickTime','connectionType','telecomsOperator']].drop_duplicates()
    test_data1 = test_data[['label','userID', 'positionID', 'clickTime','connectionType','telecomsOperator']].drop_duplicates()
    train_test_data = pd.concat([train_data1,test_data1],axis=0)
    # train_test_data = pd.concat([train_data,test_data],axis=0)
    print 'concat finish'

    actions = train_test_data.sort_values(by=['userID', 'positionID', 'clickTime'])
    actions.reset_index(inplace=True)
    del actions['index']
    # 获取user_creative操作跟上一条、下一条的时间差
    actions = get_min_click_IDS_position_time_diff(actions)
    actions = get_min_next_click_IDS_position_time_diff(actions)
    print 'time_delta user_pos finish'
    actions = get_userID_positionID_delta(actions)
    actions = get_userID_positionID_delta_next(actions)
    actions['rank_user_position_click'] = actions['user_position_delta_up'] + actions['user_position_delta_next']
    actions['rank_user_position_click'].replace([5], 1, inplace=True)
    actions.drop(['user_position_delta_up', 'user_position_delta_next'], axis=1, inplace=True)
    print 'rank user_position finish'

    # print actions[['rank_user_app_click','time_delta_user_app','time_delta_user_app_next']]
    print actions
    
    train_fea = actions[actions['clickTime'] < 31000000]
    print train_fea.shape
    test_fea = actions[actions['clickTime'] >= 31000000]
    print test_fea.shape
    train_data = pd.merge(train_data, train_fea,
                          on=['label', 'userID', 'positionID', 'clickTime', 'connectionType', 'telecomsOperator'],
                          how='left')
    test_data = pd.merge(test_data, test_fea,
                         on=['label', 'userID', 'positionID', 'clickTime', 'connectionType', 'telecomsOperator'],
                         how='left')
    print 'train大小', train_data.shape
    print 'test大小', test_data.shape

    print 'write begin'
    test_data.to_hdf(FeaFilePath + 'test_0626_pos', 'all')
    train_data.to_hdf(FeaFilePath + 'train_0626_pos', 'all')

def time_delta_map(x):
    if x == -1:
        return 0
    elif x == 0:
        return 1
    elif 0<x<=60:
        return 2
    elif 60<x<=600:
        return 3
    elif 600<x<=3600:
        return 4
    elif 3600<x<=86400:
        return 5
    else:
        return 6

def time_delta_fentong():
    train_data = pd.read_hdf('../../gen/train_0626')
    test_data = pd.read_hdf('../../gen/test_0626')
    print 'read finish'
    train_data['time_delta_user_creative_next_fentong'] = train_data['time_delta_user_creative_next'].map(time_delta_map)
    test_data['time_delta_user_creative_next_fentong'] = test_data['time_delta_user_creative_next'].map(time_delta_map)
    train_data['time_delta_user_creative_fentong'] = train_data['time_delta_user_creative'].map(time_delta_map)
    test_data['time_delta_user_creative_fentong'] = test_data['time_delta_user_creative'].map(time_delta_map)
    train_data['time_delta_user_app_next_fentong'] = train_data['time_delta_user_app_next'].map(time_delta_map)
    test_data['time_delta_user_app_next_fentong'] = test_data['time_delta_user_app_next'].map(time_delta_map)
    train_data['time_delta_user_app_fentong'] = train_data['time_delta_user_app'].map(time_delta_map)
    test_data['time_delta_user_app_fentong'] = test_data['time_delta_user_app'].map(time_delta_map)
    train_data['time_delta_user_next_fentong'] = train_data['time_delta_user_next'].map(time_delta_map)
    test_data['time_delta_user_next_fentong'] = test_data['time_delta_user_next'].map(time_delta_map)
    train_data['time_delta_user_fentong'] = train_data['time_delta_user'].map(time_delta_map)
    test_data['time_delta_user_fentong'] = test_data['time_delta_user'].map(time_delta_map)
    print test_data

    train_data.to_hdf('../../gen/train_0626_delta_fentong','all')
    test_data.to_hdf('../../gen/test_0626_delta_fentong','all')

def convert_time(x):
    day = int(str(x)[0:2])-17
    hh = int(str(x)[2:4])-0
    mm = int(str(x)[4:6])-0
    ss = int(str(x)[6:8])-0
    return int(day*86400+hh*3600+mm*60+ss)

def cal_new_time_delta():
    train_data = pd.read_hdf('../../gen/train_0626')
    test_data = pd.read_hdf('../../gen/test_0626')
    train_data.drop(['time_delta_user_creative_next','time_delta_user_creative','time_delta_user','time_delta_user_next','time_delta_user_app_next','time_delta_user_app'],axis=1,inplace=True)
    test_data.drop(['time_delta_user_creative_next','time_delta_user_creative','time_delta_user','time_delta_user_next','time_delta_user_app_next','time_delta_user_app'],axis=1,inplace=True)
    train_data['clickTime_temp'] = train_data['clickTime']
    train_data['clickTime'] = train_data['clickTime_temp'].map(convert_time)
    test_data['clickTime_temp'] = test_data['clickTime']
    test_data['clickTime'] = test_data['clickTime_temp'].map(convert_time)
    print 'read finish'


    train_data1 = train_data[['label','userID', 'creativeID', 'positionID','appID','clickTime','connectionType','telecomsOperator']].drop_duplicates()
    test_data1 = test_data[['label','userID', 'creativeID', 'positionID','appID','clickTime','connectionType','telecomsOperator']].drop_duplicates()

    train_test_data = pd.concat([train_data1,test_data1],axis=0)
    print 'concat finish'
    actions = train_test_data.sort_values(by=['userID', 'creativeID', 'clickTime'])
    actions.reset_index(inplace=True)
    del actions['index']
    # 获取user_creative操作跟上一条、下一条的时间差
    actions = get_min_click_IDS_time_diff(actions)
    actions = get_min_next_click_IDS_time_diff(actions)
    print 'time_delta finish'

    actions = actions.sort_values(by=['userID', 'clickTime'])
    actions.reset_index(inplace=True)
    del actions['index']
    # 获取用户操作跟上一条、下一条的时间差
    actions = get_min_click_ID_time_diff(actions)
    actions = get_min_next_click_ID_time_diff(actions)
    print 'time_delta finish'

    actions = actions.sort_values(by=['userID', 'appID','clickTime'])
    actions.reset_index(inplace=True)
    del actions['index']
    # 获取用户操作跟上一条、下一条的时间差
    actions = get_min_click_IDS_app_time_diff(actions)
    actions = get_min_next_click_IDS_app_time_diff(actions)
    print 'time_delta finish'
    print actions


    train_fea = actions[actions['clickTime'] < 1209600]
    print train_fea.shape
    test_fea = actions[actions['clickTime'] >= 1209600]
    print test_fea.shape
    train_data = pd.merge(train_data, train_fea,
                          on=['label', 'userID', 'creativeID', 'positionID','appID','clickTime', 'connectionType', 'telecomsOperator'],
                          how='left')
    test_data = pd.merge(test_data, test_fea,
                         on=['label', 'userID', 'creativeID', 'positionID','appID','clickTime', 'connectionType', 'telecomsOperator'],
                         how='left')

    train_data['clickTime'] = train_data['clickTime_temp']
    train_data.drop(['clickTime_temp'],axis=1,inplace=True)
    test_data['clickTime'] = test_data['clickTime_temp']
    test_data.drop(['clickTime_temp'],axis=1,inplace=True)


    print 'train大小', train_data.shape
    print 'test大小', test_data.shape

    train_data['time_delta_user_creative_next_fentong'] = train_data['time_delta_user_creative_next'].map(time_delta_map)
    test_data['time_delta_user_creative_next_fentong'] = test_data['time_delta_user_creative_next'].map(time_delta_map)
    train_data['time_delta_user_creative_fentong'] = train_data['time_delta_user_creative'].map(time_delta_map)
    test_data['time_delta_user_creative_fentong'] = test_data['time_delta_user_creative'].map(time_delta_map)
    train_data['time_delta_user_app_next_fentong'] = train_data['time_delta_user_app_next'].map(time_delta_map)
    test_data['time_delta_user_app_next_fentong'] = test_data['time_delta_user_app_next'].map(time_delta_map)
    train_data['time_delta_user_app_fentong'] = train_data['time_delta_user_app'].map(time_delta_map)
    test_data['time_delta_user_app_fentong'] = test_data['time_delta_user_app'].map(time_delta_map)
    train_data['time_delta_user_next_fentong'] = train_data['time_delta_user_next'].map(time_delta_map)
    test_data['time_delta_user_next_fentong'] = test_data['time_delta_user_next'].map(time_delta_map)
    train_data['time_delta_user_fentong'] = train_data['time_delta_user'].map(time_delta_map)
    test_data['time_delta_user_fentong'] = test_data['time_delta_user'].map(time_delta_map)
    print test_data

    print 'test write begin'
    test_data.to_hdf('../../gen/test_0627', 'all')
    train_data.to_hdf('../../gen/train_0627', 'all')


if __name__ == '__main__':
    # gen_user_app_install_num_in_installed()
    # gen_user_install_count_previous_15_day()
    # has_user_installed_app()
    # gen_user_app_install_yestoday()
    # run_merge_fea_0614()
    # app_conversion_time_average()
    # adevertiser_conversion_time_average()
    # merge_fea_0616()
    # run_merge_fea_0614()
    # train_data = pd.read_hdf(FeaFilePath + 'train_0616_c')
    # train_data = merge_tfidf_fea(train_data)
    # train_data.to_hdf(FeaFilePath + 'train_0622','all')
    # del train_data
    # gc.collect()
    # test_data = pd.read_hdf(FeaFilePath + 'test_0616_c')
    # test_data = merge_tfidf_fea(test_data)
    # test_data.to_hdf(FeaFilePath + 'test_0622', 'all')
    # if_user_app_first_last()
    cal_new_time_delta()
