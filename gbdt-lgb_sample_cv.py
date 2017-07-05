#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle
import math
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, log_loss
import lightgbm as lgb
import gc
import warnings
warnings.filterwarnings('ignore')
def drop_feature(train):
    drop_list = ['conversionTime']
    train.drop(drop_list,axis=1,inplace=True)
    return train
def get_params():
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 280,
        'learning_rate': 0.05,
        'feature_fraction': 0.90,
        'bagging_fraction': 1,
        'bagging_freq': 5,
        'verbose': 0,
        'min_gain_to_split':1,
        'lambda_l1':1
    }
    num_boost_round = 1000
    early_stopping_rounds = 20
    return params, num_boost_round, early_stopping_rounds
def predict(gbm,test):
    return gbm.predict(test,num_iteration = gbm.best_iteration)
def sample_0(train,w):
    print '下采样开始'
    temp = train[['label']]
    msk = ((np.random.rand(len(train)) <= w) | (train['label']==1))
    train = train[msk]
    print '下采样完毕'
    return train
def modelfit(train,train_label,valid,valid_label,c):
    lgb_train = lgb.Dataset(train.values,
                            label=train_label,
                            feature_name=list(train.columns))
    lgb_eval = lgb.Dataset(valid.values,
                           label=valid_label.values,
                           reference=lgb_train,
                           feature_name=list(train.columns))
    params = get_params()
    gbm = lgb.train(params[0],
                    lgb_train,
                    num_boost_round=params[1],
                    valid_sets=lgb_eval,
                    early_stopping_rounds=params[2]
                    )
    feature_score = pd.DataFrame(gbm.feature_importance(), columns=['score'])
    feature_score['feature'] = train.columns
    feature_score = feature_score.sort_values(by=['score'], ascending=False)
    feature_score.to_csv('feature_score' + str(c) + '.csv', index=False)
    return gbm
def train_timgo(train,test,w):
    
    #训练集
    train = train[train['clickTime']<30000000]
    #测试集
    pred_temp = test[['instanceID']]
    pred = test[['instanceID']]
    pred['prob']=0
    test.drop(['instanceID','clickTime'],axis=1,inplace=True)
    repeat_num = 5
    count = 0
    valid = train.tail(n=int(train.shape[0]*0.2))
    valid_label = valid[['label']]
    valid_label['prob']=0
    valid.drop(['label', 'clickTime'], axis=1, inplace=True)

    train = train.head(n=int(train.shape[0]*0.8))
    
    train.drop(['clickTime'], axis=1, inplace=True)
    
    while count < repeat_num:
        #本次验证集
        # msk = np.random.rand(len(train)) < 0.2
        # valid = train[msk]
        # #本次训练集
        # train_sample = train[~msk]
        # print train_sample
        train_sample = train.sample(frac=0.95)
        # train_sample = sample_0(train,w)
        train_label = train_sample['label']
        del train_sample['label']
        print '验证第',count,'次',train_sample.shape,valid.shape
        #训练集label
        gbm = modelfit(train_sample,train_label,valid,valid_label['label'],count)
        prob = pd.DataFrame()
        prob['temp'] = predict(gbm, valid)
        # prob['temp'] = prob['temp'].map(lambda x: x / (x + (1 - x) / w))
        valid_label['prob'] += prob['temp'].values
        print '验证第',count,'次 logloss', log_loss(y_true=valid_label['label'].values, y_pred=prob['temp'].values)
        print '它的offline,mean', prob['temp'].mean()
        
        pred_temp['prob']= predict(gbm, test)
        # pred_temp['prob']= pred_temp['prob'].map(lambda x: x / (x + (1 - x) / w))
        print '它的online,mean', pred_temp['prob'].mean()
        pred['prob'] +=pred_temp['prob'].values
        count+=1
        pred_temp['prob'] = pred['prob']/count
        pred_temp['prob'].to_csv('submission'+str(count)+'.csv',index=False)
        
        
    pred['prob']=pred['prob']/count
    valid_label['prob'] = valid_label['prob']/count
    print '线下平均 logloss', log_loss(y_true=valid_label['label'].values, y_pred=valid_label['prob'].values)
    print 'online,mean', pred['prob'].mean()
    pred.to_csv('submission.csv', index=False)
path = ''
w = 0.05
# print '读train中'

# app_vec = pd.read_csv('../gen/app_vec.csv')
# train = pd.read_hdf(path+'train_timego_em')
# train = train.reset_index()
# del train['index']
# # train = train.sample(frac=0.1)
# train = pd.merge(train,app_vec,on=['appID'],how='left')
# train.fillna(0,inplace=True)
# train.to_hdf('train_temp','all')

# print 'train读完'

# test = pd.read_hdf(path+'test_timego_em')
# # test = test.sample(frac=0.1)
# test = pd.merge(test,app_vec,on=['appID'],how='left')
# test.fillna(0,inplace=True)
# test.to_hdf('test_temp','all')


# train = pd.read_hdf('train_temp2')
# print 'read_finshed_1'
# test = pd.read_hdf('test_temp2')
# print 'read_finshed_2'

# train_e = pd.read_hdf('train_timego_ewai2')
# train_e = train_e.reset_index()
# del train_e['index']
# test_e = pd.read_hdf('test_timego_ewai2')

# trian_yao = pd.read_hdf('train_timego_yaobao')
# trian_yao = trian_yao.reset_index()
# del trian_yao['index']
# test_yao = pd.read_hdf('test_timego_yaobao')

# print 'concatting'

# # test_e = test_e.reset_index()
# # del test_e['index']
# # test_yao = test_yao.reset_index()
# # del test_yao['index']

# print train.shape,train_e.shape

# train = pd.concat([train,train_e],axis=1)
# test = pd.concat([test,test_e],axis=1)

# train = pd.concat([train,trian_yao],axis=1)
# test = pd.concat([test,test_yao],axis=1)


# # train.to_hdf('train')
# # test.to_hdf('test')

# del train_e
# del test_e
# del trian_yao
# del test_yao
# gc.collect()

# train.to_hdf('train_ttt','all')
# test.to_hdf('test_ttt','all')
train = pd.read_hdf('train_ttt')
test = pd.read_hdf('test_ttt')
# del train['userID']
# del test['userID']
print 'fffff'
drop1=['count_positionID','appPlatform',
# 'time_delta_user_position_next',
# 'rate_rank_user_position_click_positionID',
# 'rate_rank_user_creative_click_positionID_connectionType',
# 'active_rank_user_position_click_positionID',
# 'rank_user_position_click',
# 'count_rank_user_position_click_positionID_connectionType'
]

train.drop(drop1,axis=1,inplace=True)
test.drop(drop1,axis=1,inplace=True)


print test.columns
print train.columns

print train.shape,test.shape
train_timgo(train,test,w)
