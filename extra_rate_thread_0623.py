#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import math
import pickle
import threading
import gc
import collections
import scipy.special as special
import os
class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        sample = numpy.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return I, C

    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        '''estimate alpha, beta using fixed point iteration'''
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        '''fixed point iteration'''
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        sumfenzialpha = (special.digamma(success+alpha) - special.digamma(alpha)).sum()
        sumfenzibeta = ((special.digamma(tries - success + beta) - special.digamma(beta))).sum()
        sumfenmu = ((special.digamma(tries+alpha+beta) - special.digamma(alpha+beta))).sum()

        # for i in range(len(tries)):
        #     sumfenzialpha += (special.digamma(success[i]+alpha) - special.digamma(alpha))
        #     sumfenzibeta += (special.digamma(tries[i]-success[i]+beta) - special.digamma(beta))
        #     sumfenmu += (special.digamma(tries[i]+alpha+beta) - special.digamma(alpha+beta))
        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)

    def update_from_data_by_moment(self, tries, success):
        '''estimate alpha, beta using moment estimation'''
        mean, var = self.__compute_moment(tries, success)
        #print 'mean and variance: ', mean, var
        #self.alpha = mean*(mean*(1-mean)/(var+0.000001)-1)
        self.alpha = (mean+0.000001) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)
        #self.beta = (1-mean)*(mean*(1-mean)/(var+0.000001)-1)
        self.beta = (1.000001 - mean) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)

    def __compute_moment(self, tries, success):
        '''moment estimation'''
        ctr_list = []
        # var = 0.0
        mean = (success/tries).mean()
        if len(tries)==1:
            var = 0
        else:
            var = (success/tries).var()
        # for i in range(len(tries)):
        #     ctr_list.append(float(success[i])/tries[i])
        # mean = sum(ctr_list)/len(ctr_list)
        # for ctr in ctr_list:
        #     var += pow(ctr-mean, 2)

        return mean, var

def gen_appCategory(train):
	train['appCategory_1']=train['appCategory']/100
	train['appCategory_1'] = train['appCategory_1'].astype('int')
	train['appCategory_2'] = train['appCategory']%100
	return train
key_list = [
    ['appID','positionID','connectionType'],
    ['haveBaby','gender','appID'],
    ['userID','positionID'],
    ['advertiserID', 'connectionType'],
    ['appPlatform', 'positionType'],
    ['positionID', 'advertiserID'],
    ['userID','sitesetID'],
    ['creativeID'],
    ['positionID'],

    ['positionID', 'positionType'],
    ['positionID', 'gender'],
    ['hometown_1', 'residence_1'],
    ['gender', 'education'],
    ['positionID', 'marriageStatus'],
    ['age', 'marriageStatus'],
    ['positionID', 'age'],
    ['positionID', 'appID'],
    ['positionID', 'hometown_1'],
    ['positionID', 'telecomsOperator'],
    ['positionID', 'creativeID'],
    ['positionID', 'education'],
    ['camgaignID', 'connectionType'],
    ['positionID', 'connectionType'],
    ['creativeID', 'connectionType'],
    ['creativeID', 'gender'],
    ['positionID', 'camgaignID'],
    ['age',  'gender'],
    ['camgaignID', 'age' ],
    ['adID', 'connectionType'],
    ['camgaignID', 'gender'],
    
    ['positionID', 'adID'],
    ['positionID', 'appCategory_1', 'appCategory_2'],
    ['positionID', 'haveBaby'],
    ['residence_1', 'appCategory_1', 'appCategory_2'],
    ['appID'],
    ['advertiserID'],
    ['residence_1'],
    ['hometown_1'],
    ['camgaignID'],
    

    ['appCategory_1', 'appCategory_2'],
    ['advertiserID','appCategory_1','appCategory_2'],
    ['age','gender','appID'],
    ['hometown_1', 'residence_1','positionID'],
    ['clickTime_HH','positionID','connectionType'],
    ['clickTime_HH','appID'],
    ['clickTime_HH','age'],
    ['clickTime_HH','haveBaby'],
    ['clickTime_HH'],
    
    
    ['userID','connectionType'],
    ['userID','appID'],
    ['userID','appPlatform'],
    
]


key_today=[
    ['userID'],
    ['creativeID'],
    ['appID'],
    ['userID','appID'],
    ['userID','positionID'],
    ['camgaignID', 'connectionType']
]

key_huoyue_list = [
['userID','positionID'],
['userID','creativeID'],
['userID','appID'],
['positionID','userID'],
['creativeID','userID'],
['appID','userID'],
['appID','positionID'],
['positionID','appID'],
]

fea_list=[]
fea_key_list=[]
fea_num_list=[]
today_finshed=[]
lock = threading.Lock()#互斥锁

def merge_thread(today,if_first,if_train):
    count=0
    while True:
        while len(fea_list)>0:
            today = pd.merge(today,fea_list[0],on=fea_key_list[0],how='left')
            print 'merged', count, len(fea_list)
            fea_list.pop(0)
            fea_key_list.pop(0)
            count+=1
            gc.collect()
        if len(today_finshed)==1:
            break
    filename=''
    print list(today.columns)
    today=drop_base(today,if_train)
    # print today
    print list(today.columns)
    print today.shape
    if if_train:
        filename='train_timego_0624'
    else:
        filename = 'test_timego_0624'
    if if_first:
        # today.to_csv(filename,index=False,mode='w')
        today.to_hdf(filename,'all', format='table')
    else:
        # today.to_csv(filename,index=False,header=None,mode='a')
        today.to_hdf(filename,'all', format='table',append='True')
    today_finshed.append(0)
    # if len(today_finshed)==2:
    #     clear_list()
def count_today_one(today):
    for key in key_today:
        count_today='count_today'
        for ele in key:
            count_today=count_today+'_'+ele
        fea = today[key]
        fea[count_today]=1
        fea = fea.groupby(key).agg('sum').reset_index()
        today = pd.merge(today,fea,on=key,how='left')
    return today
def clear_list():
   while len(today_finshed)>0:
       today_finshed.pop(0)
def add_ele(key,fea):
    lock.acquire()
    fea_key_list.append(key)
    fea_list.append(fea)
    fea_num_list.pop()
    lock.release()
def extra_rate_active(train,key):
	# print '开始生rate,active'
	key2 = []
	key_string = ''
	for e in key:
		key2.append(e)
		key_string = key_string + '_' + e
	key2.append('label')
	fea = train[key2]
	active = 'active'+key_string
	count = 'count' + key_string
	rate = 'rate' + key_string
	fea[count] = 1
	fea = fea.groupby(key).agg('sum').reset_index()
       #平滑 
        hyper = HyperParam(1,1)
        I = fea[count]
        C = fea['label']
        # hyper.update_from_data_by_FPI(I, C, 1000, 0.00001)#000
        hyper.update_from_data_by_moment(I, C)
        print key_string,hyper.alpha, hyper.beta
        fea[rate] = (hyper.alpha + fea['label'])/(hyper.alpha + hyper.beta + fea[count])
	# fea[rate] = fea['label'] / fea[count]
        
        fea[active]=fea['label']

	fea.drop(['label',count], axis=1, inplace=True)
	fea[active].replace(0, 0.5, inplace=True)
	fea[active] = np.log2(fea[active])


	add_ele(key,fea)
def extra_count(train_8day,key):
	key_string = ''
	for e in key:
		key_string = key_string + '_' + e
	# print '开始生count'
	# today+before 7
	fea = train_8day[key]
	count = 'count' + key_string
	fea[count]=1
	fea = fea.groupby(key).agg('sum').reset_index()
	fea[count].replace(0, 0.1, inplace=True)
	fea[count] = np.log10(fea[count])
	# 生成完事儿之后，放入一个list里
	add_ele(key,fea)

# 只保留新特征
def drop_base(train,if_train):
    retain = []
    retain = ['clickTime_HH',
     'time_delta_user', 
     'time_delta_user_creative',
    'time_delta_user_next', 
    'time_delta_user_creative_next', 
    'rank_user_click', 
    'rank_user_creative_click', 
    'user_install_app_count_before', 
    'app_install_user_count_before', 
    'app_install_count_previous15', 
    'user_install_count_previous15',
    'user_app_before_has_installed',
    'creativeID','positionID','adID','camgaignID','advertiserID',
    'appID','appPlatform','positionType','appCategory_1','app_install_count_yestoday',
    'acvertiser_conversion_time_average','app_conversion_time_average',
    'userID_appCategory_tfidf','appID_age_tfidf','appID_gender_tfidf',
]
    for key in key_huoyue_list:
    	key1 = key[0]
    	key2 = key[1]
    	name1 = key1+'_click_diiff_'+key2+'_num'
    	name2 = key1+'_active_diiff_'+key2+'_num'
    	retain.append(name1)
    	retain.append(name2)
    for key in key_today:
        count_today = 'count_today'
        for e in key:
            count_today = count_today + '_' + e
        retain.append(count_today)
    for key in key_list:
        key_string = ''
        for e in key:
            key_string = key_string + '_' + e
        count = 'count' + key_string
        rate = 'rate' + key_string
        active = 'active'+key_string
        retain.append(active)
        retain.append(count)
        retain.append(rate)
    retain.append('clickTime')
    if if_train:
        retain.append('label')
    else:
        retain.append('instanceID')
    train = train[retain]
    return train
def count_today(today,train,train_8day,if_first,if_train):
    #开一个merge+write线程，每生成一个key时，merge。
    # 这儿写错了，想一下到底开一个线程merge还是多个
    t_merge = threading.Thread(target=merge_thread,args=(today,if_first,if_train))
    t_merge.start()
    for key in key_list:
        # print 'haha',key
        # 对每个key开一个线程
        fea_num_list.append(0)
        t1 = threading.Thread(target=extra_rate_active,args=(train,key))
        t1.start()
        fea_num_list.append(0)
        t2 = threading.Thread(target=extra_count,args=(train_8day,key))
        t2.start()
    while len(fea_list)>0 or len(fea_num_list)>0:#全都merge完&所有fea都生完
        pass
    today_finshed.append(0)
def count_huoyue_active(train,today):
    for key in key_huoyue_list:
	key1 = key[0]
	key2 = key[1]
	data1 = train[train['label']==1]
	data1 = data1[key].drop_duplicates()
	data1 = data1.groupby(key1).size().reset_index()
	data1.columns = [key1,key1+'_active_diiff_'+key2+'_num']
	today = pd.merge(today,data1,on=key1,how='left')
    #merge to today
    return today
def count_huoyue_click(train,today):
    for key in key_huoyue_list:
        key1 = key[0]
        key2 = key[1]
        data = train[key].drop_duplicates()
        data = data.groupby(key1).size().reset_index()
        data.columns = [key1,key1+'_click_diiff_'+key2+'_num']
        today = pd.merge(today,data,on=key1,how='left')
    #merge to today
    return today

def for_train(train_all):
    start_before=17
    end_before = 28
    start_today=28
    end_today=29
    count=0
    if_first=True
    while start_today<30:
        #拿到七天前数据
        train = train_all[(train_all['clickTime']>=start_before*1000000)&(train_all['clickTime']<end_before*1000000)]
        #拿到当天数据
        today = train_all[(train_all['clickTime']>=start_today*1000000)&(train_all['clickTime']<end_today*1000000)]
        # 拿到包括本天的8天数据
        train_8day = train_all[(train_all['clickTime']>=start_before*1000000)&(train_all['clickTime']<end_today*1000000)]
        #统计当天
        today = count_today_one(today)
        # print today.dtypes
        print '统计',start_today,end_today,start_before,end_before
        print 'today.shape,before.shape',today.shape,train.shape
        today = count_huoyue_active(train,today)
        today = count_huoyue_click(train_8day,today)
        count_today(today, train,train_8day,if_first,True)
        while len(today_finshed)<2:
            pass
        clear_list()
        start_before+=1
        end_before+=1
        start_today+=1
        end_today+=1
        if_first=False
def for_test(train,train_8day,test):
    clear_list()
    # 统计当天
    test = count_today_one(test)
    test = count_huoyue_active(train,test)
    test = count_huoyue_click(train_8day,test)
    count_today(test, train, train_8day,True,False)
    while len(today_finshed) < 2:
        pass
path = '../gen/'

train = pd.read_hdf(path+'train_0622')
print 'read finshed'
train['appID'] = train['appID'].astype('int')
train['clickTime_HH'] = train['clickTime_HH'].astype('int')
# train = train.sample(frac=0.001)
train = train.reset_index()
del train['index']
print train.dtypes
print train.shape

for_train(train)
# print 'train finshed'
gc.collect()
train = train[(train['clickTime']>=20000000)&(train['clickTime']<31000000)]
#读test
test=pd.read_hdf(path+'test_0622')

test['appID'] = test['appID'].astype('int')
test['clickTime_HH'] = test['clickTime_HH'].astype('int')
# train_8day = pd.concat([train,test],axis=0)
train_8day = pd.read_hdf('../mol/train_8day')
gc.collect()
for_test(train,train_8day,test)

# cmd = 'nohup python gbdt-lgb_sample_cv.py > fuck.out &'
# os.system(cmd)
