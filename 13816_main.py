# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 23:43:36 2020

@author: jungl
"""

from RL_brain import DuelingDQN
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

tf.set_random_seed(1)
np.random.seed(1)

def D2Genome(trainset,genome):
    ind = np.where(trainset[1] == genome[:,0])
    rat = np.append(trainset[0],trainset[2])
    d = np.append(genome[ind,1:],rat).reshape(1,1130)
    return d

def loadData():
    from sklearn.externals import joblib
    with open('./svd_13816.p', 'rb') as fp:
        model = joblib.load(fp)
    trainset = np.load('./rat_13816_5000_l3_d10.npy')
    label = np.load('./onehot_13816.npy')
    genome = np.load('./mid_gen.npy')
    mclass = np.load('./resultc.npy',allow_pickle='TRUE').item()
    return model,trainset,label,genome,mclass

def D2Label(mid,label):
    mi = np.where(mid == label[:,0])
    ml = label[mi]
    return ml[0,1:]

def nextState(a,r,user,label):
    s_d = np.zeros((3),dtype=np.float32)
    s_d[0] = user
    s_d[1] = label[a[1],0]
    s_d[2] = r[0]
    return s_d

def get_dcg(r):
    d,dcg = 0,0  
    for i in range(3):
        d = (2 ** r[i] - 1) / np.log2(1+(i+1))
        dcg += d
    return dcg

def nDCG(r):
    r = np.square(r)
    dcg = get_dcg(r)
    ir = r[np.argsort(-r)]
    idcg = get_dcg(ir)
    ndcg = dcg / idcg
    return ndcg

def topkAcc(label,s_ml,a):
    A1 = np.logical_and(s_ml, label[a[0],1:])
    #A2 = np.logical_and(s_ml, label[a[1],1:])
    #A3 = np.logical_and(s_ml, label[a[2],1:])
    if (np.sum(A1 == 1))  >= 1: return 1
    else: return 0

    
def a2A(a,s,mclass,test):
    if test == 1:
        a0 = np.argsort(-np.linalg.norm(mclass[a[0]][:,1:] - s[0,0:-2],ord=1,keepdims=True,axis=1).reshape(-1))[-1:][::-1]
        a1 = np.argsort(-np.linalg.norm(mclass[a[1]][:,1:] - s[0,0:-2],ord=1,keepdims=True,axis=1).reshape(-1))[-1:][::-1]
        a2 = np.argsort(-np.linalg.norm(mclass[a[2]][:,1:] - s[0,0:-2],ord=1,keepdims=True,axis=1).reshape(-1))[-1:][::-1]
    else:
        a0 = np.random.choice(-np.argsort(np.linalg.norm(mclass[a[0]][:,1:] - s[0,0:-2],ord=1,keepdims=True,axis=1).reshape(-1))[-2:][::-1])
        a1 = np.random.choice(-np.argsort(np.linalg.norm(mclass[a[1]][:,1:] - s[0,0:-2],ord=1,keepdims=True,axis=1).reshape(-1))[-2:][::-1])
        a2 = np.random.choice(-np.argsort(np.linalg.norm(mclass[a[2]][:,1:] - s[0,0:-2],ord=1,keepdims=True,axis=1).reshape(-1))[-2:][::-1])
    return np.array([int(mclass[a[0]][a0,0]),int(mclass[a[1]][a1,0]),int(mclass[a[2]][a2,0])])

def rewardCalu(user,a,label,model,s_ml):
    r = np.array([0,0,0],dtype=np.float32)
    hit,reward,ctr,fail = 0,0,0,False
    ai = [np.where(label[:,0] == a[0]),np.where(label[:,0] == a[1]),np.where(label[:,0] == a[2])]
    for i in range(3):
        r[i] = model.predict(user, a[i])[3]
        if r[i] >= 4:
            hit += 10
            ctr += 1
        if r[i] >= 3 and r[i] < 4:
            hit += 1  
            ctr += 1
        if r[i] < 3:
            hit += 0
    ndcg = nDCG(r)
    topka = topkAcc(label,s_ml,ai)
    ctr = ctr/3
    if r[0] < 3 or r[1] < 3 or r[2] < 3:
        fail = True
        reward = (hit + ndcg*10 + topka*10 + ctr*10)*0.5
    else:
        reward = (hit + ndcg*10 + topka*10 + ctr*10)
    s_ = nextState(ai,r,user,label)
    return s_,reward,ctr,ndcg,topka,fail

MEMORY_SIZE = 500
ACTION_SPACE = 1128
N_FEATURES = 1130
MAX_step = 10

sess = tf.Session()

with tf.variable_scope('dueling'):
    dueling_DQN = DuelingDQN(
        n_actions=ACTION_SPACE, n_features=N_FEATURES, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, sess=sess, dueling=True, output_graph=True)

sess.run(tf.global_variables_initializer())

def train(RL,model,dataset,label,genome,mclass):
    tacc_r,tctr_r,tndcg_r,ttopka_r,step_r,stepc = [],[],[],[],[],0
    tacc_r,stepc = [],0
    np.random.shuffle(dataset)
    trainset,testset = train_test_split(dataset,test_size=0.5)
    #testset,validateset = train_test_split(testdataset,test_size=0.5)
    for episode in range(len(trainset)):
        total_steps = 1
        s_d = trainset[episode]
        s = D2Genome(s_d,genome)
        s_user = s[0,-2]
        s_mid = trainset[episode,1]
        while True:      
            s_ml = D2Label(s_mid,label)
            a = RL.choose_action(s,0)
            A = a2A(a,s,mclass,0)
            s_,r,ctr,ndcg,topka,fail = rewardCalu(s_user,A,label,model,s_ml)
            s_ = D2Genome(s_,genome) 
            RL.store_transition(s[0], a[0], r, s_[0])
            if total_steps > MAX_step:
                RL.learn()
                stepc += total_steps
                break
            if fail == True:
                RL.learn()
                stepc += total_steps
                break
            s = s_
            total_steps += 1
        if episode % 500 == 0:
            step_r.append(stepc/500)
            ttr,ttctr,tttopk,ttndcg,tc,stepc = 0,0,0,0,0,0
            ttr,tc = 0,0
            for te in range(200):
                ts_d = testset[te+(tc*200)]
                ts = D2Genome(ts_d,genome)
                ts_user = ts[0,-2]
                ts_mid = testset[te,1]
                ts_ml = D2Label(ts_mid,label)
                ta = RL.choose_action(ts,1)
                tA = a2A(ta,s,mclass,1)
                ts_,tr,tctr,tndcg,ttopka,tfail = rewardCalu(ts_user,tA,label,model,ts_ml)
                ttr += tr 
                ttctr += tctr
                tttopk += ttopka
                ttndcg += tndcg  
            tacc_r.append(ttr/200)
            tctr_r.append(ttctr/200)
            tndcg_r.append(ttndcg/200)
            ttopka_r.append(tttopk/200)
            print("episode:",episode,",reward:",tacc_r[-1],",ctr:",tctr_r[-1],",ndcg:",tndcg_r[-1],",topk:",ttopka_r[-1],",step:",step_r[-1])
            tc += 1
    return RL.cost_his, tacc_r, tctr_r, tndcg_r, ttopka_r, step_r

model,trainset,label,genome,mclass = loadData()
c_dueling, t_r, t_ctr, t_ndcg, t_topk, t_step = train(dueling_DQN,model,trainset,label,genome,mclass)

plt.figure(1)
plt.plot(np.array(c_dueling), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('cost')
plt.xlabel('training steps')
plt.grid()

plt.figure(2)
plt.plot(np.array(t_r), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('test reward')
plt.xlabel('training steps')
plt.grid()

plt.figure(3)
plt.plot(np.array(t_ctr), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('test ctr')
plt.xlabel('training steps')
plt.grid()

plt.figure(4)
plt.plot(np.array(t_ndcg), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('test ndcg')
plt.xlabel('training steps')
plt.grid()

plt.figure(5)
plt.plot(np.array(t_topk), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('test topk')
plt.xlabel('training steps')
plt.grid()

plt.figure(6)
plt.plot(np.array(t_step), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('train avg step')
plt.xlabel('training steps')
plt.grid()

plt.show()