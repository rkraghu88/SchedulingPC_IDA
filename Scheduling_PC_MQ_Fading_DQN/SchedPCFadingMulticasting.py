# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.environ["TF_XLA_FLAGS"]="--tf_xla_auto_jit=16 --tf_xla_cpu_global_jit"
# os.environ["USE_DAAL4PY_SKLEARN"]="YES"

import Scheduling_PC_MQ_Fading_DQN.MQ_Scheduling_PC_Inst as MQ
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")
N_Files = 100
total_users=10
good_users=5
total_services=100000
analysis_window=10000
service_time=1
#samples=np.ceil(total_services*service_time*total_lambda)
cache_size=0
x = np.arange(1, N_Files+1)
a = 1.0001
weights = x ** (-a)
weights /= weights.sum()
bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))
lambda_vec=[.2,.4,.6,.8,1,2,3]
# lambda_vec=[.8,2]
# lambda_vec=[1]
#lambda_vec=[5]
sojourn_vec=[]
total_lambda=2
for i in range(len(lambda_vec)):
    total_lambda=lambda_vec[i]
    #print(total_lambda)
    samples=np.ceil(total_services*service_time*total_lambda)
    seq1 = bounded_zipf.rvs(size=int(samples+1))
    #inter_arrival_times= stats.expon.rvs(total_lambda, size=int(samples))
    inter_arrival_times = np.cumsum(np.random.exponential(1/(total_lambda),int(samples)))
    timelines=np.append([0], inter_arrival_times)
    users=stats.randint.rvs(0,total_users,size=int(samples+1))

    requests=np.array(seq1)
    services=0
    #FadingMQ=MQ.MulticastQueue(requests, timelines, users, service_time, total_users, cache_size)
    FadingMQ=MQ.DQNMulticastFadingQueue(requests, timelines, users, service_time, total_users, good_users,cache_size,total_services)
    ret_val=1
    while(ret_val):
        #print('--')
        #print(FadingMQ.userCaches.cache)
        ret_val=FadingMQ.acceptServeRequests()
        #print('--')
        #print(FadingMQ.userCaches.cache)
        #if ret_val==0:
        #    break
        #services=FadingMQ.services
        #print(services)
    #ST=[x for x in FadingMQ.sojournTimes if x.size>0]
    #print(total_lambda,(1-(sum(FadingMQ.userCaches.hit)/sum(FadingMQ.userCaches.requests)))*np.mean(np.concatenate(ST)))
    powSave="pow_vec_%f.txt"%total_lambda
    sojSave="soj_vec_%f.txt"%total_lambda
    lagSave="beta_vec_%f.txt"%total_lambda
    actSave="action_prob_vec_%f.txt"%total_lambda
    rewSave="reward_vec_%f.txt"%total_lambda
    np.savetxt(powSave,FadingMQ.powerVecs)
    np.savetxt(sojSave,FadingMQ.sojournTimes)
    np.savetxt(lagSave,FadingMQ.DDQNA.penalty_lambda_array)
    np.savetxt(actSave,FadingMQ.actionProbVec)
    np.savetxt(rewSave,FadingMQ.reward_array)

    print('Final_PVec:',FadingMQ.action_prob)
    print('Avg Power:',np.mean(FadingMQ.powerVecs[-1000:]))
    print(total_lambda,(1-(sum(FadingMQ.userCaches.hit)/sum(FadingMQ.userCaches.requests)))*np.mean(FadingMQ.sojournTimes[-np.min([analysis_window,FadingMQ.sojournTimes.__len__()]).astype(int):]))
    p_hit=(sum(FadingMQ.userCaches.hit)/sum(FadingMQ.userCaches.requests))
    sojourn_vec.append((1-p_hit)*np.mean(FadingMQ.sojournTimes[-np.min([analysis_window,FadingMQ.sojournTimes.__len__()]).astype(int):]))
    # sojourn_vec.append((1-p_hit)*np.mean(FadingMQ.sojournTimes[-np.round(.2*FadingMQ.sojournTimes.__len__()).astype(int):]))
    print(FadingMQ.service_vecs, 'Imitation Times:',FadingMQ.imit_times)
    #plt.plot(range(FadingMQ.DQNA.reward_array.__len__()),FadingMQ.DQNA.reward_array)
    #plt.show()
plt.plot(lambda_vec,sojourn_vec)
    #ST=[x for x in FadingMQ.sojournTimes if x.size>0]
    #print(total_lambda,(1-(sum(FadingMQ.userCaches.hit)/sum(FadingMQ.userCaches.requests)))*np.mean(np.concatenate(ST)))
    #sojourn_vec.append(np.mean(np.concatenate(ST)))
#plt.plot(lambda_vec,sojourn_vec)
plt.show()
