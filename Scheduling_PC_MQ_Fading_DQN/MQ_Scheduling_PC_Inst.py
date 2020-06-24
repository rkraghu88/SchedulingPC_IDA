import random
import numpy as np
import scipy.stats as stats
import Scheduling_PC_MQ_Fading_DQN.dqn_WCNC as dqn
# import DDPG.ddpg_custom as ddpgc
import matplotlib.pyplot as plt
import Scheduling_PC_MQ_Fading_DQN.AdamOpt as AdamOpt
import Scheduling_PC_MQ_Fading_DQN.NeuralApprox as NA
# import gym
from collections import deque
from statistics import mean

# import pickle

# import tensorflow as tf
class Elements:
    def __init__(self,r_file,r_user,r_time):
        self.req_file = np.array(r_file)
        self.set_of_users = np.array(r_user)
        self.time_of_request=np.array(r_time)

class LRU_MQ_Cache:
    def __init__(self, cache_size, L): #L is the total number of users
        self.cache_size = [cache_size for i in range(L)]
        self.hit = [0 for i in range(L)]
        self.requests = [0 for i in range(L)]
        self.cache = [deque(maxlen=cache_size) for i in range(L)]
        #self.user_ID = user_ID
        self.fwd_requests = [set([]) for i in range(L)]
        self.L=L
        self.fwd_temp= [set([]) for i in range(self.L)]

    # This function serves the requests if the file is in cache and updates the cache,
    # if the file is not available then forwards the same is forwarded.
    def requestsToCache(self, requests=np.array([]), users=np.array([])):
        fwd_index=[] #indices of requests forwarded
        self.fwd_temp= [set([]) for i in range(self.L)]
        for i in range(requests.size):
            Cind=int(users[i])
            element = int(requests[i])

            self.requests[Cind] += 1
            curr_length = len(self.cache[Cind])
            if curr_length == self.cache_size[Cind]:
                if element in self.cache[Cind]:
                    self.cache[Cind].remove(element)
                    self.hit[Cind] += 1
                    self.cache[Cind].appendleft(element)
                else:
                    self.fwd_temp[Cind].add(element)
                    fwd_index.append(i)
                    #self.fwd_requests.append(element)
            else:
                if element in self.cache:
                    self.cache[Cind].remove(element)
                    self.hit[Cind] += 1
                    self.cache[Cind].appendleft(element)
                else:
                    self.fwd_temp[Cind].add(element)
                    fwd_index.append(i)
        return fwd_index
    def MQ2Cache(self,element_in_service):
        if element_in_service.req_file.size==0:
            element={0}
        else:
            element=set(element_in_service.req_file.tolist())
        #print(element, element_in_service.req_file)
        for Cind in range(self.L):
            if (element).issubset(self.fwd_requests[Cind]):
                if self.cache[Cind].__len__()==self.cache_size:
                    self.cache[Cind].pop()
                self.cache[Cind].appendleft(list(element)[0])
                self.fwd_requests[Cind].remove(list(element)[0])
            self.fwd_requests[Cind]=self.fwd_requests[Cind].union(self.fwd_temp[Cind])
            self.fwd_temp[Cind]=set([])

class DQNMulticastFadingQueue:
    def __init__(self, requests, timelines, users, service_time, total_users, total_good_users, cache_size,total_services):
        self.requests = np.array(requests)
        self.timelines = np.array(timelines)
        self.users= np.array(users)
        self.total_users=total_users
        self.total_good_users=total_good_users
        self.total_bad_users=total_users-total_good_users
        self.queue = deque()  #Multicast Queue
        self.defer_queue=np.array([])

        self.noise_power=1
        self.bandwidth=10 #MHz
        self.rate=10 #Mbps

        self.service_time = service_time
        self.serve_start_index = 0
        self.serve_start_time = 0
        self.serve_stop_time = self.serve_start_time + self.service_time
        self.sojournTimes=np.array([])
        self.powerVecs=np.array([])
        self.powerVecsPolicy=np.array([7])
        self.element_in_service = Elements([],[],[])
        self.userCaches=LRU_MQ_Cache(cache_size, total_users)
        self.services = 0
        self.servicable_users=np.array([])


        #DQN Parameters
        self.enable_ddpg=0
        self.enable_sch=1
        self.retransmit_no=1
        self.stop_sch_training=0
        self.inputvector=[]
        self.LoopDefState=np.array([])
        self.act_dist=[]
        self.queue_window=1 # represents total actions, state dimension is this*5: See AutoEncoder
        self.service_vecs = [0 for i in range(self.queue_window)]
        self.TransLoopDefer_vec=[0,0,0]
        self.schWindow=100
        self.schTime=0
        self.state_memory=deque(maxlen=10000)
        self.target_memory=deque(maxlen=10000)
        self.starting_vector=np.random.randint(0,self.schWindow,size=(self.schWindow,3))
        self.starting_vector=np.divide(self.starting_vector,self.starting_vector.sum(axis=1).reshape(self.schWindow,1))
        self.reward_window_sch=deque(maxlen=10000)
        self.reward_sch=0
        self.reward_window=deque(maxlen=10000) #Holds last 500 sojourn times
        self.power_window=deque(maxlen=1000) #Holds a maximum of 1000 power actions
        self.max_power=20
        self.avg_power_constraint=7
        self.transmit_power=self.avg_power_constraint
        self.power_beta=1/self.avg_power_constraint
        self.eta_beta=.00001   #.0005 working
        self.tau_ddpg=.01   #.001 working
        self.AdamOpt=AdamOpt.AdamOpt(step=self.eta_beta)
        self.sojournTimes_window_avg=np.array([])
        self.LoopDefWindow=1
        self.action=0
        self.action_prob=np.array([1, 0, 0])
        self.actionProbVec=np.array([])
        #self.ddpg_action_prob=np.array([-1,1,-1,self.transmit_power*2/self.max_power-1])
        # self.ddpg_action_prob=np.array([self.transmit_power*2/self.max_power-1])
        self.reward=0
        self.ddpg_action_prob=np.array([0])
        #self.DQNA = dqn.DQNAgent(int(self.queue_window*5+self.total_users), int(self.queue_window))
        self.imit_decay=1/2500
        self.imitate_prob=1/(1+self.imit_decay*np.arange(np.round(total_services*1.5).astype(int))) #Number inside the arange is larger tham the simulation time
        self.imit_choose=np.random.binomial(1,self.imitate_prob)
        self.fading_samples=np.random.exponential(1,(total_users,np.round(total_services*1.5).astype(int)))
        self.fading_samples[int(total_good_users):int(total_users)]=0.1*self.fading_samples[int(total_good_users):int(total_users)] #bad user fading states
        self.imit_times=0
        #self.Autoencode()
        self.queue_decision=1
        [self.AutoEncoderLoopDef() for i in range(0,self.LoopDefWindow)]
        self.action_vector=np.array([1,2,4,6,8,10,12,14,16,18,20,25,30,40,50])
        # self.DDPGA = ddpgc.DDPG(self.ddpg_action_prob.size, self.LoopDefState.shape,1,1,50,lr=.05,tau=self.tau_ddpg)
        self.DDQNA = dqn.DQNAgent(self.LoopDefState.size,self.action_vector.size,self.avg_power_constraint,self.action_vector)
        self.DNN=NA.DNNApproximator((1,3),1,.01,.01)
        self.reward_array=np.array([])
        self.first=0
        self.curr_state=self.LoopDefState
        self.next_state=self.LoopDefState
        self.LoopDefState=np.array([])
        self.time=0
    # This function serves the requests and updates the cache
    def live_plotter(self,x_vec,y1_data,identifier='',pause_time=0.1):
        if self.first==0:
            # this is the call to matplotlib that allows dynamic plotting
            plt.ion()
            #fig = plt.figure(figsize=(13,6))
            #ax = fig.add_subplot(111)
            # create a variable for the line so we can later update it
            self.line, = plt.plot(x_vec,y1_data,'-o',alpha=0.8)
            #update plot label/title
            plt.ylabel('Reward')
            plt.title('Title: {}'.format(identifier))
            plt.show()
            self.first=1
        else:
            # after the figure, axis, and line are created, we only need to update the y-data
            self.line.set_ydata(y1_data)
            self.line.set_xdata(x_vec)
            # adjust limits if new data goes beyond bounds
            if np.min(y1_data)<=self.line.axes.get_ylim()[0] or np.max(y1_data)>=self.line.axes.get_ylim()[1]:
                plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
            if np.min(x_vec)<=self.line.axes.get_xlim()[0] or np.max(x_vec)>=self.line.axes.get_xlim()[1]:
                plt.xlim([np.min(x_vec)-np.std(x_vec),np.max(x_vec)+np.std(x_vec)])
            # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
            plt.pause(pause_time)

    def thresholdPowerIndex(self,j):
        thresh=(self.noise_power/self.transmit_power)*(2**(self.rate/self.bandwidth)-1)
        #self.servicable_users=[i for i in range(self.queue[j].set_of_users.size) if self.fading_samples[self.queue[j].set_of_users[i],self.services]>=thresh]
        #print(self.queue[j].set_of_users,self.services)
#        print(self.queue.__len__())
        if self.queue.__len__()>0:
            self.queue[j].set_of_users=np.array(self.queue[j].set_of_users)
            if self.queue[j].set_of_users.size > 0:
                #print([np.around(self.queue[j].set_of_users),self.services])
                u_ind=self.queue[j].set_of_users.astype(int)
                fad_users=self.fading_samples[np.around(u_ind),self.services]
                fad_users=np.reshape(fad_users,np.array(self.queue[j].set_of_users).size)
                self.servicable_users=[i for i in range(fad_users.size) if(fad_users[i]>=thresh)]
                self.servicable_users=np.reshape(self.servicable_users,np.array(self.servicable_users).size)
            else:
                self.servicable_users=np.array([])
        else:
            self.servicable_users=np.array([])

    def makeElementInService(self,i):
        if self.queue.__len__()>0:
            usr_temp=np.array(self.queue[i].set_of_users)
            time_temp=np.array(self.queue[i].time_of_request)
            if self.servicable_users.size > 0:
                self.element_in_service.set_of_users=usr_temp[self.servicable_users]
                self.element_in_service.time_of_request=time_temp[self.servicable_users]
                self.element_in_service.req_file=self.queue[i].req_file
            else:
                self.element_in_service.set_of_users=np.array([])
                self.element_in_service.time_of_request=np.array([])
                self.element_in_service.req_file=np.array([])
        else:
            self.element_in_service.set_of_users=np.array([])
            self.element_in_service.time_of_request=np.array([])
            self.element_in_service.req_file=np.array([])
    def deleteUsers(self,i):
        if self.servicable_users.__len__()>0:
            self.queue[i].set_of_users=list(np.delete(self.queue[i].set_of_users,self.servicable_users,0))
            self.queue[i].time_of_request=list(np.delete(self.queue[i].time_of_request,self.servicable_users,0))
            if len(self.queue[i].set_of_users)==0:
               del self.queue[i]
        q_size=self.queue.__len__()
        del_vec=np.array([])
        k=0
        for j in range(q_size):
            if (k>=0)&(self.queue[k].req_file.size==0):
                del self.queue[k]
                j=j-1
            k=min(j,self.queue.__len__()-1)

    def acceptServeRequests(self): #Accepts Requests and Serves the Multicast queue
        req_index = np.argwhere((self.timelines >= self.serve_start_time) & (self.timelines < self.serve_stop_time))
        req_index = np.reshape(req_index,req_index.size)
        #req_index = np.argwhere((self.timelines >= self.timelines[self.serve_start_index]) & (self.timelines < self.timelines[self.serve_start_index]+self.service_time))
        #print(self.serve_start_time,np.max(self.timelines))
        fwd_index = self.userCaches.requestsToCache(self.requests[req_index], self.users[req_index])
        req_index=req_index[fwd_index]
        if len(req_index) > 0:
            if self.queue.__len__() == 0:

                self.element_in_service=Elements([self.requests[req_index[0]]],[self.users[req_index[0]]],[self.timelines[req_index[0]]])
                for ser in range(0,self.defer_queue.__len__()):
                    if self.defer_queue[ser].req_file == self.element_in_service.req_file:
                        #temp=[self.queue[queue_ind].set_of_users.append(tp) for tp in (add_ele.set_of_users)]
                        #temp=[self.queue[queue_ind].time_of_request.append(tp) for tp in (add_ele.time_of_request)]
                        self.element_in_service.set_of_users=np.append(self.element_in_service.set_of_users, self.defer_queue[ser].set_of_users)
                        self.element_in_service.time_of_request=np.append(self.element_in_service.time_of_request,self.defer_queue[ser].time_of_request)
                        self.defer_queue=np.delete(self.defer_queue,ser)
                        break
                self.queue.append(self.element_in_service)
                #print(len(self.queue))
                self.serveQueueDQN()
                # self.thresholdPowerIndex(0)
                # self.makeElementInService(0)
                # self.deleteUsers(0)
                self.serve_start_time=self.timelines[req_index[0]]
                stop_time_old=self.serve_stop_time
                self.serve_stop_time=self.serve_start_time+self.service_time
                #add Missed requests
                temp_req_index = np.argwhere((self.timelines >= stop_time_old) & (self.timelines < self.serve_stop_time))
                temp_req_index=np.reshape(temp_req_index,temp_req_index.size)
                temp_fwd_index = self.userCaches.requestsToCache(self.requests[temp_req_index], self.users[temp_req_index])
                #print(req_index, temp_fwd_index)
                temp_req_index=temp_req_index[temp_fwd_index]
                req_index=np.append(req_index,temp_req_index)
                #delete queued request
                np.delete(req_index,0)
            else: ## Serve requests using DQN
                self.serveQueueDQN()
                #self.element_in_service=self.queue[0] # Enable for Simple multicast
                #self.queue.popleft()    # Enable for Simple multicast

            for i in req_index:
                add_ele = Elements([self.requests[i]], [self.users[i]], [self.timelines[i]])
                for ser in range(0,self.defer_queue.__len__()):
                    if self.defer_queue[ser].req_file == add_ele.req_file:
                        #temp=[self.queue[queue_ind].set_of_users.append(tp) for tp in (add_ele.set_of_users)]
                        #temp=[self.queue[queue_ind].time_of_request.append(tp) for tp in (add_ele.time_of_request)]
                        add_ele.set_of_users=np.append(add_ele.set_of_users, self.defer_queue[ser].set_of_users)
                        add_ele.time_of_request=np.append(add_ele.time_of_request,self.defer_queue[ser].time_of_request)
                        self.defer_queue=np.delete(self.defer_queue,ser)
                        break
                if self.queue.__len__() == 0:
                    self.queue.append(add_ele)
                else:
                    #temp_queue = self.queue
                    queue_ind = 0
                    while queue_ind<self.queue.__len__():
                        if self.queue[queue_ind].req_file == add_ele.req_file:
                            #temp=[self.queue[queue_ind].set_of_users.append(tp) for tp in (add_ele.set_of_users)]
                            #temp=[self.queue[queue_ind].time_of_request.append(tp) for tp in (add_ele.time_of_request)]
                            self.queue[queue_ind].set_of_users=np.append(self.queue[queue_ind].set_of_users,add_ele.set_of_users)
                            self.queue[queue_ind].time_of_request=np.append(self.queue[queue_ind].time_of_request,add_ele.time_of_request)
                            break
                        queue_ind += 1
                    if queue_ind == self.queue.__len__():
                        self.queue.append(add_ele)
        else:
            if self.queue.__len__() == 0:
                self.element_in_service=Elements([],[],[])
                self.services-=1
            else:
                self.serveQueueDQN()
                #self.element_in_service=self.queue[0] # Enable for Simple multicast
                #self.queue.popleft()    # Enable for Simple multicast
        self.userCaches.MQ2Cache(self.element_in_service)
        self.element_in_service.time_of_request=np.array(self.element_in_service.time_of_request)
        if self.element_in_service.time_of_request.size>0:
            # print(self.serve_stop_time-self.element_in_service.time_of_request)
            # for rm_nan in range(self.element_in_service.time_of_request.size-1,-1,-1):
            x=self.serve_stop_time-self.element_in_service.time_of_request
            x = x[~np.isnan(x)]
            self.sojournTimes=np.append(self.sojournTimes, x)
        self.serve_start_time=self.serve_stop_time
        self.serve_stop_time=self.serve_stop_time+self.service_time
        self.services+=1
        if (((self.serve_start_time>self.timelines[-1]))):
            return 0
        return 1


    def Autoencode(self):#AutoEncoder
        self.inputvector=[]
        for i in range(len(self.queue)):
            if i < self.queue_window:
                fileState=self.queue[i].req_file
                if fileState.size>0:
                    fileState=np.reshape(fileState,1)
                    #print(self.queue[i].set_of_users)
                    temp_usr_state=np.array(self.queue[i].set_of_users)
                    temp_time_state=np.array(self.queue[i].time_of_request)
                    user_pow=map(np.unique,temp_usr_state)
                    user_pow=np.fromiter(user_pow, dtype=np.int)
                    userState=2**user_pow
                    #userState=[2**j for j in (set((temp_usr_state[self.servicable_users]).tolist()))]
                    sojVec=(self.serve_stop_time-temp_time_state)
                    sojoState=np.array([np.max(sojVec), np.mean(sojVec), np.min(sojVec)])
                    #print(fileState,np.array([np.sum(userState)]))
                    fileState=np.array([-1]) # Disable to include File State in the State vector
                    #userState=np.array([-1]) # Disable to include User State in the State vector
                    self.inputvector.append(np.concatenate([fileState,np.array([np.sum(userState)]),sojoState]))
                else:
                    self.inputvector.append([-1,-1,10000,10000,10000])
                #self.thresholdPowerIndex(i)
                """ 
                if self.servicable_users.__len__()>0:
                    fileState=self.queue[i].req_file
                    fileState=np.reshape(fileState,1)
                    #print(self.queue[i].set_of_users)
                    temp_usr_state=np.array(self.queue[i].set_of_users)
                    temp_time_state=np.array(self.queue[i].time_of_request)
                    user_pow=map(np.unique,temp_usr_state[self.servicable_users])
                    user_pow=np.fromiter(user_pow, dtype=np.int)
                    userState=2**user_pow
                    #userState=[2**j for j in (set((temp_usr_state[self.servicable_users]).tolist()))]
                    sojVec=(self.serve_stop_time-temp_time_state[self.servicable_users])
                    sojoState=np.array([np.max(sojVec), np.mean(sojVec), np.min(sojVec)])
                    #print(fileState,np.array([np.sum(userState)]))
                    self.inputvector.append(np.concatenate([fileState,np.array([np.sum(userState)]),sojoState]))
                else:
                    self.inputvector.append([-1,-1,10000,10000,10000])
                """
        if len(self.queue)<self.queue_window:
            temp=[self.inputvector.append([-1,-1,10000,10000,10000]) for i in range(self.queue_window-len(self.queue))]
        fading_state_input=np.reshape(self.fading_samples[:,self.services],self.total_users)
        self.inputvector=np.concatenate(self.inputvector)
        self.inputvector=np.append(self.inputvector,fading_state_input)
        #print(self.inputvector)

    def InstReward(self):
        temp_time_rw=np.array(self.queue[self.action].time_of_request)
        if self.servicable_users.__len__()>0:
            sojActionVec=(self.serve_stop_time-temp_time_rw[self.servicable_users])
            [self.reward_window.append(i) for i in sojActionVec]
            self.reward=1*np.mean(self.reward_window)
        temp_vec=np.array([])
        for j in range(0,np.min([self.queue.__len__(),self.queue_window])):
            temp_time_rw=np.array(self.queue[j].time_of_request)
            #if self.servicable_users.__len__()>0:
            sojActionVec=np.array(self.serve_stop_time-temp_time_rw)
            temp_vec=np.append(temp_vec, sojActionVec)
        if (temp_vec.size!=0):
            self.reward=self.reward+.5*np.mean(temp_vec)

        #self.reward=np.max(sojActionVec)
    def quantize_fading(self,input):
        quant_stat=np.array([.001,.01,.04,.06,.08,.1,.4,.6,.8,1,np.inf])
        ret_vec=np.ones(shape=input.shape)
        for i in range(input.size):
            for j in range(quant_stat.size-1):
                if input[i]>=quant_stat[0]:
                    if (input[i]>=quant_stat[j]) & (input[i]<quant_stat[j+1]):
                        ret_vec[i]=quant_stat[j]
                else:
                    ret_vec[i]=quant_stat[0]

        # print(ret_vec)
        return ret_vec
    def AutoEncoderLoopDef(self):
        self.inputvector=np.array([])
        if self.queue.__len__()>0:
            if self.queue[0].req_file.size>0:
                fileState=self.queue[0].req_file
                fileState=np.reshape(fileState,1)
                #print(self.queue[i].set_of_users)
                temp_usr_state=np.array(self.queue[0].set_of_users)
                temp_time_state=np.array(self.queue[0].time_of_request)
                user_pow=map(np.unique,temp_usr_state)
                user_pow=np.fromiter(user_pow, dtype=np.int)
                userState=2**user_pow
                userState=np.zeros(shape=(self.total_users,))
                for i in range(user_pow.size):
                    userState[user_pow[i].astype(int)]=1
                #userState=[2**j for j in (set((temp_usr_state[self.servicable_users]).tolist()))]
                # sojVec=(self.serve_stop_time-temp_time_state)
                # sojoState=np.array([np.max(sojVec), np.mean(sojVec), np.min(sojVec)])
                #print(fileState,np.array([np.sum(userState)]))
                fileState=np.array([-1]) # Disable to include File State in the State vector
                #userState=np.array([-1]) # Disable to include User State in the State vector
                self.inputvector=np.append(self.inputvector,np.concatenate([np.array((userState))]))
                # self.inputvector=np.append(self.inputvector,np.concatenate([np.reshape([self.queue_decision],(1,)),fileState,np.array([np.sum(userState)])]))
                # self.inputvector=np.append(self.inputvector,np.concatenate([np.reshape([self.queue_decision],(1,)),np.array([np.sum(userState)])]))
            else:
                self.inputvector=np.append(self.inputvector,np.zeros(shape=(self.total_users,)))
                # self.inputvector=np.append(self.inputvector,[self.queue_decision,-1,-1])
                # self.inputvector=np.append(self.inputvector,[self.queue_decision,-1])
        else:
                self.inputvector=np.append(self.inputvector,np.zeros(shape=(self.total_users,)))
            # self.inputvector=np.append(self.inputvector,[self.queue_decision,-1,-1])
            # self.inputvector=np.append(self.inputvector,[self.queue_decision,-1])
        fading_state_input=np.reshape(self.fading_samples[:,self.services],self.total_users)
        # self.inputvector=np.concatenate(self.inputvector)
        # self.inputvector=np.append(self.inputvector,self.quantize_fading(fading_state_input))
        self.inputvector=np.append(self.inputvector,self.quantize_fading(fading_state_input))
        self.LoopDefState=np.append(self.LoopDefState,self.inputvector)

    def AutoEncoderLoopDef_old(self):
        self.inputvector=np.array([])
        if self.queue.__len__()>0:
            if self.queue[0].req_file.size>0:
                fileState=self.queue[0].req_file
                fileState=np.reshape(fileState,1)
                #print(self.queue[i].set_of_users)
                temp_usr_state=np.array(self.queue[0].set_of_users)
                temp_time_state=np.array(self.queue[0].time_of_request)
                user_pow=map(np.unique,temp_usr_state)
                user_pow=np.fromiter(user_pow, dtype=np.int)
                userState=2**user_pow
                #userState=[2**j for j in (set((temp_usr_state[self.servicable_users]).tolist()))]
                # sojVec=(self.serve_stop_time-temp_time_state)
                # sojoState=np.array([np.max(sojVec), np.mean(sojVec), np.min(sojVec)])
                #print(fileState,np.array([np.sum(userState)]))
                fileState=np.array([-1]) # Disable to include File State in the State vector
                #userState=np.array([-1]) # Disable to include User State in the State vector
                self.inputvector=np.append(self.inputvector,np.concatenate([fileState,np.array([np.sum(userState)])]))
                # self.inputvector=np.append(self.inputvector,np.concatenate([np.reshape([self.queue_decision],(1,)),fileState,np.array([np.sum(userState)])]))
                # self.inputvector=np.append(self.inputvector,np.concatenate([np.reshape([self.queue_decision],(1,)),np.array([np.sum(userState)])]))
            else:
                self.inputvector=np.append(self.inputvector,[-1,-1])
                # self.inputvector=np.append(self.inputvector,[self.queue_decision,-1,-1])
                # self.inputvector=np.append(self.inputvector,[self.queue_decision,-1])
        else:
                self.inputvector=np.append(self.inputvector,[-1,-1])
            # self.inputvector=np.append(self.inputvector,[self.queue_decision,-1,-1])
            # self.inputvector=np.append(self.inputvector,[self.queue_decision,-1])
        fading_state_input=np.reshape(self.fading_samples[:,self.services],self.total_users)
        # self.inputvector=np.concatenate(self.inputvector)
        self.inputvector=np.append(self.inputvector,self.quantize_fading(fading_state_input))
        self.LoopDefState=np.append(self.LoopDefState,self.inputvector)

    def AutoEncoderLoopDef_Soj(self):
        # self.inputvector=[self.queue_decision]
        self.LoopDefState=np.append(self.LoopDefState,self.queue_decision)

    def InstRewardSch(self):
        temp_time_rw=np.array(self.queue[self.action].time_of_request)
        if self.servicable_users.__len__()>0:
            sojActionVec=(self.serve_stop_time-temp_time_rw[self.servicable_users])
            [self.reward_window_sch.append(i) for i in sojActionVec]

    def InstRewardLoopDef(self):
        temp_time_rw=np.array(self.queue[self.action].time_of_request)
        if self.servicable_users.__len__()>0:
            sojActionVec=(self.serve_stop_time-temp_time_rw[self.servicable_users])
            # [self.reward_window.append(i) for i in sojActionVec]
            # self.reward_window.append(np.unique(self.servicable_users).__len__())
            self.reward_window.append(np.unique(self.queue[self.action].set_of_users[self.servicable_users]).__len__())
        else:
            self.reward_window.append(0)
        self.power_window.append(self.transmit_power)
        self.powerVecs=np.append(self.powerVecs,self.transmit_power)
    def deferElement(self):
        if self.queue.__len__()>1:
            self.defer_queue=np.append(self.defer_queue,self.queue[0])
            self.queue.popleft()

    def loopElement(self):

        firstElement=self.queue[0]
        self.queue.popleft()
        self.queue.append(firstElement)

    def serveQueueDQN(self):
        #self.Autoencode()#Encodes Input Vector Giving Current state
        #curr_state=self.inputvector
        # self.AutoEncoderLoopDef()

        self.act_dist=stats.rv_discrete(name='act_dist', values=([1,2,3], self.action_prob))
        if np.remainder(self.service_vecs[0],self.retransmit_no)==0:
            self.queue_decision=self.act_dist.rvs(size=1)
            if self.queue_decision==2:
                self.loopElement()
            elif self.queue_decision==3:
                self.deferElement()
        old_state=self.curr_state
        self.LoopDefState=np.array([])
        self.AutoEncoderLoopDef()
        self.curr_state=self.LoopDefState
        if self.enable_ddpg==1:
            self.DDQNA.remember(old_state, self.ddpg_action_prob, self.reward, self.curr_state, False)
            self.ddpg_action_prob=self.DDQNA.get_action(self.curr_state)
            self.transmit_power = self.action_vector[self.ddpg_action_prob]
            self.powerVecsPolicy=np.append(self.powerVecsPolicy,self.transmit_power)


        self.action=0

        self.thresholdPowerIndex(self.action)#Identify servicable users and calculate reward
        self.InstRewardLoopDef()   #Gets instantatneous reward for action using servicable users
        self.InstRewardSch() #Reward for Scheduling
        self.makeElementInService(self.action) #Include only servicable users
        self.deleteUsers(self.action) #Delete only serviced users
        self.service_vecs[self.action]+=1



        if (self.enable_ddpg == 1):

            self.time += 1
            # self.reward=1*np.mean(self.reward_window)-self.power_beta*self.transmit_power
            self.reward=1*np.mean(self.reward_window)
            self.reward_window.clear()
            self.power_window.clear()
            # self.DDPGA.cumul_reward+=self.reward
            # self.reward_array=np.append(self.reward_array, self.DDPGA.cumul_reward/self.time)

            # self.DDPGA.cumul_reward+=self.reward
            # power_grad=0
            if self.time > (300):  # replay if memory has more than 200 elements
                # self.DDPGA.critic.tau=np.max([self.tau_ddpg/(self.time-300)**0,0])
                # self.DDPGA.actor.tau=np.max([self.tau_ddpg/(self.time-300)**0,0])
                # self.DDPGA.train()
                self.DDQNA.replay()
                # if np.remainder(self.time,1)==0:
                    # if np.abs(np.mean(self.powerVecs[-300:])-self.avg_power_constraint)>.01:
                    #     power_grad=np.mean(self.powerVecs[-300:])-self.avg_power_constraint
                    # else:
                    #     power_grad=0

                    # power_grad=np.mean(self.powerVecs[-300:])-self.avg_power_constraint
                    # self.power_beta=np.clip(self.power_beta+self.eta_beta*(power_grad)**0,-self.total_good_users/self.avg_power_constraint,self.total_good_users/self.avg_power_constraint)
                    # self.power_beta=self.AdamOpt.AdamOptimizer(self.power_beta,power_grad,1/(self.time-300)**0)
                    # self.power_beta=np.clip(self.AdamOpt.AdamOptimizer(self.power_beta,power_grad,1/(self.time-300)**0), -self.total_good_users/self.avg_power_constraint,self.total_good_users/self.avg_power_constraint)
        self.reward_array=np.append(self.reward_array, np.mean(self.sojournTimes[-np.min([2000,self.sojournTimes.__len__()]):]))
        self.reward_array=self.reward_array[~np.isnan(self.reward_array)]
        if (np.remainder(self.service_vecs[0], self.schWindow) == 0) & (self.enable_sch == 1):

            self.schTime+=1
            self.reward_sch=1*np.mean(self.reward_window_sch)
            self.reward_window_sch.clear()
            if self.schTime>=10:
                if(np.sum(np.abs(self.action_prob-self.state_memory[-2]))>0):
                    self.state_memory.append(self.action_prob)
                    self.target_memory.append([self.reward_sch])
                    self.stop_sch_training=0
                else:
                    self.stop_sch_training+=0
            else:
                self.state_memory.append(self.action_prob)
                self.target_memory.append([self.reward_sch])
            if self.schTime<20:
                self.action_prob=self.starting_vector[self.schTime]
                samp_ind=np.random.randint(0,self.target_memory.__len__(),50)
                tar=(np.array(self.target_memory)[samp_ind]).reshape(50,1,1)
                stat=(np.array(self.state_memory)[samp_ind]).reshape(50,1,3)
                # tar=(np.array(self.target_memory)[-100:]).reshape(100,1,1)
                # stat=(np.array(self.state_memory)[-100:]).reshape(100,1,3)

                [self.DNN.train_on_batch(stat,tar) for i in range(0,1)]
            # elif self.schTime==20:
            #     self.action_prob=np.array([1,1,1])/3
            else:  # replay if memory has more than 32 elements
                if self.stop_sch_training<10:
                    samp_ind=np.random.randint(0,self.target_memory.__len__(),50)
                    tar=(np.array(self.target_memory)[samp_ind]).reshape(50,1,1)
                    stat=(np.array(self.state_memory)[samp_ind]).reshape(50,1,3)
                    # tar=(np.array(self.target_memory)[-100:]).reshape(100,1,1)
                    # stat=(np.array(self.state_memory)[-100:]).reshape(100,1,3)

                    [self.DNN.train_on_batch(stat,tar) for i in range(0,10)]
                grad=self.DNN.approx_gradient(self.action_prob)
                # decay_dnn=1/(1+0.00001*(self.schTime-19)*(np.log10(10+np.log10(10+self.schTime-19))))
                decay_dnn=1
                self.action_prob=np.clip(self.DNN.AdamOpt.AdamOptimizer(self.action_prob,grad,decay_dnn)+.0005*(.99**(self.schTime-0))*np.random.uniform(0,1,size=3),0,1) # To avoid zero gradients in the beginning
                # self.action_prob=np.clip(self.DNN.AdamOpt.AdamOptimizer(self.action_prob,grad,1/(self.schTime-9)**0)+.0005*(.99**(self.schTime-0))*np.random.uniform(0,1,size=3),0,1) # To avoid zero gradients in the beginning
                # self.action_prob=np.clip((self.action_prob-.01*grad)+.00005*(.9**(self.schTime-100))*np.random.uniform(0,1,size=3),0,1) # To avoid zero gradients in the beginning
                # self.action_prob=np.clip(self.DNN.AdamOpt.AdamOptimizer(self.action_prob,grad),0,1)
                self.action_prob=self.action_prob/np.sum(self.action_prob)
                self.actionProbVec=np.append(self.actionProbVec,self.action_prob)
        # if np.remainder(self.time,100)==0:
        if np.remainder(self.service_vecs[0],100)==0:
            if self.reward_array.size:
                self.live_plotter(np.arange(0,self.reward_array.size),self.reward_array)
                print(self.action_prob,self.transmit_power,self.DDQNA.penalty_lambda,[np.min(self.powerVecsPolicy[-np.min([1000,self.powerVecs.__len__()]):]),np.mean(self.powerVecs[-np.min([1000,self.powerVecs.__len__()]):]),np.max(self.powerVecsPolicy[-np.min([1000,self.powerVecs.__len__()]):])],np.std(self.powerVecsPolicy[-np.min([1000,self.powerVecs.__len__()]):]))
                # print(noise_var,self.transmit_power,self.ddpg_action_prob*(self.max_power-self.avg_power_constraint))
                plt.savefig('Reward.png')
                # if np.remainder(self.time,1000)==0:
                #     print(self.service_vecs)

