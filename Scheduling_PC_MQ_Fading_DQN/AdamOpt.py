import numpy as np
class AdamOpt:
    def __init__(self,step=.01,sign=1):
        self.beta1=.9
        self.beta2=.999
        self.m=0
        self.v=0
        self.t=0
        self.eps=10**-8
        self.sign=sign
        self.step=step
    def AdamOptimizer(self, param,grad_input,decay):
        self.t+=1
        self.m=self.beta1*self.m+(1-self.beta1)*grad_input
        self.v=self.beta2*self.v+(1-self.beta2)*(grad_input**2)
        m_est=self.m/(1-self.beta1**self.t)
        v_est=self.v/(1-self.beta2**self.t)
        # print(v_est)
        param=np.array(param+decay*self.step*self.sign*m_est/(np.sqrt(v_est)+self.eps))
        return param
