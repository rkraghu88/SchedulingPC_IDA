import numpy as np
import tensorflow as tf
import keras.backend as K
import Scheduling_PC_MQ_Fading_DQN.AdamOpt as AdamOpt
from scipy.optimize import approx_fprime
from keras.initializers import RandomUniform
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, concatenate, LSTM, Reshape, BatchNormalization, Lambda, Flatten, Conv1D


class DNNApproximator:
    """ Critic for the DDPG Algorithm, Q-Value function approximator
    """

    def __init__(self, inp_dim, out_dim, lr, tau):
        # Dimensions and Hyperparams
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.tau, self.lr = tau, lr
        self.model = self.network()
        self.model.compile(Adam(self.lr), 'mse')
        self.AdamOpt=AdamOpt.AdamOpt(sign=-1,step=self.tau)
    def network(self):
        """ Assemble Critic network to predict q-values
        """
        state = Input((self.env_dim))
        x = Dense(32, activation='elu')(state)
        x = Dense(16, activation='elu')(x)
        out = Dense(1, activation='linear', kernel_initializer=RandomUniform())(x)
        return Model(state, out)

    def approx_gradient(self,inp):

        return approx_fprime(inp,self.gradient_function,.01)

    def gradient_function(self,inp_raw):
        inp=np.array(inp_raw).reshape(1,1,np.array(inp_raw).__len__())
        return self.target_predict(inp)

    def target_predict(self, inp):
        """ Predict Q-Values using the target network
        """
        return self.model.predict(inp)

    def train_on_batch(self, states, critic_target):
        """ Train the critic network on batch of sampled experience
        """
        # print(actions)
        # return self.model.train_on_batch([states, actions], critic_target)

        return self.model.train_on_batch(states, critic_target)
    #
    # def transfer_weights(self):
    #     """ Transfer model weights to target model with a factor of Tau
    #     """
    #     W, target_W = self.model.get_weights(), self.target_model.get_weights()
    #     for i in range(len(W)):
    #         target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
    #     self.target_model.set_weights(target_W)
    #
    # def save(self, path):
    #     self.model.save_weights(path + '_critic.h5')
    #
    # def load_weights(self, path):
    #     self.model.load_weights(path)
