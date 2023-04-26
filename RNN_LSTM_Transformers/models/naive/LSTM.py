"""
LSTM model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class init function, forward function and weight initialization

    def __init__(self, input_size, hidden_size):
        """ Init function for LSTM class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes as you wish here.                      #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   You also need to include correct activation functions                      #
        ################################################################################
        self.weight_ii = nn.Parameter(torch.zeros(self.input_size, self.hidden_size))
        self.bias_ii = nn.Parameter(torch.ones(self.hidden_size))        
        self.weight_hi = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))
        self.bias_hi = nn.Parameter(torch.ones(self.hidden_size))        
        
        self.weight_if = nn.Parameter(torch.zeros(self.input_size, self.hidden_size))
        self.bias_if = nn.Parameter(torch.ones(self.hidden_size))        
        self.weight_hf = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))
        self.bias_hf = nn.Parameter(torch.ones(self.hidden_size))                
        
        self.weight_ig = nn.Parameter(torch.zeros(self.input_size, self.hidden_size))
        self.bias_ig = nn.Parameter(torch.ones(self.hidden_size))        
        self.weight_hg = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))
        self.bias_hg = nn.Parameter(torch.ones(self.hidden_size))        
        
        self.weight_io = nn.Parameter(torch.zeros(self.input_size, self.hidden_size))
        self.bias_io = nn.Parameter(torch.ones(self.hidden_size))        
        self.weight_ho = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))
        self.bias_ho = nn.Parameter(torch.ones(self.hidden_size))        
        
        # i_t: input gate
        self.i_t = nn.Sigmoid()

        # f_t: the forget gate
        self.f_t = nn.Sigmoid()

        # g_t: the cell gate
        self.g_t = nn.Tanh()
        
        # o_t: the output gate
        self.o_t = nn.Sigmoid()
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""

        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              # 
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################
        h_t, c_t = None, None
        if init_states is None:
            h_t = torch.autograd.Variable(x.new_zeros(x.size(0), self.hidden_size))
            h_t = (h_t, h_t)
        
        h_t, c_t = h_t         
        
        for i in range(x.shape[1]):
            # i_t: input gate
            i_t = self.i_t(torch.matmul(x[:,i,:], self.weight_ii) + self.bias_ii + torch.matmul(h_t, self.weight_hi) + self.bias_hi)
            # f_t: the forget gate
            f_t = self.f_t(torch.matmul(x[:,i,:], self.weight_if) + self.bias_if + torch.matmul(h_t, self.weight_hf) + self.bias_hf)
            # g_t: the cell gate
            g_t = self.g_t(torch.matmul(x[:,i,:], self.weight_ig) + self.bias_ig + torch.matmul(h_t, self.weight_hg) + self.bias_hg)
            # o_t: the output gate
            o_t = self.o_t(torch.matmul(x[:,i,:], self.weight_io) + self.bias_io + torch.matmul(h_t, self.weight_ho) + self.bias_ho)
            c_t = c_t * f_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)   
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)
