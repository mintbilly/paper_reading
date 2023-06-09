"""
S2S Decoder model.  (c) 2021 Georgia Tech

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

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN", "LSTM".                                                   #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################    
        # 1) Embedding Layer
        self.embedding = nn.Embedding(self.output_size, self.emb_size)
        
        # 2) Recurrent Layer
        if self.model_type == 'RNN':
            self.recurrent = nn.RNN(self.emb_size, self.decoder_hidden_size, batch_first = True)
        elif model_type == 'LSTM':
            self.recurrent = nn.LSTM(self.emb_size, self.decoder_hidden_size, batch_first = True)
            
        # 3) Linear layers with Logsoftmax activation 
        self.linear_softmax = nn.Sequential(
            nn.Linear(in_features = self.decoder_hidden_size, out_features = self.output_size),
            nn.LogSoftmax(dim=1)
        )
        
        # 4) Drop-out Layer
        self.dropout = nn.Dropout(p = dropout)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input, hidden):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, 1); HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden weights of the previous time step from the decoder
            Returns:
                output (tensor): the output of the decoder
                hidden (tensor): the weights coming out of the hidden unit
        """

        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #       Apply linear layer and softmax activation to output tensor before   #
        #       returning it.                                                       #
        #############################################################################
        embedding = self.dropout(self.embedding(input))
        if self.model_type == 'RNN':
            output, hidden = self.recurrent(embedding, hidden)
        elif self.model_type == 'LSTM':
            output, hidden = self.recurrent(embedding, hidden)  
        output = self.linear_softmax(output[:,0,:]) 
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return output, hidden
