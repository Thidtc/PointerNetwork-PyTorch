# coding=utf-8

import torch
from torch import nn
import torch.nn.functional as F

from layers.seq2seq.encoder import rnn_factory
from layers.attention import Attention

class RNNDecoderBase(nn.Module):
  """ RNN decoder base class
  Args:
    rnn_type : rnn cell type, ["LSTM", "GRU", "RNN"]
    bidirectional : whether use bidirectional rnn
    num_layers : number of layers in stacked rnn
    input_size : input dimension size
    hidden_size : rnn hidden dimension size
    dropout : dropout rate
  """
  def __init__(self, rnn_type, bidirectional, num_layers,
    input_size, hidden_size, dropout):
    super(RNNDecoderBase, self).__init__()
    if bidirectional:
      assert hidden_size % 2 == 0
      hidden_size = hidden_size // 2
    self.rnn, _ = rnn_factory(rnn_type,
      input_size=input_size,
      hidden_size=hidden_size,
      bidirectional=bidirectional,
      num_layers=num_layers,
      dropout=dropout)
  
  def forward(self, tgt, memory_bank, hidden, memory_lengths=None):
    """
    Args:
      tgt: target sequence
      memory_bank : memory from encoder or other source
      hidden : init hidden state
      memory_lengths : lengths of memory
    """
    raise NotImplementedError()


