# coding=utf-8

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

def rnn_factory(rnn_type, **kwargs):
  pack_padded_seq = True
  if rnn_type in ["LSTM", "GRU", "RNN"]:
    rnn = getattr(nn, rnn_type)(**kwargs)
  return rnn, pack_padded_seq

class EncoderBase(nn.Module):
  """ encoder base class
  """
  def __init__(self):
    super(EncoderBase, self).__init__()

  def forward(self, src, lengths=None, hidden=None):
    """
    Args:
      src (FloatTensor) : input sequence 
      lengths (LongTensor) : lengths of input sequence
      hidden : init hidden state
    """
    raise NotImplementedError()

class RNNEncoder(EncoderBase):
  """ RNN encoder class

  Args:
    rnn_type : rnn cell type, ["LSTM", "GRU", "RNN"]
    bidirectional : whether use bidirectional rnn
    num_layers : number of layers in stacked rnn
    input_size : input dimension size
    hidden_size : rnn hidden dimension size
    dropout : dropout rate
    use_bridge : TODO: implement bridge
  """
  def __init__(self, rnn_type, bidirectional, num_layers,
    input_size, hidden_size, dropout, use_bridge=False):
    super(RNNEncoder, self).__init__()
    if bidirectional:
      assert hidden_size % 2 == 0
      hidden_size = hidden_size // 2
    self.rnn, self.pack_padded_seq = rnn_factory(rnn_type,
      input_size=input_size,
      hidden_size=hidden_size,
      bidirectional=bidirectional,
      num_layers=num_layers,
      dropout=dropout)
    self.use_bridge = use_bridge
    if self.use_bridge:
      raise NotImplementedError()
  
  def forward(self, src, lengths=None, hidden=None):
    """
    Same as BaseEncoder.forward
    """
    packed_src = src
    if self.pack_padded_seq and lengths is not None:
      lengths = lengths.view(-1).tolist()
      packed_src = pack(src, lengths)

    memory_bank, hidden_final = self.rnn(packed_src, hidden)

    if self.pack_padded_seq and lengths is not None:
      memory_bank = unpack(memory_bank)[0]
    
    if self.use_bridge:
      raise NotImplementedError()
    return memory_bank, hidden_final