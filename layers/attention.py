# coding=utf-8

import torch
from torch import nn
import torch.nn.functional as F

def sequence_mask(lengths, max_len=None):
  """ Crete mask for lengths
  Args:
    lengths (LongTensor) : lengths (bz)
    max_len (int) : maximum length
  Return:
    mask (bz, max_len)
  """
  bz = lengths.numel()
  max_len = max_len or lengths.max()
  return (torch.arange(0, max_len)
        .type_as(lengths)
        .repeat(bz, 1)
        .lt(lengths))

class Attention(nn.Module):
  """ Attention layer
  Args:
    attn_type : attention type ["dot", "general"]
    dim : input dimension size
  """
  def __init__(self, attn_type, dim):
    super(Attention, self).__init__()
    self.attn_type = attn_type
    bias_out = attn_type == "mlp"
    self.linear_out = nn.Linear(dim *2, dim, bias_out)
    if self.attn_type == "general":
      self.linear = nn.Linear(dim, dim, bias=False)
    elif self.attn_type == "dot":
      pass
    else:
      raise NotImplementedError()
  
  def score(self, src, tgt):
    """ Attention score calculation
    Args:
      src : source values (bz, src_len, dim)
      tgt : target values (bz, tgt_len, dim)
    """
    bz, src_len, dim = src.size()
    _, tgt_len, _ = tgt.size()

    if self.attn_type in ["genenral", "dot"]:
      tgt_ = tgt
      if self.attn_type == "general":
        tgt_ = self.linear(tgt_)
      src_ = src.transpose(1, 2)
      return torch.bmm(tgt_, src_)
    else:
      raise NotImplementedError()
  
  def forward(self, src, tgt, src_lengths=None):
    """
    Args:
      src : source values (bz, src_len, dim)
      tgt : target values (bz, tgt_len, dim)
      src_lengths : source values length
    """
    if tgt.dim() == 2:
      one_step = True
      src = src.unsqueeze(1)
    else:
      one_step = False
    
    bz, src_len, dim = src.size()
    _, tgt_len, _ = tgt.size()

    align_score = self.score(src, tgt)

    if src_lengths is not None:
      mask = sequence_mask(src_lengths)
      # (bz, max_len) -> (bz, 1, max_len)
      # so mask can broadcast
      mask = mask.unsqueeze(1)
      align_score.data.masked_fill_(1 - mask, -float('inf'))
    
    # Normalize weights
    align_score = F.softmax(align_score, -1)

    c = torch.bmm(align_score, src)

    concat_c = torch.cat([c, tgt], -1)
    attn_h = self.linear_out(concat_c)

    if one_step:
      attn_h = attn_h.squeeze(1)
      align_score = align_score.squeeze(1)
    
    return attn_h, align_score