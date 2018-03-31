# coding=utf-8

import numpy as np
import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import argparse
import logging
import sys
from tensorboardX import SummaryWriter

from dataset import CHDataset
from pointer_network import PointerNet, PointerNetLoss

if __name__ == "__main__":
  # Parse argument
  parser = argparse.ArgumentParser("Convex Hull")
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument("--bz", type=int, default=256)
  parser.add_argument("--max_in_seq_len", type=int, default=5)
  parser.add_argument("--max_out_seq_len", type=int, default=6)
  parser.add_argument("--rnn_hidden_size", type=int, default=128)
  parser.add_argument("--attention_size", type=int, default=128)
  parser.add_argument("--num_layers", type=int, default=1)
  parser.add_argument("--beam_width", type=int, default=2)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--clip_norm", type=float, default=5.)
  parser.add_argument('--weight_decay', type=float, default=0.1)
  parser.add_argument("--check_interval", type=int, default=20)
  parser.add_argument("--nepoch", type=int, default=200)
  parser.add_argument("--train_filename", type=str, default="./data/convex_hull_5_test.txt")
  parser.add_argument("--model_file", type=str, default=None)
  parser.add_argument("--log_dir", type=str, default="./log")

  args = parser.parse_args()

  # Pytroch configuration
  if args.gpu >= 0 and torch.cuda.is_available():
    args.use_cuda = True
    torch.cuda.device(args.gpu)
  else:
    args.use_cuda = False

  # Logger
  logger = logging.getLogger("Convex Hull")
  formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
  console_handler = logging.StreamHandler(sys.stdout)
  console_handler.formatter = formatter
  logger.addHandler(console_handler)
  logger.setLevel(logging.DEBUG)

  # Summary writer
  writer = SummaryWriter(args.log_dir)

  # Loading data
  train_ds = CHDataset(args.train_filename, args.max_in_seq_len,
    args.max_out_seq_len)
  logger.info("Train data size: {}".format(len(train_ds)))

  train_dl = DataLoader(train_ds, num_workers=2, batch_size=args.bz) 

  # Init model
  model = PointerNet("LSTM",
    True,
    args.num_layers,
    2,
    args.rnn_hidden_size,
    0.0)
  criterion = PointerNetLoss()
  optimizer = optim.Adam(model.parameters(), lr=args.lr)

  if args.use_cuda:
    model.cuda()

  # Training
  for epoch in range(args.nepoch):
    model.train()
    total_loss = 0.
    batch_cnt = 0.
    for b_inp, b_inp_len, b_outp_in, b_outp_out, b_outp_len in train_dl:
      b_inp = Variable(b_inp)
      b_outp_in = Variable(b_outp_in)
      b_outp_out = Variable(b_outp_out)
      if args.use_cuda:
        b_inp = b_inp.cuda()
        b_inp_len = b_inp_len.cuda()
        b_outp_in = b_outp_in.cuda()
        b_outp_out = b_outp_out.cuda()
        b_outp_len = b_outp_len.cuda()
      
      optimizer.zero_grad()
      align_score = model(b_inp, b_inp_len, b_outp_in, b_outp_len)
      loss = criterion(b_outp_out, align_score, b_outp_len)

      l = loss.data[0]
      total_loss += l
      batch_cnt += 1

      loss.backward()
      clip_grad_norm(model.parameters(), args.clip_norm)
      optimizer.step()
    writer.add_scalar('train/loss', total_loss / batch_cnt, epoch)
    logger.info("Epoch : {}, loss {}".format(epoch, total_loss / batch_cnt))

    # Checkout
    if epoch % args.check_interval == args.check_interval - 1:
      # Save model
      if args.model_file is not None:
        torch.save(model.state_dict(), args.model_file)