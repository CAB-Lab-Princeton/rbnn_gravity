import time
import torch 
from rbnn_gravity.models import RBNN
from rbnn_gravity.utils import setup_reproducibility
from rbnn_gravity.dataset import build_dataloader
from rbnn_gravity.configuration import config


def train(args, model, videos_dataloader, retrain=False):
    # define t as something less than T
    t = 1
    training_loss = []
    
    # TO Read: cosine loss and if it is a better one to use instead of MSE for omega
    loss_fcn = torch.nn.MSELoss()
    
    params = model.parameters()
    optim = torch.optim.Adam(params)
    
    start_time = time.time()

    for epoch in range(config.experiment.n_epoch):
        for idx, data in enumerate(videos_dataloader):
            R_data, omega_data = data
            R_data.requires_grad = True
            # omega_data.requires_grad = True
            # import pdb; pdb.set_trace()
            # print(R_data.shape, omega_data.shape)
        
            R_cur, R_next = R_data[:, :t].type(torch.float), R_data[:, t:].type(torch.float)
            omega_cur, omega_next = omega_data[:, :t].type(torch.float), omega_data[:, t:].type(torch.float)
            # print(R_cur.shape, omega_cur.shape)
            # Training over multiple vs single step (SRNN reference has some info on this) 
            R_next_hat, omega_next_hat = model(R_cur.squeeze(), omega_cur.squeeze())
            loss = loss_fcn(R_next.squeeze(), R_next_hat) + loss_fcn(omega_next.squeeze(), omega_next_hat)
        
            optim.zero_grad()
            loss.backward()
            optim.step()
        
            if idx % config.experiment.print_every == 0:
              batch_loss = loss.item()
              training_loss.append(batch_loss)
              print(batch_loss)


def run_experiment(
    args, run_train=True, retrain=False, return_net=False, shuffle=True, drop_last=True
):
    setup_reproducibility(args.seed)
    model = RBNN()

    params = model.parameters()
    optim = torch.optim.Adam(params)
    
    videos_dataloader = build_dataloader(args)
    model = RBNN()
    train(args, model, videos_dataloader, retrain=config.experiment.retrain)