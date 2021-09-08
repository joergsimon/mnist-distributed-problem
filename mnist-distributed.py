# imports
import argparse
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.append("../")

# import tlf.train_helper as th
# import tlf.distributed_utils as du
# import tlf.utils as utils


def train_ppd(pnum, config, args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    rank = config.node_num * config.gpus + config.gpu
    print(f'spawned training on gpu:{config.gpu} and node:{config.nodes}, resulting in rank:{rank} full '
          f'config( {config} ) and args ( {args} )')
    if torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)
        dist.init_process_group(backend='nccl', init_method='env://', world_size=config.world_size, rank=rank)
    else:
        dist.init_process_group(backend='gloo', world_size=config.world_size, rank=rank)
    print('process group initiated')
    # Declare transform to convert raw data to tensor
    trsfms = transforms.Compose([
        transforms.ToTensor()
    ])

    # Loading Data and splitting it into train and validation data
    train = datasets.MNIST('', train=True, transform=trsfms, download=True)
    train, valid = random_split(train, [50000, 10000])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train, num_replicas=config.world_size,
                                                                    rank=rank) if config.world_size > 1 else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid, num_replicas=config.world_size,
                                                                    rank=rank) if config.world_size > 1 else None

    # Create Dataloader of the above tensor with batch size = 32
    trainloader = DataLoader(train, batch_size=128, sampler=train_sampler, shuffle=(train_sampler is None),
                             pin_memory=True, num_workers=0)
    validloader = DataLoader(valid, batch_size=128, sampler=valid_sampler, shuffle=(valid_sampler is None),
                             pin_memory=True, num_workers=0)

    # Building Our Mode
    class Network(nn.Module):
        # Declaring the Architecture
        def __init__(self):
            super(Network, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)

        # Forward Pass
        def forward(self, x):
            x = x.view(x.shape[0], -1)  # Flatten the images
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = Network()
    if torch.cuda.is_available():
        model = model.cuda(config.gpu)

    # Declaring Criterion and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    if torch.cuda.is_available():
        criterion = criterion.cuda(config.gpu)

    model = DDP(model, device_ids=[config.gpu], output_device=config.gpu) if torch.cuda.is_available() else DDP(model)

    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.div_(batch_size))
            return res

    # Training with Validation
    # epochs = config.epochs
    # min_valid_loss = np.inf
    #
    # start = datetime.now()
    # for e, train_loss, valid_loss, per_epoch_metrics in tb.train_for('xyz2', "", {'lr': 0.01}, epochs, model,
    #                                                                  trainloader,
    #                                                                  validloader, optimizer, criterion,
    #                                                                  metrics_function,
    #                                                                  config.gpu, True, rank):
    #     if config.world_size > 1:
    #         trainloader.sampler.set_epoch(e)
    #         validloader.sampler.set_epoch(e)
    #     print(f'Epoch {e + 1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss}')
    #
    #     if min_valid_loss > valid_loss:
    #         print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
    #         min_valid_loss = valid_loss
    #
    #         # Saving State Dict
    #         torch.save(model.state_dict(), 'saved_model.pth')

    epochs = config.epochs
    min_valid_loss = np.inf

    start = datetime.now()
    for e in range(epochs):
        if config.world_size > 1:
            trainloader.sampler.set_epoch(e)
            validloader.sampler.set_epoch(e)

        train_losses = []
        valid_losses = []
        valid_accuracies = []

        model.train()
        for data, labels in trainloader:
            if torch.cuda.is_available():
                data, labels = data.cuda(config.gpu), labels.cuda(config.gpu)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for data, labels in validloader:
                if torch.cuda.is_available():
                    data, labels = data.cuda(config.gpu), labels.cuda(config.gpu)
                pred = model(data)
                loss = criterion(pred, labels)
                valid_losses.append(loss.item())
                valid_accuracies.append(accuracy(pred, labels)[0].item())

        train_loss = np.mean(np.array(train_losses))
        valid_loss = np.mean(np.array(valid_losses))
        valid_accuracy = np.mean(np.array(valid_accuracies))

        print(f'Epoch {e + 1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss}')

        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss

            # Saving State Dict
            torch.save(model.state_dict(), 'saved_model.pth')

    if rank == 0:
        print(f'final valid accuracy: {valid_accuracy}')
        print('Training complete in: ' + str(datetime.now() - start))


@dataclass
class DDPConfig:
    # environment vars
    address: str = 'localhost'
    port: Optional[str] = None
    nccl_socket_ifname: str = 'lo'
    ncll_ib_disable: str = '1'
    # ddp config
    nodes: int = 1
    gpus: int = 1
    node_num: int = 0
    gpu: int = 0
    visible_gpu: int = 0
    world_size: int = 1
    epochs: int = 2

def parse_config(parser):
    args = parser.parse_args()
    config = DDPConfig(nodes=args.nodes, gpus=args.gpus, node_num=args.nr, gpu=args.gpu, visible_gpu=args.visiblegpu,
                       epochs=args.epochs)
    if args.address is not None:
        config.address = args.address
    if args.port is not None:
        config.port = args.port
    if args.ifname is not None:
        config.nccl_socket_ifname = args.ifname
    if args.ibdisable is not None:
        config.ncll_ib_disable = args.ibdisable
    if args.worldsize == -1:
        config.world_size = config.nodes * config.gpus
    return config, args



def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def set_environment_for_ddp(conf: DDPConfig, print_info=True):
    port = conf.port
    if conf.port is None:
        port = find_free_port()
        print(f'found free port at {port}')
    os.environ['MASTER_ADDR'] = conf.address
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{conf.visible_gpu}'
    os.environ['MASTER_PORT'] = port
    os.environ['NCCL_SOCKET_IFNAME'] = conf.nccl_socket_ifname
    os.environ['NCCL_IB_DISABLE'] = conf.ncll_ib_disable

    if print_info:
        print('set env vars :')
        print(f'MASTER_ADDR: {conf.address}')
        print(f'CUDA_VISIBLE_DEVICES: {conf.visible_gpu}')
        print(f'MASTER_PORT: {port}')
        print(f'NCCL_SOCKET_IFNAME: {conf.nccl_socket_ifname}')
        print(f'NCCL_IB_DISABLE: {conf.ncll_ib_disable}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N', help='number of nodes')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('-gp', '--gpu', default=0, type=int, help='number of gpu within the node')
    parser.add_argument('-e', '--epochs', default=2, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-vg', '--visiblegpu', type=int, help='number of GPU in cluster')
    parser.add_argument('-ws', '--worldsize', default=-1, type=int,
                        help='number of processes to spawn, if left blank will be nodes*gpus')
    parser.add_argument('-addr', '--address', type=str, help='the IP address of the master node')
    parser.add_argument('-p', '--port', type=str, help='the port of the master node')
    parser.add_argument('-if', '--ifname', type=str, help='the interface name to reach the master node (like eth0)')
    parser.add_argument('-ibd', '--ibdisable', type=str, help='something with nccl, must be 1 or 0')
    config, args = parse_config(parser)
    set_environment_for_ddp(config)
    mp.spawn(train_ppd, nprocs=config.gpus, args=(config,args))


if __name__ == '__main__':
    main()
