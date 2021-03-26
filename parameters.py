import os
import argparse

def build_parser():
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training and pruning')
    #use in training
    parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')

    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')

    parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')

    parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='refine from prune model')

    parser.add_argument('--model', default='model_best.pth.tar', type=str, metavar='PATH',
                    help='model obtained by training')

    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')

    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, 
                    metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
    #work in main_1.py
    parser.add_argument('--train-flag', action='store_true',
                    help='flag for training network', default=True)#False

    parser.add_argument('--prune-flag', action='store_true',
                    help='flag for pruning network', default=False)#True
    #use in prune
    parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')

    parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save prune model (default: none)')
    return parser

def get_parameter():
    parser = build_parser()
    args = parser.parse_args()

    print("-*-" * 10 + "\n\t\tArguments\n" + "-*-" * 10)
    for key, value in vars(args).items():
        print("%s: %s" % (key, value))

    return args