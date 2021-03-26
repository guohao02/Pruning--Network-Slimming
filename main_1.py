from training import train_net
from parameters import get_parameter
from prune import prune_begain

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    args = get_parameter()

    if args.train_flag:
        print('train_flag:',args.train_flag)
        train_net(args)
    elif args.prune_flag:
        print('prune_flag:',args.prune_flag)
        network = prune_begain(args)
        print(network)

