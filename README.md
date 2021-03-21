# Training
RUN main.py --s 0.001  --epochs 160  --refine ''(空）
# Pruning
RUN prune.py --model model_best.pth.tar --save pruned.pth.tar --percent 0.5
# Retraining
RUN main.py -refine pruned.pth.tar --epochs 40
# Issues
剪枝比例过高容易出现通道数为0，需要在剪枝的是的时候添加约束条件。
