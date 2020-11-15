from a2c import A2C
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

def kl_div(P, Q):
    P_np = P.detach().numpy()
    Q_np = Q.detach().numpy()
    kl_divs = P * (P / Q).log()
    kl_div = kl_divs.sum()

    #kldiv2 = torch.nn.functional.kl_div(Q.log(), P, reduction='sum')

    return kl_div

#P = torch.Tensor([0.36, 0.48, 0.16])
#Q = torch.Tensor([0.333, 0.333, 0.333])
#kl_div(P, Q)

device = 'cpu'
env_name = 'PointMass-v0'        # 学習環境(自作環境)
here = os.path.dirname(os.path.abspath(__file__))
base_savedir = '{}/result/{}/multi_task/'.format(here, env_name)      # 結果の保存ディレクトリ

# サンプリングしたzの値
zs = []
for x in np.arange(-0.5, 0.6, 0.1):
    for y in np.arange(-0.5, 0.6, 0.1):
        zs.append([x, y])
zs_tensor = torch.tensor(zs).float()

# ベースのz
task_id = (0, 0)
data_num = len(zs)
model = A2C.load('{}embedding_model.zip'.format(base_savedir), device=device)

task_ids = [task_id for _ in range(data_num)]
#_, log_probs_base, _ = model.embedding_net.forward(torch.tensor(task_ids).float())
log_probs_base, _ = model.embedding_net.evaluate_actions(torch.tensor(task_ids).float(), zs_tensor)
mean, std = model.embedding_net.get_mean_std(torch.tensor(task_id).float())
print(f'{task_id}, mean:{mean.detach().numpy()}, std:{std.detach().numpy()}')
print('-------------------')

mean_list = []
std_list = []
task_id_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
for task_id in task_id_list:
    task_ids = [task_id for _ in range(data_num)]
    #_, log_probs, _ = model.embedding_net.forward(torch.tensor(task_ids).float())
    log_probs, _ = model.embedding_net.evaluate_actions(torch.tensor(task_ids).float(), zs_tensor)
    mean, std = model.embedding_net.get_mean_std(torch.tensor(task_id).float())
    mean_list.append(mean.detach().numpy())
    std_list.append(std.detach().numpy())

    kldiv = torch.nn.functional.kl_div(log_probs_base, log_probs.exp(), reduction='batchmean')
    #kldiv = torch.nn.functional.kl_div(log_probs_base, log_probs.exp())
    #kldiv = kl_div(log_probs_base.exp(), log_probs.exp())
    print(f'{task_id} kldiv:{kldiv.item()}, mean:{mean.detach().numpy()}, std:{std.detach().numpy()}')
