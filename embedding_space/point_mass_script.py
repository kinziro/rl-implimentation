"""
こちらのスクリプトは強化学習手法PPOの学習と評価を行うためのサンプルコードになります。

"""

import gym, pybullet_envs
import os
import time
from datetime import datetime
import pytz
from OpenGL import GLU
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from stable_baselines3.ppo import MlpPolicy
#from a2c_test import A2C
#from stable_baselines3.common.policies import MlpPolicy
from network import PolicyNet
from a2c import A2C
#from stable_baselines3.common import set_global_seeds
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
#from stable_baselines3 import PPO, A2C
#from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
    get_device,
    get_schedule_fn,
    set_random_seed,
    update_learning_rate,
)
from my_make_vec_env import make_vec_env

import torch as th

import exenv.point_mass
from exenv.point_mass.point_mass_env import PointMassEnv

def action_plot(model, task_id, add_label, figdir):
    xs = np.arange(-10, 10, 0.1)
    ys = np.arange(-10, 10, 0.1)
    a_s = np.arange(0, 30, 1)
    a = np.pi * 0.5 * 10

    # 3D表示
    from mpl_toolkits.mplot3d import Axes3D

    ret = []
    for x in xs:
        for y in ys:
            z, _, _ = model.embedding_net.predict(th.tensor(task_id).float())
            z_list =  z.detach().numpy().flatten()
            obs_add_z = np.hstack([x, y, z.detach().numpy().flatten()])
            action, _states = model.predict(obs_add_z, deterministic=True)
            #action, _ = model.predict([x, y, a])
            ans = [x, y]
            ans.append(action[0])
            ans.append(action[1])
            ret.append(ans)
    ret_np = np.array(ret)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('action_x')
    ax.plot(ret_np[:, 0], ret_np[:, 1], ret_np[:, 2], marker='o', linestyle='None')
    plt.savefig(os.path.join(figdir, '{}_{}.png'.format('action_3D', add_label)))

    # 各軸のみ変更
    # xのみ
    ret = []
    for x in xs:
        z, _, _ = model.embedding_net.predict(th.tensor(task_id).float())
        z_list =  z.detach().numpy().flatten()
        obs_add_z = np.hstack([x, 0, z.detach().numpy().flatten()])
        action, _ = model.predict(obs_add_z, deterministic=True)
        ans = [x]
        ans.append(action[0])
        ret.append(ans)
    ret_np = np.array(ret)

    fig = plt.figure()
    plt.plot(ret_np[:, 0], ret_np[:, 1])
    plt.xlabel('obs_x')
    plt.ylabel('act_x')
    plt.grid()
    plt.savefig(os.path.join(figdir, '{}_{}.png'.format('x_action', add_label)))

    # yのみ
    ret = []
    for y in ys:
        z, _, _ = model.embedding_net.predict(th.tensor(task_id).float())
        z_list =  z.detach().numpy().flatten()
        obs_add_z = np.hstack([0, y, z.detach().numpy().flatten()])
        action, _ = model.predict(obs_add_z, deterministic=True)
        ans = [y]
        ans.append(action[1])
        ret.append(ans)
    ret_np = np.array(ret)

    fig = plt.figure()
    plt.plot(ret_np[:, 0], ret_np[:, 1])
    plt.xlabel('obs_y')
    plt.ylabel('act_y')
    plt.grid()
    plt.savefig(os.path.join(figdir, '{}_{}.png'.format('y_action', add_label)))

def plot_history(data, title, xlabel, ylabel, figdir):
    fig = plt.figure()
    plt.plot(data)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(figdir, f'{title}.png'))

def plot_embedding_space(model, task_id_list, title, figdir):
    colors = ['b', 'r', 'k', 'g']

    theta = np.linspace(0, 2*np.pi, 65)

    fig = plt.figure()
    for i, task_id in enumerate(task_id_list):
        mean, std = model.embedding_net.get_mean_std(th.tensor(task_id).float())
        mean_np = mean.detach().numpy().flatten()
        std_np = std.detach().numpy().flatten()
        x = std_np[0] * np.cos(theta) + mean_np[0]
        y = std_np[1] * np.sin(theta) + mean_np[1]

        plt.plot(x, y, label=f'{task_id}', color=colors[i])
        plt.scatter(mean_np[0], mean_np[1], marker='+', color=colors[i])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)
    plt.savefig(os.path.join(figdir, f'{title}.png'))


# 設定項目
train = True            # 学習を行うかどうか
validation = True       # 評価を行うかどうか
video = False
val_num = 1

env_name = 'PointMass-v0'        # 学習環境(自作環境)
#num_cpu = 2                   # 分散処理させる数(CPUのハイパースレッドの全数を上限が目安)
num_cpu = 32                   # 分散処理させる数(CPUのハイパースレッドの全数を上限が目安)
device = 'cpu'
total_timesteps = 2*(10**6)     # 学習を行うタイムステップ数
#total_timesteps = 4*(10**5)     # 学習を行うタイムステップ数


here = os.path.dirname(os.path.abspath(__file__))
base_savedir = '{}/result/{}/'.format(here, env_name)      # 結果の保存ディレクトリ

starttime = datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%Y/%m/%d %H:%M:%S")

# 学習の実行
task_int_list = [0, 1, 2, 3, 4, 5, 6, 7]
task_id_list = np.eye(8)[task_int_list]
env_kwargs_list = [dict(task_id=task_id) for task_id in task_id_list]
env = make_vec_env(PointMassEnv, n_envs=num_cpu, env_kwargs_list=env_kwargs_list)
env.reset()

savedir = '{}multi_task/'.format(base_savedir)
logdir = '{}tensorboard_log/'.format(savedir)       # tensorboardのログ保存ディレクトリ
os.makedirs(savedir, exist_ok=True)
if train:
    #policy_kwargs = dict(net_arch=[200, 100])
    policy_kwargs = dict(net_arch=[50, 50])
    inference_kwargs = dict(net_arch=[50, 50])
    embedding_kwargs = dict(net_arch=[50, 50])
    schedule_fn = get_schedule_fn(7e-4)
    optimizer_kwargs = dict(lr=schedule_fn(1), alpha=0.99, eps=1e-5, weight_decay=0.001)
    model = A2C(PolicyNet, env, verbose=1, tensorboard_log=logdir, policy_kwargs=policy_kwargs, 
                inference_kwargs=inference_kwargs, embedding_kwargs=embedding_kwargs, 
                optimizer_kwargs=optimizer_kwargs, task_id_list=task_id_list, embedding_dim=2, device=device, n_steps=5)
    #model = A2C(MlpPolicy, env, verbose=1, tensorboard_log=logdir, policy_kwargs=policy_kwargs, 
    #            device=device, n_steps=10)

    #from torchsummary import summary
    #summary(model.policy, input_size=(5,))
    plot_embedding_space(model, task_id_list, 'embedding_space_unleaned', figdir=savedir)

    #for task_id in task_id_list:
    #    val_savedir = os.path.join(savedir, f'{task_id[0]}_{task_id[1]}')
    #    os.makedirs(val_savedir, exist_ok=True)
    #    action_plot(model, task_id, 'unlearned', figdir=val_savedir)
    
    model.learn(total_timesteps=total_timesteps)
    plot_history(model.policy_loss_history, title='policy_loss', xlabel='episode', ylabel='loss', figdir=savedir)
    plot_history(model.inference_loss_history, title='inference_loss', xlabel='episode', ylabel='loss', figdir=savedir)
    plot_history(model.embedding_loss_history, title='embedding_loss', xlabel='episode', ylabel='loss', figdir=savedir)
    plot_history(model.reward_history, title='reward_sum', xlabel='episode', ylabel='reward_sum', figdir=savedir)
    error_zs = np.array(model.error_zs_history)
    for col in range(error_zs.shape[1]):
        plot_history(error_zs[:, col], title=f'error_z_{col}', xlabel='episode', ylabel='error_z', figdir=savedir)

    for data, label in zip([model.policy_g_history, model.inference_g_history, model.embedding_g_history, model.policy_w_history, model.inference_w_history, model.embedding_w_history], 
                            ['policy_g', 'inference_g', 'embedding_g', 'policy_w', 'inference_w', 'embedding_w']):
        data_np = np.array(data)
        fig = plt.figure()
        plt.plot(data_np[:, 0], label='max')
        plt.plot(data_np[:, 1], label='min')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(savedir, f'{label}_max_min.png'))


    model.save('{}embedding_model'.format(savedir))
env.close()
endtime = datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%Y/%m/%d %H:%M:%S")


# 学習結果の確認
if validation:

    model = A2C.load('{}embedding_model.zip'.format(savedir), device=device)
    plot_embedding_space(model, task_id_list, 'embedding_space_leaned', figdir=savedir)

    for task_id in task_id_list:
        val_savedir = os.path.join(savedir, f'{task_id[0]}_{task_id[1]}')
        os.makedirs(val_savedir, exist_ok=True)

        #env0 = make_vec_env(env_name, n_envs=1).envs[0]
        env_kwargs_list = [dict(task_id=task_id)]
        env0 = make_vec_env(PointMassEnv, n_envs=1, env_kwargs_list=env_kwargs_list).envs[0]

        wrapping_flags = [False] * val_num
        if video: wrapping_flags[-1] = True

        sum_r_list = []
        pos_list = []

        # zはタスク一回ごとに一度のみ計算
        z, _, _ = model.embedding_net.predict(th.tensor(task_id).float())

        for i, wrapping in enumerate(wrapping_flags):
            print('--- validation {} ---'.format(i))

            if wrapping:
                from gym import wrappers

                video_path = '{}video'.format(savedir)
                env0 = wrappers.Monitor(env0, video_path, force=True)
            
            obs = env0.reset()
            info = env0.get_info()
            pos_list.append(info['position'].tolist())

            done = False
            ac_obs_list = []

            sum_r = 0
            for step in range(50):
                if step % 10 == 0 and wrapping: print("--step :", step)
                if done:
                    time.sleep(1)
                    #o = env0.reset()
                    break

                obs_add_z = np.hstack([obs, z.detach().numpy().flatten()])
                action, _states = model.predict(obs_add_z, deterministic=True)
                
                obs, rewards, done, info = env0.step(action)

                ac_obs = list(action)
                ac_obs.extend(obs_add_z.tolist())
                ac_obs_list.append(ac_obs)
                sum_r += rewards
                pos_list.append(info['position'].tolist())

                print(step, action, obs_add_z)

            sum_r_list.append(sum_r)

            pd.DataFrame(ac_obs_list).to_csv(os.path.join(val_savedir, 'ac_obs.csv'))
            with open(os.path.join(val_savedir, 'reward_sum.txt'), mode='w') as f:
                f.write(str(sum_r))
        
        print(sum_r_list)

        pos_np = np.array(pos_list)
        fig = plt.figure()
        plt.plot(pos_np[:, 0], pos_np[:, 1])
        plt.scatter(pos_list[0][0], pos_list[0][1], marker='x', color='g')
        plt.scatter(env0.goal[0], env0.goal[1], marker='o', color='r')
        plt.grid()
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        plt.savefig(os.path.join(val_savedir, 'trajectory.png'))

        env0.close()
        #ori_env.close()

        #action_plot(model, task_id, 'learned', figdir=val_savedir)

    print(starttime)
    print(endtime)
