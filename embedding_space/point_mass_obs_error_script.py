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
import shutil
import yaml

#from stable_baselines3.ppo import MlpPolicy
#from a2c_test import A2C
#from stable_baselines3.common.policies import MlpPolicy
from network import PolicyNet
#from a2c import A2C
from a2c_task_and_env_id import A2C
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
#from exenv.point_mass.point_mass_env import PointMassEnv
from exenv.point_mass.point_mass_env_obs_error import PointMassEnv
from mpl_toolkits.mplot3d import Axes3D

def action_plot(model, task_id, add_label, figdir):
    xs = np.arange(-10, 10, 0.1)
    ys = np.arange(-10, 10, 0.1)
    a_s = np.arange(0, 30, 1)
    a = np.pi * 0.5 * 10

    # 3D表示

    ret = []
    for x in xs:
        for y in ys:
            z, _, _ = model.embedding_net.predict(th.tensor(task_id[:task_id_dim]).float(), th.tensor(task_id[task_id_dim:]).float())
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
        z, _, _ = model.embedding_net.predict(th.tensor(task_id[:task_id_dim]).float(), th.tensor(task_id[task_id_dim:]).float())
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
        z, _, _ = model.embedding_net.predict(th.tensor(task_id[:task_id_dim]).float(), th.tensor(task_id[task_id_dim:]).float())
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

def plot_embedding_space(model, task_id_list, title, figdir, val_flag_list=None):
    colors = ['b', 'r', 'k', 'g', 'b', 'r', 'k', 'g']
    #markers = ['+', '+', '+', '+', 'o', 'o', 'o', 'o']
    markers = ['$0$', '$1$', '$2$', '$3$', '$4$', '$5$', '$6$', '$7$']

    theta = np.linspace(0, 2*np.pi, 65)

    fig = plt.figure()
    for i, task_id in enumerate(task_id_list):
        if val_flag_list is not None:
            if val_flag_list[i] == 0:
                continue
        mean, std = model.embedding_net.get_mean_std(th.tensor(task_id[:task_id_dim]).float(), th.tensor(task_id[task_id_dim:]).float())
        mean_np = mean.detach().numpy().flatten()
        std_np = std.detach().numpy().flatten()
        z1 = std_np[0] * np.cos(theta) + mean_np[0]
        z2 = std_np[1] * np.sin(theta) + mean_np[1]

        task_int = np.argmax(np.array(task_id[:task_id_dim]).reshape(1, -1), axis=1)[0]
        env_int = np.argmax(np.array(task_id[task_id_dim:]).reshape(1, -1), axis=1)[0]

        plt.plot(z1, z2, label=f'No.{i}_{task_int}_{env_int}', color=colors[i])
        plt.scatter(mean_np[0], mean_np[1], marker=markers[i], color=colors[i])
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.grid()
    #plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0)
    plt.subplots_adjust(left=0.125, right=0.8, bottom=0.1, top=0.9)
    plt.savefig(os.path.join(figdir, f'{title}_z1_2.png'))

    fig = plt.figure()
    for i, task_id in enumerate(task_id_list):
        if val_flag_list is not None:
            if val_flag_list[i] == 0:
                continue
        mean, std = model.embedding_net.get_mean_std(th.tensor(task_id[:task_id_dim]).float(), th.tensor(task_id[task_id_dim:]).float())
        mean_np = mean.detach().numpy().flatten()
        std_np = std.detach().numpy().flatten()
        z1 = std_np[1] * np.cos(theta) + mean_np[1]
        z2 = std_np[2] * np.sin(theta) + mean_np[2]

        task_int = np.argmax(np.array(task_id[:task_id_dim]).reshape(1, -1), axis=1)[0]
        env_int = np.argmax(np.array(task_id[task_id_dim:]).reshape(1, -1), axis=1)[0]

        plt.plot(z1, z2, label=f'No.{i}_{task_int}_{env_int}', color=colors[i])
        plt.scatter(mean_np[1], mean_np[2], marker=markers[i], color=colors[i])
    plt.xlabel('z2')
    plt.ylabel('z3')
    plt.grid()
    #plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0)
    plt.subplots_adjust(left=0.125, right=0.8, bottom=0.1, top=0.9)
    plt.savefig(os.path.join(figdir, f'{title}_z2_3.png'))



    if len(mean_np) == 3:
        fig = plt.figure()
        ax = Axes3D(fig)

        for i, task_id in enumerate(task_id_list):
            if val_flag_list is not None:
                if val_flag_list[i] == 0:
                    continue
            mean, std = model.embedding_net.get_mean_std(th.tensor(task_id[:task_id_dim]).float(), th.tensor(task_id[task_id_dim:]).float())
            mean_np = mean.detach().numpy().flatten()
            std_np = std.detach().numpy().flatten()
            z1 = std_np[0] * np.cos(theta) + mean_np[0]
            z2 = std_np[1] * np.sin(theta) + mean_np[1]
            z3 = std_np[2] * np.sin(theta) + mean_np[2]

            task_int = np.argmax(np.array(task_id[:task_id_dim]).reshape(1, -1), axis=1)[0]
            env_int = np.argmax(np.array(task_id[task_id_dim:]).reshape(1, -1), axis=1)[0]

            ax.plot(z1, z2, z3, label=f'No.{i}_{task_int}_{env_int}', color=colors[i])
            ax.scatter(mean_np[0], mean_np[1], mean_np[2], marker=markers[i], color=colors[i])

        ax.set_xlabel('z1')
        ax.set_ylabel('z2')
        ax.set_zlabel('z3')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.legend()
        #ax.subplots_adjust(left=0.125, right=0.8, bottom=0.1, top=0.9)
        plt.savefig(os.path.join(figdir, f'{title}_3D.png'))


# 設定ファイルの読み込み
here = os.path.dirname(os.path.abspath(__file__))
config_fn = "config_obs_error.yml"
with open(os.path.join(here, config_fn)) as f:
    config = yaml.safe_load(f.read())

# 設定項目
env_name = 'PointMassObsError-v0'        # 学習環境(自作環境)

# 学習の実行
task_int_list = config["env"]["task_int_list"]
max_task_int = config["env"]["max_task_int"]
task_id_dim = max_task_int + 1
env_int_list = config["env"]["env_int_list"]
max_env_int = config["env"]["max_env_int"]
env_id_dim = max_env_int + 1
task_id_list = np.hstack([np.eye(task_id_dim)[task_int_list], np.eye(env_id_dim)[env_int_list]])

base_savedir = '{}/result/{}/'.format(here, env_name)      # 結果の保存ディレクトリ
savedir = '{}multi_task/'.format(base_savedir)
logdir = '{}tensorboard_log/'.format(savedir)       # tensorboardのログ保存ディレクトリ
os.makedirs(savedir, exist_ok=True)

starttime = datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%Y/%m/%d %H:%M:%S")

# train
if config["learn"]["flag"]:
    # 設定ファイルをコピー
    shutil.copyfile(os.path.join(here, config_fn), os.path.join(savedir, config_fn))

    n_cpu = config["learn"]["n_cpu"]                   # 分散処理させる数(CPUのハイパースレッドの全数を上限が目安)
    # 環境の構築
    env_kwargs = config["env"]["env_kwargs"]
    env_init_random = config["learn"]["env_init_random"]
    env_kwargs_list = [dict(task_id=task_id, task_len=task_id_dim, init_random=env_init_random) for task_id in task_id_list]
    [env_k.update(env_kwargs) for env_k in env_kwargs_list]
    env = make_vec_env(PointMassEnv, n_envs=n_cpu, env_kwargs_list=env_kwargs_list)
    env.reset()

    # ネットワークの定義
    task_embedding_dim = config["algorithm"]["task_embedding_dim"]
    env_embedding_dim = config["algorithm"]["env_embedding_dim"]
    policy_kwargs = config["algorithm"]["policy_kwargs"]
    inference_kwargs = config["algorithm"]["inference_kwargs"]
    embedding_kwargs = config["algorithm"]["embedding_kwargs"]
    optimizer_kwargs = config["algorithm"]["optimizer_kwargs"]
    loss_alphas = config["algorithm"]["loss_alphas"]

    # 学習条件の定義
    total_timesteps = config["learn"]["total_timesteps"]     # 学習を行うタイムステップ数
    device = config["learn"]["device"]
    n_steps =  config["learn"]["n_steps"]
    verbose =  config["learn"]["verbose"]

    # モデルの作成
    if config["learn"]["retrain"]:
        model = A2C.load('{}embedding_model.zip'.format(savedir), env=env, device=device)
    else:
        model = A2C(PolicyNet, env, verbose=verbose, tensorboard_log=logdir, policy_kwargs=policy_kwargs, 
                    inference_kwargs=inference_kwargs, embedding_kwargs=embedding_kwargs, 
                    optimizer_kwargs=optimizer_kwargs, task_id_list=task_id_list, task_id_dim=task_id_dim, env_id_dim=env_id_dim, 
                    task_embedding_dim=task_embedding_dim, env_embedding_dim=env_embedding_dim, device=device, n_steps=n_steps, 
                    loss_alphas=loss_alphas, max_grad_norm=0.3)
    #model = A2C(MlpPolicy, env, verbose=1, tensorboard_log=logdir, policy_kwargs=policy_kwargs, 
    #            device=device, n_steps=10)

    #from torchsummary import summary
    #summary(model.policy, input_size=(5,))
    plot_embedding_space(model, task_id_list, 'embedding_space_unleaned', figdir=savedir)

    #for task_id in task_id_list:
    #    val_savedir = os.path.join(savedir, f'{task_id[0]}_{task_id[1]}')
    #    os.makedirs(val_savedir, exist_ok=True)
    #    action_plot(model, task_id, 'unlearned', figdir=val_savedir)
    
    model.learn(total_timesteps=total_timesteps, savedir=savedir)
    plot_history(model.policy_loss_history, title='policy_loss', xlabel='episode', ylabel='loss', figdir=savedir)
    plot_history(model.inference_loss_history, title='inference_loss', xlabel='episode', ylabel='loss', figdir=savedir)
    plot_history(model.embedding_loss_history, title='embedding_loss', xlabel='episode', ylabel='loss', figdir=savedir)
    plot_history(model.reward_history, title='reward_sum', xlabel='episode', ylabel='reward_sum', figdir=savedir)
    plot_history(model.R_history, title='R_sum', xlabel='episode', ylabel='R_sum', figdir=savedir)
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
    plt.cla()
endtime = datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%Y/%m/%d %H:%M:%S")


# 学習結果の確認
if config["validation"]["flag"]:
    # 検証条件
    device = config["validation"]["device"]
    video = config["validation"]["flag"]
    n_val = config["validation"]["n_val"]
    timestep = config["validation"]["timestep"]
    env_kwargs = config["env"]["env_kwargs"]
    env_init_random = config["learn"]["env_init_random"]

    val_flag_list = config["validation"]["val_flag_list"]

    model = A2C.load('{}embedding_model.zip'.format(savedir), device=device)
    plot_embedding_space(model, task_id_list, 'embedding_space_leaned', figdir=savedir, val_flag_list=val_flag_list)

    for task_id, val_flag in zip(task_id_list, val_flag_list):
        if val_flag == 0:
            continue
        task_int = np.argmax(np.array(task_id[:task_id_dim]).reshape(1, -1), axis=1)[0]
        env_int = np.argmax(np.array(task_id[task_id_dim:]).reshape(1, -1), axis=1)[0]

        val_savedir = os.path.join(savedir, f'{task_int}_{env_int}')
        os.makedirs(val_savedir, exist_ok=True)

        env_kwargs_list = [dict(task_id=task_id, task_len=task_id_dim, init_random=env_init_random)]
        [env_k.update(env_kwargs) for env_k in env_kwargs_list]
        env0 = make_vec_env(PointMassEnv, n_envs=1, env_kwargs_list=env_kwargs_list).envs[0]

        wrapping_flags = [False] * n_val
        if video: wrapping_flags[-1] = True

        sum_r_list = []
        real_pos_list = []
        obs_pos_list = []

        # zはタスク一回ごとに一度のみ計算
        z, _, _ = model.embedding_net.predict(th.tensor(task_id[:task_id_dim]).float(), th.tensor(task_id[task_id_dim:]).float())
        pd.DataFrame(z.detach().numpy()).to_csv(os.path.join(val_savedir, 'z.csv'))

        for i, wrapping in enumerate(wrapping_flags):
            print('--- validation {} ---'.format(i))
            print('z:', z)

            if wrapping:
                from gym import wrappers

                video_path = '{}video'.format(savedir)
                env0 = wrappers.Monitor(env0, video_path, force=True)
            
            obs = env0.reset()
            info = env0.get_info()
            real_pos_list.append(info['real_position'].tolist())
            obs_pos_list.append(info['obs_position'].tolist())

            done = False
            ac_obs_list = []

            sum_r = 0
            for step in range(timestep):
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
                real_pos_list.append(info['real_position'].tolist())
                obs_pos_list.append(info['obs_position'].tolist())

                print(step, action, obs_add_z)

            sum_r_list.append(sum_r)

            pd.DataFrame(ac_obs_list).to_csv(os.path.join(val_savedir, 'ac_obs.csv'))
            with open(os.path.join(val_savedir, 'reward_sum.txt'), mode='w') as f:
                f.write(str(sum_r))
        
        print(sum_r_list)

        for pos_list, l in zip([real_pos_list, obs_pos_list], ['real', 'obs']):
            pos_np = np.array(pos_list)
            fig = plt.figure()
            plt.plot(pos_np[:, 0], pos_np[:, 1], marker='.')
            plt.scatter(pos_list[0][0], pos_list[0][1], marker='x', color='g')
            plt.scatter(env0.goal[0], env0.goal[1], marker='o', color='r')
            plt.grid()
            plt.xlim([-5, 5])
            plt.ylim([-5, 5])
            plt.savefig(os.path.join(val_savedir, f'trajectory_{l}.png'))

        env0.close()
        #ori_env.close()

        action_plot(model, task_id, 'learned', figdir=val_savedir)

        plt.cla()

    print(starttime)
    print(endtime)
