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
os.makedirs(savedir, exist_ok=True)

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
    task_id_list = [[1, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 1, 0]]
    task_ints = [0, 0, 0]
    env_ints = [0, 0, 0]
    z_list = [[0.9356837,0.47905245,0.6849074],
              [0.9356837,0.93695736,-0.8765184],
              [0.9356837,1.4,-2]]
    labels = ['z_of_env0', 'z_of_env1', 'z_of_ind1']

    for task_id, task_int, env_int, z, l in zip(task_id_list, task_ints, env_ints, z_list, labels):
        #task_int = np.argmax(np.array(task_id[:task_id_dim]).reshape(1, -1), axis=1)[0]
        #env_int = np.argmax(np.array(task_id[task_id_dim:]).reshape(1, -1), axis=1)[0]

        z = th.tensor(z)

        val_savedir = os.path.join(savedir, f'z_fix_{l}_{task_int}_{env_int}')
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
        #z, _, _ = model.embedding_net.predict(th.tensor(task_id[:task_id_dim]).float(), th.tensor(task_id[task_id_dim:]).float())
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
