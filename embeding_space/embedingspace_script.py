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

#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.ppo import MlpPolicy
#from stable_baselines3.common import set_global_seeds
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
#from stable_baselines3 import PPO, A2C
from a2c import A2C
from stable_baselines3.common.cmd_util import make_vec_env

import torch as th

import exenv.kuka

def action_plot(model, add_label):
    xs = np.arange(-20, 20, 1)
    ys = np.arange(-20, 20, 1)
    a_s = np.arange(0, 30, 1)
    a = np.pi * 0.5 * 10

    # 3D表示
    from mpl_toolkits.mplot3d import Axes3D

    ret = []
    for x in xs:
        for y in ys:
            action, _ = model.predict([x, y, a])
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
    plt.savefig(os.path.join(savedir, '{}_{}.png'.format('action_3D', add_label)))

    # 各軸のみ変更
    # xのみ
    ret = []
    for x in xs:
        action, _ = model.predict([x, 0, a], deterministic=True)
        ans = [x]
        ans.append(action[0])
        ret.append(ans)
    ret_np = np.array(ret)

    fig = plt.figure()
    plt.plot(ret_np[:, 0], ret_np[:, 1])
    plt.xlabel('obs_x')
    plt.ylabel('act_x')
    plt.grid()
    plt.savefig(os.path.join(savedir, '{}_{}.png'.format('x_action', add_label)))

    # yのみ
    ret = []
    for y in ys:
        action, _ = model.predict([0, y, a], deterministic=True)
        ans = [y]
        ans.append(action[1])
        ret.append(ans)
    ret_np = np.array(ret)

    fig = plt.figure()
    plt.plot(ret_np[:, 0], ret_np[:, 1])
    plt.xlabel('obs_y')
    plt.ylabel('act_y')
    plt.grid()
    plt.savefig(os.path.join(savedir, '{}_{}.png'.format('y_action', add_label)))

    # aのみ
    ret = []
    for a in a_s:
        action, _ = model.predict([0, 0, a], deterministic=True)
        #action, _ = model.predict([a], deterministic=True)
        ans = [a]
        #ans.append(action[2])
        ans.append(action[0])
        ret.append(ans)
    ret_np = np.array(ret)

    fig = plt.figure()
    plt.plot(ret_np[:, 0], ret_np[:, 1])
    plt.xlabel('obs_a')
    plt.ylabel('act_a')
    plt.grid()
    plt.savefig(os.path.join(savedir, '{}_{}.png'.format('a_action', add_label)))




#def make_env(env_name, rank, seed=0):
#    """
#    Utility function for multiprocessed env.
#
#    :param env_name: (str) the environment ID
#    :param rank: (int) index of the subprocess
#    :param seed: (int) the inital seed for RNG
#    """
#    def _init():
#        env = gym.make(env_name)
#        env.seed(seed + rank)
#        return env
#    set_global_seeds(seed)
#    return _init


# 設定項目
train = True            # 学習を行うかどうか
validation = True       # 評価を行うかどうか
video = True
val_num = 1

#env_name = 'Pendulum-v0'        # 学習環境(ペンデュラム)
#env_name = 'KukaBulletEnv-v0'        # 学習環境(Kuka iiwa)
env_name = 'KukaEnvExperiment-v0'        # 学習環境(自作環境)
#env_name = 'MountainCar-v0'        # 学習環境(マウンテンカー)
#env_name = 'MountainCarContinuous-v0'        # 学習環境(マウンテンカー)
#env_name = 'CartPole-v1'        # 学習環境(カーポール)
#env_name = 'RoboschoolHumanoid-v1'        # 学習環境(ヒューマノイド)
num_cpu = 4                   # 分散処理させる数(CPUのハイパースレッドの全数を上限が目安)
total_timesteps = 1*(10**2)     # 学習を行うタイムステップ数
#total_timesteps = 4*(10**5)     # 学習を行うタイムステップ数

#ori_env = gym.make(env_name)
#ori_env.reset()
#env = DummyVecEnv([lambda: ori_env])           # シングルタスク用の環境
#env = SubprocVecEnv([make_env(env_name, i) for i in range(num_cpu)])        # マルチタスク用の環境
env = make_vec_env(env_name, n_envs=num_cpu)
env.reset()
#env.render()
#time.sleep(5)

here = os.path.dirname(os.path.abspath(__file__))
savedir = '{}/result/{}/'.format(here, env_name)      # 結果の保存ディレクトリ
logdir = '{}tensorboard_log/'.format(savedir)       # tensorboardのログ保存ディレクトリ
os.makedirs(savedir, exist_ok=True)

starttime = datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%Y/%m/%d %H:%M:%S")

# 学習の実行
task_id = (0, 0)
if train:
    #model = DDPG(MlpPolicy, env, verbose=1, tensorboard_log=logdir)
    #model = PPO(MlpPolicy, env, verbose=1, tensorboard_log=logdir)
    policy_kwargs = dict(net_arch=[200, 100])
    model = A2C(MlpPolicy, env, verbose=1, tensorboard_log=logdir, policy_kwargs=policy_kwargs, task_id_dim=2, embeding_dim=3)
    #model = A2C(MlpPolicy, env, verbose=1, tensorboard_log=logdir, policy_kwargs=policy_kwargs)

    #from torchsummary import summary
    #summary(model.policy, input_size=(6,))

    #action_plot(model, 'unlearned')
    
    model.learn(total_timesteps=total_timesteps, task_id=task_id)
    #model.learn(total_timesteps=total_timesteps)

    model.save('{}ppo_model'.format(savedir))
    '''
    # 学習途中を表示する場合はコメントアウトを外す
    obs = env.reset()
    for i in range(10):
        print('timesteps : {}'.format(i))
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
    '''
env.close()
endtime = datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%Y/%m/%d %H:%M:%S")


# 学習結果の確認
if validation:

    model = A2C.load('{}ppo_model'.format(savedir))
    env0 = make_vec_env(env_name, n_envs=1).envs[0]

    wrapping_flags = [False] * val_num
    if video: wrapping_flags[-1] = True

    height_list = []
    sum_r_list = []
    blockPos_list = []

    for i, wrapping in enumerate(wrapping_flags):
        print('--- validation {} ---'.format(i))

        if wrapping:
            from gym import wrappers

            video_path = '{}video'.format(savedir)
            env0 = wrappers.Monitor(env0, video_path, force=True)
        
        obs = env0.reset()
        blockPos_list.append(env0.init_blockPos)

        done = False
        ac_obs_list = []

        sum_r = 0
        for step in range(1000):
            if step % 10 == 0 and wrapping: print("--step :", step)
            if done:
                time.sleep(1)
                #o = env0.reset()
                break

            z, _ = model.embeding_net.predict(th.tensor(task_id).float())
            z_list =  z.detach().numpy().flatten()
            obs_add_z = np.hstack([obs, z.detach().numpy().flatten()])
            action, _states = model.predict(obs_add_z, deterministic=True)
            
            #action = [0, 0, 0]
            #action = [0, 0]
            #action = [0]
            #obs, rewards, done, info = wrap_env.step(action) if wrapping else env0.step(action)
            obs, rewards, done, info = env0.step(action)
            #print('reward: ', rewards)

            state = env0.get_endeffectorPos()
            print('{}  action: {},  obs: {}, z {}'.format(step, action, obs, state[2]))
            #ac_obs = action.tolist()
            ac_obs = list(action)
            ac_obs.extend(obs.tolist())
            ac_obs_list.append(ac_obs)
            sum_r += rewards

            #if step >= 500:
            #    print('')

        block_height = env0.get_blockPos()[2]
        print('block_h', block_height)
        height_list.append(block_height)
        sum_r_list.append(sum_r)

        pd.DataFrame(ac_obs_list).to_csv('{}ac_obs.csv'.format(savedir))
        #print('reward', sum_r)
        with open('{}reward_sum.txt'.format(savedir), mode='w') as f:
            f.write(str(sum_r))
    
    blockPos_list = np.array(blockPos_list).T.tolist()
    #success_list = [1 if r > 0 else 0 for r in sum_r_list]
    #success_list = [1 if h > -0.07 else 0 for h in height_list]
    success_list = [1 if h > 0.1 else 0 for h in height_list]
    print(height_list)
    print(sum_r_list)
    blockPos_list.append(success_list)
    pd.DataFrame(blockPos_list).to_csv('{}blockPos.csv'.format(savedir))

    ok_list = []
    ng_list = []
    for b_pos in np.array(blockPos_list).T:
        if b_pos[3] == 1:
            ok_list.append([b_pos[0], b_pos[1]])
        else:
            ng_list.append([b_pos[0], b_pos[1]])

    # xy座標に対する成否プロット
    ok_np = np.array(ok_list)
    ng_np = np.array(ng_list)
    fig = plt.figure()
    title = 'grasping_result'
    if len(ok_np) > 0: plt.scatter(ok_np[:, 0], ok_np[:, 1], label='OK', color='b')
    if len(ng_np) > 0: plt.scatter(ng_np[:, 0], ng_np[:, 1], label='NG', color='r')
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.savefig(os.path.join(savedir, '{}.png'.format(title)))

    #print('height_list', height_list)
    #print('sum_r_list', sum_r_list)
    #if wrapping: env0.close()
    env0.close()
    #ori_env.close()

print(starttime)
print(endtime)


# データ化
model = A2C.load('{}ppo_model'.format(savedir))
#action_plot(model, 'learned')


#from mpl_toolkits.mplot3d import Axes3D
#
#fig = plt.figure()
#ax = Axes3D(fig)
#
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('action_x')
#ax.plot(ret_np[:, 0], ret_np[:, 1], ret_np[:, 2], marker='o', linestyle='None')
#plt.savefig(os.path.join(savedir, '{}.png'.format('x_action')))

#fig = plt.figure()
#ax = Axes3D(fig)
#
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('action_y')
#ax.plot(ret_np[:, 0], ret_np[:, 1], ret_np[:, 3], marker='o', linestyle='None')
#plt.savefig(os.path.join(savedir, '{}.png'.format('y_action')))




