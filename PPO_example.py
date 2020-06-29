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

#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.ppo import MlpPolicy
#from stable_baselines3.common import set_global_seeds
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env

import ex_env.kuka


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

#env_name = 'Pendulum-v0'        # 学習環境(ペンデュラム)
#env_name = 'KukaBulletEnv-v0'        # 学習環境(Kuka iiwa)
env_name = 'KukaEnvExperiment-v0'        # 学習環境(自作環境)
#env_name = 'MountainCar-v0'        # 学習環境(マウンテンカー)
#env_name = 'MountainCarContinuous-v0'        # 学習環境(マウンテンカー)
#env_name = 'CartPole-v1'        # 学習環境(カーポール)
#env_name = 'RoboschoolHumanoid-v1'        # 学習環境(ヒューマノイド)
num_cpu = 8                   # 分散処理させる数(CPUのハイパースレッドの全数を上限が目安)
total_timesteps = 1*(10**7)     # 学習を行うタイムステップ数
#total_timesteps = 4*(10**5)     # 学習を行うタイムステップ数

#ori_env = gym.make(env_name)
#ori_env.reset()
#env = DummyVecEnv([lambda: ori_env])           # シングルタスク用の環境
#env = SubprocVecEnv([make_env(env_name, i) for i in range(num_cpu)])        # マルチタスク用の環境
env = make_vec_env(env_name, n_envs=num_cpu)
env.reset()
#env.render()
#time.sleep(5)

savedir = './result/{}/'.format(env_name)      # 結果の保存ディレクトリ
logdir = '{}tensorboard_log/'.format(savedir)       # tensorboardのログ保存ディレクトリ
os.makedirs(savedir, exist_ok=True)

starttime = datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%Y/%m/%d %H:%M:%S")

# 学習の実行
if train:
    #model = DDPG(MlpPolicy, env, verbose=1, tensorboard_log=logdir)
    model = PPO(MlpPolicy, env, verbose=1, tensorboard_log=logdir)
    model.learn(total_timesteps=total_timesteps)
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
endtime = datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%Y/%m/%d %H:%M:%S")

wrapping = True

# 学習結果の確認
if validation:
    model = PPO.load('{}ppo_model'.format(savedir))
    env0 = env.envs[0]

    if wrapping:
        from gym import wrappers

        video_path = '{}video'.format(savedir)
        wrap_env = wrappers.Monitor(env0, video_path, force=True)

        obs = wrap_env.reset()
    else:
        obs = env0.reset()

    done = False
    obs_list = []

    sum_r = 0
    for step in range(2000):
        if step % 10 == 0: print("step :", step)
        if done:
            time.sleep(1)
            o = wrap_env.reset() if wrapping else env0.reset()
            #o = env0.reset()
            break

        action, _states = model.predict(obs)
        #action = [0, 0, 0]
        obs, rewards, done, info = wrap_env.step(action) if wrapping else env0.step(action)
        #obs, rewards, done, info = env0.step(action)
        obs_list.append(obs.tolist())
        sum_r += rewards

    pd.DataFrame(obs_list).to_csv('{}obs.csv'.format(savedir))
    print('reward', sum_r)
    with open('{}reward_sum.txt'.format(savedir), mode='w') as f:
        f.write(str(sum_r))
    if wrapping: wrap_env.close()
    #ori_env.close()
env.close()

print(starttime)
print(endtime)
