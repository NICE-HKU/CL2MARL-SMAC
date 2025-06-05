from runner import Runner
from smacv2.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args
import os
from datetime import datetime
import random
import numpy as np
import torch

def set_seed(seed):
    """设置 Python、NumPy、PyTorch 的随机种子"""
    random.seed(seed)  # Python 随机数生成器
    np.random.seed(seed)  # NumPy 随机数生成器
    torch.manual_seed(seed)  # PyTorch CPU 随机种子
    torch.cuda.manual_seed(seed)  # PyTorch GPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True  # 确保 CuDNN 使用确定性算法
    torch.backends.cudnn.benchmark = False  # 关闭 CuDNN 的自动优化，保证可复现性

if __name__ == '__main__':
    args = get_common_args()
    # 设置随机种子（例如 123或者None）
    if args.seed is not None:
        print(f'set random seed to **{args.seed}**.')
        set_seed(int(args.seed))
    else:
        pass #不设置随机种子，就等于是每次实验都是随机的

    if args.alg.find('coma') > -1:
        args = get_coma_args(args)
    elif args.alg.find('central_v') > -1:
        args = get_centralv_args(args)
    elif args.alg.find('reinforce') > -1:
        args = get_reinforce_args(args)
    else:
        args = get_mixer_args(args)

    if args.alg.find('commnet') > -1:
        args = get_commnet_args(args)
    if args.alg.find('g2anet') > -1:
        args = get_g2anet_args(args)
    # 获取当前日期和时间
    now = datetime.now()
    # 格式化为"YYYYMMDDHHMM"的字符串
    formatted_date = now.strftime("%Y%m")  # %H%M
    formatted_hoursminutesseconds = now.strftime("%H_%M_%S")
    # 用来保存plt和pkl
    save_path = args.result_dir + '/' + 'currilen_advantage_' + formatted_date + '/' + args.map + '/' + formatted_hoursminutesseconds # /home/hk/MARLalgorithm-fuwuqi/result
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.save_path = save_path
    print('save at: ', os.path.abspath(save_path))
    for i in range(args.total_test_number):

        env = StarCraft2Env(map_name=args.map,
                            step_mul=args.step_mul,
                            difficulty=args.difficulty,
                            game_version=args.game_version,
                            replay_dir=args.replay_dir)

        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]

        runner = Runner(env, args)
        if not args.evaluate:
            runner.run(i)
        else:
            dynamic_win_rate, _, default_win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, default_win_rate))
            # print('win_rate is ', default_win_rate, ', training win_rate is ', dynamic_win_rate)
            break
        env.close()
