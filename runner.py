import numpy as np
from common.rollout import RolloutWorker, CommRolloutWorker
from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from datetime import datetime
from pprint import pprint


class Runner:
    def __init__(self, env, args):
        self.env = env
        self.default_difficulty = int(args.difficulty)
        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if not args.evaluate and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:  # these 3 algorithms are on-poliy
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.dynamic_win_rates = []
        self.dynamic_episode_rewards = []
        self.default_win_rates = []
        self.default_episode_rewards = []
        self.save_path = self.args.save_path
        # print('Experimental args: ', self.args)
        args_dict = vars(self.args)
        pprint(args_dict, indent=2, width=100, sort_dicts=True)  # 缩进+宽度控制+排序

    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        game_max_difficulty = 9  # 最高默认是9. 标准默认是7
        # 保证开始和最大默认的正中间，刚好是标准默认 即： 5，6，**7**，8，9
        start_difficulty = int(self.default_difficulty - (game_max_difficulty-self.default_difficulty))
        trainingmax_difficulty = self.default_difficulty if self.args.isdefaultmax_else_gamemax else game_max_difficulty
        current_difficulty = start_difficulty
        self.rolloutWorker.env.set_difficulty(current_difficulty)  # 需环境支持
        timestep_show_interval = 1000
        last_record_ts = 0

        observation_window = 5  # 分析窗口大小
        min_adjust_interval = 3  # 最小调整间隔（评估次数）
        last_adjust_step = -1  # 上次调整时的评估步数
        momentum = 0  # 难度调整动量（-1~+1）
        last_update = 0
        minimum_update_interval = 35
        gamma = 0.8  # 动量衰减系数
        simplestats_windows_length = 10  # hyper-parameter
        simplestats_win_rate_gap = 0.03  # hyper-parameter

        def should_adjust_difficulty(win_rates, episode_rewards, current_diff):
            nonlocal momentum, last_update_idx
            if len(win_rates) - last_update_idx < minimum_update_interval:  # 加入冷却间隔，防止频繁难度调整导致策略学习动荡
                return 0
            # 获取窗口数据
            win_rates = np.array(win_rates[-observation_window:])
            ep_rewards = np.array(episode_rewards[-observation_window:])
            # 计算统计特征
            mu_w = np.mean(win_rates)
            sigma_w = np.std(win_rates)
            mu_r = np.mean(ep_rewards)
            sigma_r = np.std(ep_rewards)
            # 计算趋势斜率
            x = np.arange(len(win_rates))
            beta_w = (len(x) * np.sum(x * win_rates) - np.sum(x) * np.sum(win_rates)) / \
                     (len(x) * np.sum(x ** 2) - np.sum(x) ** 2)
            # 计算奖励二阶差分（需要至少3个数据点）
            if len(ep_rewards) >= 3:
                nabla2_r = ep_rewards[-1] - 2 * ep_rewards[-2] + ep_rewards[-3]
            else:
                nabla2_r = 0
            momentum = gamma * momentum + (1 - gamma) * np.tanh(beta_w + 0.5 * nabla2_r)  # 更新动量
            stability = (sigma_w < 0.08) and (sigma_r < 0.1 * mu_r)  # 计算稳定性指标
            # 动态阈值计算
            tau_h = min(0.8, 0.5 + 0.03 * current_diff)  # 上界阈值
            tau_l = max(0.3, 0.4 - 0.02 * current_diff)  # 下界阈值
            # 难度调整决策
            if mu_w > tau_h and stability and momentum > 0.2:
                last_update_idx = len(win_rates)
                return +1  # 提升难度
            elif (mu_w < tau_l or nabla2_r < -0.05) and momentum < -0.2:
                last_update_idx = len(win_rates)
                return -1  # 降低难度
            return 0

        while time_steps < self.args.n_steps:
            if time_steps - last_record_ts > timestep_show_interval:
                # formatted_date =
                print(f'[{datetime.now().strftime("%Y-%m-%d-%H-%M")}]: '+'Run {}, time_steps {}, difficulty {}'.format(num, time_steps, self.rolloutWorker.env.difficulty))
                last_record_ts = time_steps
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                dynamic_win_rate, dynamic_episode_reward, default_win_rate, default_episode_reward = self.evaluate()
                # print('win_rate is ', win_rate)
                self.dynamic_win_rates.append(dynamic_win_rate)
                self.dynamic_episode_rewards.append(dynamic_episode_reward)
                self.default_win_rates.append(default_win_rate)
                self.default_episode_rewards.append(default_episode_reward)
                self.plt_dynamic(num)
                self.plt_default(num)
                evaluate_steps += 1
                # 通过默认地图上胜率还是现训练的地图上做测试的来调整难度？ 从课程学习思想上，需要用当前训练难度的测试效果做依据，只需要保证最后一段的测试是default 7就行
                if self.args.flexdiffways  == "fixed":  # 当在某level环境下，超过固定胜率表现时，默认达到较好表现
                    win_rate = default_win_rate if self.args.Default_Diff_forChange_orTrain else dynamic_win_rate
                    if win_rate > self.args.diff_change_threshold:  # 胜率超过XX%时提升难度
                        new_current_difficulty = min(current_difficulty + 1, trainingmax_difficulty)  # 难度等级修改
                elif self.args.flexdiffways == "simplestats":
                    check_win_rates = self.default_win_rates if self.args.Default_Diff_forChange_orTrain else self.dynamic_win_rates
                    if not len(check_win_rates) < 2* simplestats_windows_length:
                        # Get the latest 10 win rates
                        latest_window = check_win_rates[-simplestats_windows_length:]
                        # Get the previous window (positions -20 to -10)
                        previous_window = check_win_rates[-2*simplestats_windows_length:-simplestats_windows_length]
                        max_latest = max(latest_window)
                        max_previous = max(previous_window)
                        # Check if latest maximum exceeds previous maximum by at least 5
                        if not max_latest >= max_previous + simplestats_win_rate_gap:
                            new_current_difficulty = min(current_difficulty + 1, trainingmax_difficulty)
                elif self.args.flexdiffways == "flexdiffways":
                    check_win_rates = self.default_win_rates if self.args.Default_Diff_forChange_orTrain else self.dynamic_win_rates
                    check_episode_rewards = self.default_episode_rewards if self.args.Default_Diff_forChange_orTrain else self.dynamic_episode_rewards
                    adjust_difficulty_change = should_adjust_difficulty(check_win_rates, check_episode_rewards, current_difficulty)
                    new_current_difficulty = min(current_difficulty + adjust_difficulty_change, trainingmax_difficulty)
                # 基于上述统计逻辑，动态修改SMAC环境对手难度超参数
                if new_current_difficulty != current_difficulty:
                    print(f'now we change the difficulty from {self.rolloutWorker.env.difficulty} to current_difficulty')
                    self.rolloutWorker.env.set_difficulty(new_current_difficulty)  # 需环境支持
                    print(f'Difficulty increased to {new_current_difficulty}')
                    current_difficulty = new_current_difficulty

            episodes = []
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                episode, _, _, steps = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                time_steps += steps
                # print(_)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                train_steps += 1
            else:
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1
        dynamic_win_rate, dynamic_episode_reward, default_win_rate, default_episode_reward= self.evaluate()
        self.print_log('win_rate is ', default_win_rate, ', training win_rate is ', dynamic_win_rate)

        self.dynamic_win_rates.append(dynamic_win_rate)
        self.dynamic_episode_rewards.append(dynamic_episode_reward)
        self.default_win_rates.append(default_win_rate)
        self.default_episode_rewards.append(default_episode_reward)
        self.plt_dynamic(num)
        self.plt_default(num)
        # 每个run跑完之后，保存并输出一下所有的测试的log的列表
        self.print_log(f"self.dynamic_win_rates, {self.dynamic_win_rates}")
        self.print_log(f"self.dynamic_episode_rewards, {self.dynamic_episode_rewards}")
        self.print_log(f"self.default_win_rates, {self.default_win_rates}")
        self.print_log(f"self.default_episode_rewards, {self.default_episode_rewards}")
        # print(f"self.dynamic_win_rates, {self.dynamic_win_rates}")
        # print(f"self.dynamic_episode_rewards, {self.dynamic_episode_rewards}")
        # print(f"self.default_win_rates, {self.default_win_rates}")
        # print(f"self.default_episode_rewards, {self.default_episode_rewards}")

    def evaluate(self):
        # 额外加入在当前难度的地图下的表现的评估
        dynamic_win_number = 0
        dynamic_episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            dynamic_episode_rewards += episode_reward
            if win_tag:
                dynamic_win_number += 1
        # 测试时, 需要保证脚本难度为默认值；测试完成后恢复训练时候，同时也恢复训练特有的脚本难度
        default_win_number = 0
        default_episode_rewards = 0
        # if self.args.Default_Diff_forChange_orTrain: % 这个不应该放在这，应该放在到时候用winrate和reward的时候
        current_difficulty = self.rolloutWorker.env.difficulty
        self.rolloutWorker.env.set_difficulty(self.default_difficulty)
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            default_episode_rewards += episode_reward
            if win_tag:
                default_win_number += 1
        self.rolloutWorker.env.set_difficulty(current_difficulty)  # reback
        return dynamic_win_number / self.args.evaluate_epoch, dynamic_episode_rewards / self.args.evaluate_epoch, \
        default_win_number/ self.args.evaluate_epoch, default_episode_rewards / self.args.evaluate_epoch

    def plt(self, num):
        plt.figure()
        plt.ylim([0, 105])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('win_rates')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
        plt.close()

    def plt_default(self, num):
        plt.figure()
        plt.ylim([0, 105])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.default_win_rates)), self.default_win_rates)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('win_rates_de')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.default_episode_rewards)), self.default_episode_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards_de')

        plt.savefig(self.save_path + '/plt_{}_de.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.default_win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.default_episode_rewards)
        plt.close()

    def plt_dynamic(self, num):
        plt.figure()
        plt.ylim([0, 105])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.dynamic_win_rates)), self.dynamic_win_rates)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('win_rates_dy')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.dynamic_episode_rewards)), self.dynamic_episode_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards_dy')

        plt.savefig(self.save_path + '/plt_{}_dy.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.dynamic_win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.dynamic_episode_rewards)
        plt.close()


    def print_log(self, *args):
        """
        自定义打印函数，同时输出到控制台和日志文件
        用法：self.print_log("内容") 或 self.print_log(f"变量: {var}")
        """
        # 构建要输出的内容
        message = " ".join(str(arg) for arg in args)
        # 确保save_path存在
        if hasattr(self, 'save_path'):
            log_file = f"{self.save_path}/running.log"
            # 写入文件（追加模式）
            with open(log_file, 'a') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] {message}\n")
        else:
            print("Warning: self.save_path not found, cannot write to log file")




