import torch
import torch.nn as nn
import torch.nn.functional as F

from common.arguments import get_common_args


class MultiScaleEncoder(nn.Module):
    """ 多尺度状态编码器 """

    def __init__(self, args):
        super().__init__()
        # 微观编码器（处理单位级特征）
        self.micro_encoder = nn.Sequential(
            nn.Linear(args.obs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        # 宏观编码器（处理全局状态）
        self.macro_encoder = nn.Sequential(
            nn.Linear(args.state_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        # 特征融合层
        self.fusion = nn.Linear(32 * 2, args.state_shape)  # 输出维度与原状态一致

    def forward(self, obs, state):
        """
        输入:
            obs: [batch * n_agents, obs_dim]
            state: [batch, state_dim]
        输出:
            fused_state: [batch, state_dim] (与原始状态同维度)
        """
        micro_feat = self.micro_encoder(obs)  # [batch*n_agents, 32]
        micro_feat = micro_feat.mean(dim=0, keepdim=True)  # 聚合智能体特征 [1, 32]

        macro_feat = self.macro_encoder(state)  # [batch, 32]
        fused = torch.cat([micro_feat.expand_as(macro_feat), macro_feat], dim=-1)  # [batch, 64]
        return self.fusion(fused)  # [batch, state_dim]


class QattenNet(nn.Module):
    def __init__(self, args):
        super(QattenNet, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = args.state_shape
        self.embed_dim = args.qmix_hidden_dim  # 保持与QMIX相同的隐层维度

        self.multi_scale_encoder = MultiScaleEncoder(args)

        # 多头注意力参数（默认4头）
        self.n_heads = 4
        self.head_dim = self.embed_dim // self.n_heads

        # 注意力键值变换（完全匹配QMIX的超网络结构）
        if args.two_hyper_layers:
            self.key_extractors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(args.state_shape, args.hyper_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(args.hyper_hidden_dim, self.head_dim)
                ) for _ in range(self.n_heads)
            ])
            self.agent_extractors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(1, args.hyper_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(args.hyper_hidden_dim, self.head_dim)
                ) for _ in range(self.n_heads)
            ])
        else:
            self.key_extractors = nn.ModuleList([
                nn.Linear(args.state_shape, self.head_dim)
                for _ in range(self.n_heads)
            ])
            self.agent_extractors = nn.ModuleList([
                nn.Linear(1, self.head_dim)
                for _ in range(self.n_heads)
            ])

        # 状态价值函数V(s)（保持与QMIX相同的结构）
        if args.two_hyper_layers:
            self.value_net = nn.Sequential(
                nn.Linear(args.state_shape, args.hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hyper_hidden_dim, self.embed_dim)
            )
        else:
            self.value_net = nn.Linear(args.state_shape, self.embed_dim)

        # 输出层（严格匹配QMIX的输出结构）
        self.output_net = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, q_values, states,obs=None):
        """
        输入输出维度与原始QMixNet完全一致：
        输入:
            q_values: [episode_num, max_episode_len, n_agents]
            states: [episode_num, max_episode_len, state_dim]
        输出:
            q_total: [episode_num, max_episode_len, 1]
        """
        episode_num = q_values.size(0)
        max_episode_len = q_values.size(1)

        # 1. 维度展平（与QMIX完全一致的处理方式）
        q_values = q_values.view(-1, 1, self.n_agents)  # [episode_num * max_episode_len, 1, n_agents]
        states = states.reshape(-1, self.state_dim)  # [episode_num * max_episode_len, state_dim]

        # 多尺度状态编码
        if obs is not None:
            obs = obs.reshape(-1, self.args.obs_shape)  # [episode_num * max_ep_len * n_agents, obs_dim]
            states = self.multi_scale_encoder(obs, states)  # 编码后维度不变

        total_samples = q_values.size(0)

        # 2. 计算多头注意力（保持batch维度为total_samples）
        weights = []
        for i in range(self.n_heads):
            # 2.1 处理agent Q值（严格保持维度）
            agent_input = q_values.transpose(1, 2).reshape(-1, 1)  # [total_samples * n_agents, 1]
            agent_proj = self.agent_extractors[i](agent_input)  # [total_samples * n_agents, head_dim]
            agent_proj = agent_proj.view(total_samples, self.n_agents,
                                         self.head_dim)  # [total_samples, n_agents, head_dim]

            # 2.2 处理状态（与QMIX的超网络相同处理方式）
            keys = self.key_extractors[i](states).unsqueeze(1)  # [total_samples, 1, head_dim]

            # 2.3 计算注意力权重
            attention = torch.bmm(agent_proj, keys.transpose(1, 2)) / (self.head_dim ** 0.5)
            weights.append(F.softmax(attention, dim=1))  # [total_samples, n_agents, 1]

        # 3. 加权求和（维度处理与QMIX的混合操作对应）
        values = self.value_net(states).view(total_samples, 1, -1)  # [total_samples, 1, embed_dim]
        weighted = sum([w * values for w in weights])  # [total_samples, n_agents, embed_dim]
        weighted = weighted.mean(dim=1)  # [total_samples, embed_dim]

        # 4. 输出全局Q值（严格匹配QMIX的输出形状）
        q_total = self.output_net(weighted)  # [total_samples, 1]
        return q_total.view(episode_num, max_episode_len, 1)  # [episode_num, max_episode_len, 1]


if __name__ == '__main__':
    # 测试代码
    args = get_common_args()
    args.n_agents = 5
    args.state_shape = 120
    args.qmix_hidden_dim = 32
    args.batch_size = 32

    qatten = QattenNet(args)
    q_values = torch.rand(1920, 1, 5)  # batch_size * seq_len = 1920 (假设32*60)
    states = torch.rand(1920, 120)

    output = qatten(q_values, states)
    print(output.shape)  # 应该输出 torch.Size([32, 60, 1])
