import math
import torch.nn as nn
import torch.nn.functional as F
import torch


class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        input = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(input + x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_channels, channel_height, channel_width, action_size, option_seq_length, option_action_size):
        super(PolicyNetwork, self).__init__()
        self.channel_height = channel_height
        self.channel_width = channel_width
        self.num_output_channels = math.ceil(action_size / (channel_height * channel_width))
        self.action_size = action_size
        self.option_seq_length = option_seq_length
        self.option_action_size = option_action_size
        self.conv = nn.Conv2d(num_channels, self.num_output_channels * option_seq_length, kernel_size=1)
        self.bn = nn.BatchNorm2d(self.num_output_channels * option_seq_length)
        self.fcs = nn.ModuleList([nn.Linear(self.num_output_channels * channel_height * channel_width, action_size if i == 0 else option_action_size) for i in range(option_seq_length)])
        for i, fc in enumerate(self.fcs):
            if i == 0:
                continue
            bias = torch.zeros_like(fc.bias.data)
            bias[option_action_size - 1] = 10
            fc.bias.data = bias

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        option_logit = torch.empty(0)
        policy_logit = torch.empty(0)
        for i, fc in enumerate(self.fcs):
            x_i = x[:, i * self.num_output_channels: (i + 1) * self.num_output_channels]
            x_i = x_i.view(-1, self.num_output_channels * self.channel_height * self.channel_width)
            if policy_logit.numel() == 0:
                policy_logit = fc(x_i).view(-1, self.action_size)
            elif option_logit.numel() == 0:
                option_logit = fc(x_i).view(-1, 1, self.option_action_size)
            else:
                option_logit = torch.cat((option_logit, fc(x_i).view(-1, 1, self.option_action_size)), dim=1)
        tmp = torch.full((policy_logit.shape[0], 1), -100).to(policy_logit.device)
        tmp = torch.cat((policy_logit, tmp), dim=1)
        if option_logit.numel() == 0:
            option_logit = tmp.view(-1, 1, self.option_action_size)
        else:
            option_logit = torch.cat((tmp.view(-1, 1, self.option_action_size), option_logit), dim=1)
        return policy_logit, option_logit


class MuZeroConsistencyNetwork(nn.Module):
    def __init__(self, num_hidden_channels, hidden_channel_height, hidden_channel_width):
        super(MuZeroConsistencyNetwork, self).__init__()
        self.porjection_in_dim = num_hidden_channels * hidden_channel_height * hidden_channel_width
        self.projection = nn.Sequential(
            nn.Linear(self.porjection_in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 256),
        )

    def forward(self, hidden_state, with_grad: bool):
        # return hidden_state
        # only the branch of proj + pred can share the gradients
        hidden_state = hidden_state.view(-1, self.porjection_in_dim)
        proj = self.projection(hidden_state)

        # with grad, use proj_head
        if with_grad:
            proj = self.projection_head(proj)
            # return proj
            return {"proj": proj}
        else:
            # return proj.detach()
            return {"proj": proj.detach()}


class ValueNetwork(nn.Module):
    def __init__(self, num_channels, channel_height, channel_width, num_value_hidden_channels):
        super(ValueNetwork, self).__init__()
        self.channel_height = channel_height
        self.channel_width = channel_width
        self.conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(channel_height * channel_width, num_value_hidden_channels)
        self.fc2 = nn.Linear(num_value_hidden_channels, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(-1, self.channel_height * self.channel_width)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x


class DiscreteValueNetwork(nn.Module):
    def __init__(self, num_channels, channel_height, channel_width, num_value_hidden_channels, value_size):
        super(DiscreteValueNetwork, self).__init__()
        self.channel_height = channel_height
        self.channel_width = channel_width
        self.hidden_channels = math.ceil(value_size / (channel_height * channel_width))
        self.conv = nn.Conv2d(num_channels, self.hidden_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(self.hidden_channels)
        self.fc1 = nn.Linear(channel_height * channel_width * self.hidden_channels, num_value_hidden_channels)
        self.fc2 = nn.Linear(num_value_hidden_channels, value_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(-1, self.channel_height * self.channel_width * self.hidden_channels)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
