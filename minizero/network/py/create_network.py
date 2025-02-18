from .alphazero_network import AlphaZeroNetwork
from .muzero_network import MuZeroNetwork
from .muzero_atari_network import MuZeroAtariNetwork
from .muzero_gridworld_network import MuZeroGridWorldNetwork


def create_network(game_name="tietactoe",
                   num_input_channels=4,
                   input_channel_height=3,
                   input_channel_width=3,
                   num_hidden_channels=16,
                   hidden_channel_height=3,
                   hidden_channel_width=3,
                   num_action_feature_channels=1,
                   num_blocks=1,
                   action_size=9,
                   option_seq_length=1,
                   option_action_size=19,
                   num_value_hidden_channels=256,
                   discrete_value_size=601,
                   network_type_name="alphazero"):

    network = None
    if network_type_name == "alphazero":
        network = AlphaZeroNetwork(game_name,
                                   num_input_channels,
                                   input_channel_height,
                                   input_channel_width,
                                   num_hidden_channels,
                                   hidden_channel_height,
                                   hidden_channel_width,
                                   num_blocks,
                                   action_size,
                                   num_value_hidden_channels,
                                   discrete_value_size)
    elif network_type_name == "muzero":
        if "atari" in game_name:
            network = MuZeroAtariNetwork(game_name,
                                         num_input_channels,
                                         input_channel_height,
                                         input_channel_width,
                                         num_hidden_channels,
                                         hidden_channel_height,
                                         hidden_channel_width,
                                         num_action_feature_channels,
                                         num_blocks,
                                         action_size,
                                         option_seq_length,
                                         option_action_size,
                                         num_value_hidden_channels,
                                         discrete_value_size)
        elif "gridworld" in game_name:
            network = MuZeroGridWorldNetwork(game_name,
                                             num_input_channels,
                                             input_channel_height,
                                             input_channel_width,
                                             num_hidden_channels,
                                             hidden_channel_height,
                                             hidden_channel_width,
                                             num_action_feature_channels,
                                             num_blocks,
                                             action_size,
                                             option_seq_length,
                                             option_action_size,
                                             num_value_hidden_channels,
                                             discrete_value_size)
        else:
            network = MuZeroNetwork(game_name,
                                    num_input_channels,
                                    input_channel_height,
                                    input_channel_width,
                                    num_hidden_channels,
                                    hidden_channel_height,
                                    hidden_channel_width,
                                    num_action_feature_channels,
                                    num_blocks,
                                    action_size,
                                    option_seq_length,
                                    option_action_size,
                                    num_value_hidden_channels,
                                    discrete_value_size)
    else:
        assert False

    return network
