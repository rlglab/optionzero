#include "gridworld.h"
#include "color_message.h"
#include "random.h"
#include "sgf_loader.h"
#include <algorithm>
#include <bitset>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
namespace minizero::env::gridworld {

using namespace minizero::utils;

std::string GridWorldEnv::selectMap(const std::string& folder_name, int sub_m, bool randomly_select_m)
{
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(folder_name)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path().string());
        }
    }

    if (files.empty()) {
        std::cerr << "No files found in the directory." << std::endl;
        return ""; // Return an empty string if no files are found
    }

    // Generate a random index
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, files.size() - 1);
    int random_index = dis(gen);

    std::ifstream fileStream(files[random_index]);
    std::string line;
    int row = 0;

    // Clear the existing board, actions, and positions
    board_.clear();
    actions_.clear();
    player_position_ = {0, 0}; // Assuming starting position is (0, 0)
    goal_position_ = {1, 1};
    map_.clear();

    max_width_ = 0;
    max_height_ = 0;
    bool set_player = false;
    bool set_goal = false;
    // Resize the board and initialize with empty spaces
    board_.resize(kMaxGridWorldBoardSizeHeight, std::vector<char>(kMaxGridWorldBoardSizeWidth, ' '));

    while (std::getline(fileStream, line)) {
        if (row < kMaxGridWorldBoardSizeHeight) {
            int col_count = 0;

            for (int col = 0; col < std::min(static_cast<int>(line.size()), kMaxGridWorldBoardSizeWidth); ++col) {
                if (line[col] != '\n') {
                    if (line[col] == ' ') {
                        map_ += 'E';
                    } else {
                        map_ += line[col];
                    }
                    ++col_count;
                }

                char element = line[col];
                board_[row][col] = element;

                // Update the player position
                if (element == '@') {
                    set_player = true;
                    player_position_ = {row, col};
                }

                // Update the goal positions
                if (element == '#' || element == '*') {
                    set_goal = true;
                    goal_position_ = std::make_pair(row, col);
                }
            }
            map_ += '!';
            ++row;

            if (col_count > max_width_) {
                max_width_ = col_count;
            }

        } else {
            break; // Ignore additional lines beyond board_size_
        }
    }

    std::srand(static_cast<unsigned>(std::time(0)));
    int lower_bound = 1;
    int upper_bound = kGridWorldResolution;

    if (minizero::config::fix_player == true && !set_player) {
        player_position_ = {8, 1};
        board_[8][1] = '@';
        set_player = true;
    }
    if (minizero::config::fix_goal == true && !set_goal) {
        goal_position_ = {1, 16};
        board_[1][16] = '#';
        set_goal = true;
    }

    // set player at random
    while (!set_player) {
        int random_number1 = std::rand() % (upper_bound - lower_bound + 1);
        int rand_row = random_number1 / (kMaxGridWorldBoardSizeWidth);
        int rand_col = random_number1 % (kMaxGridWorldBoardSizeWidth);
        if (board_[rand_row][rand_col] == ' ' && (rand_row != goal_position_.first && rand_col != goal_position_.second)) {
            player_position_ = {rand_row, rand_col};
            set_player = true;
            board_[rand_row][rand_col] = '@';
        }
    }
    while (!set_goal) {
        int random_number2 = std::rand() % (upper_bound - lower_bound + 1);
        int rand_row = random_number2 / (kMaxGridWorldBoardSizeWidth);
        int rand_col = random_number2 % (kMaxGridWorldBoardSizeWidth);
        if (board_[rand_row][rand_col] == ' ' && (rand_row != player_position_.first && rand_col != player_position_.second)) {
            // if (board_[rand_row][rand_col] == ' ' && (std::sqrt(std::pow(rand_row - player_position_.first, 2) + std::pow(rand_col - player_position_.second, 2))) >= std::pow(distance_between_player_goal, 2)) {
            goal_position_ = {rand_row, rand_col};
            set_goal = true;
            board_[rand_row][rand_col] = '#';
        }
    }

    std::string new_map_rand;
    for (int i = 0; i < kMaxGridWorldBoardSizeHeight; i++) {
        for (int j = 0; j < kMaxGridWorldBoardSizeWidth; j++) {
            if (board_[i][j] == ' ') {
                new_map_rand += 'E';
            } else {
                new_map_rand += board_[i][j];
            }
        }
        new_map_rand += '!';
    }
    std::string new_map;
    int count = 0;
    int max_range = static_cast<int>(map_.length());

    for (int i = 0; i < max_range; i++) {
        if (map_[i] != '!') {
            new_map += map_[i];
            count++;
        } else {
            new_map.append(max_width_ - count, 'E');
            new_map += '!';
            count = 0;
        }
    }

    map_ = new_map_rand;

    fileStream.close(); // Close the file after reading
    actions_.clear();   // Clear actions after resetting the board

    return map_;
}

void GridWorldEnv::reset()
{
    reward_ = 0;
    total_reward_ = 200;
    num_moves_made = 0;
    turn_ = Player::kPlayer1;

    int sub_m = kGridWorldResolution;
    bool randomly_select_m = true;

    selectMap(minizero::config::env_gridworld_maps_dir, sub_m, randomly_select_m);

    initializeFeatureMaps();
}

void GridWorldEnv::initializeFeatureMaps()
{
    // Reset the features from the past n moves
    returned_featureMaps_.clear();
    // goal
    std::vector<float> tmp(kGridWorldResolution, 0.0f);
    tmp[goal_position_.first * kMaxGridWorldBoardSizeWidth + goal_position_.second] = 1.0f;
    returned_featureMaps_.insert(returned_featureMaps_.end(), tmp.begin(), tmp.end());
    // road
    for (int row = 0; row < kMaxGridWorldBoardSizeHeight; row++) {
        for (int col = 0; col < kMaxGridWorldBoardSizeWidth; col++) {
            returned_featureMaps_.push_back(board_[row][col] != 'X' ? 1.0f : 0.0f);
        }
    }
    tmp.resize(kGridWorldResolution, 0.0f);
    tmp[player_position_.first * kMaxGridWorldBoardSizeWidth + player_position_.second] = 1.0f;
    for (int i = 0; i < past_n_moves_; ++i) { returned_featureMaps_.insert(returned_featureMaps_.end(), tmp.begin(), tmp.end()); }
    // current step / 200
    tmp.resize(kGridWorldResolution, 0.0f);
    returned_featureMaps_.insert(returned_featureMaps_.end(), tmp.begin(), tmp.end());
}

void GridWorldEnv::reset(const std::string& map)
{
    num_moves_made = 0;
    std::string line; // No need for istringstream
    int row = 0;
    int col = 0;

    // Clear the existing board, actions, and positions
    board_.clear();
    actions_.clear();
    player_position_ = {0, 0}; // Assuming starting position is (0, 0)
    goal_position_ = {1, 1};

    map_ = map;

    // Resize the board and initialize with empty spaces
    board_.resize(kMaxGridWorldBoardSizeHeight, std::vector<char>(kMaxGridWorldBoardSizeWidth, ' '));

    int max_range = static_cast<int>(map.length());

    for (int i = 0; i < max_range; i++) {
        char element = map[i];

        if (element == '!') {
            // Move to the next row
            ++row;
            col = 0; // Reset column index for the new row
        } else {
            if (row < kMaxGridWorldBoardSizeHeight && col < kMaxGridWorldBoardSizeWidth) {
                // Update the board with the current element
                if (element == 'E') {
                    board_[row][col] = ' ';
                } else {
                    board_[row][col] = element;
                }

                // Update the player position
                if (element == '@') {
                    player_position_ = {row, col};
                }

                // Update the goal positions
                if (element == '#' || element == '*') {
                    goal_position_ = std::make_pair(row, col);
                }
                ++col; // Move to the next column
            }
        }
    }

    initializeFeatureMaps();
}
bool GridWorldEnv::act(const GridWorldAction& action)
{
    int direction = -1;
    reward_ = -1;

    if (action.getActionID() < kGridWorldActionSize) {
        // actions_.push_back(action);
        if (!isLegalAction(action)) { return false; }
        actions_.push_back(action);
        num_moves_made = num_moves_made + 1;

        // Get the direction from the action
        direction = action.getActionID();

        // Update player and box positions based on the direction
        updatePlayerPosition(direction);
        total_reward_ += reward_;

    } else {
        reward_ = 0;
        actions_.push_back(action);
        float discount = 1.0f;
        for (auto& action_id : action.getOption()) {
            float re = -1.0f;
            direction = action_id;
            GridWorldAction action_primitive(direction, turn_);
            // actions_.push_back(action);
            if (!isLegalAction(action_primitive)) { return false; }
            num_moves_made = num_moves_made + 1;

            // Get the direction from the action

            // Update player and box positions based on the direction
            updatePlayerPosition(direction);
            total_reward_ += re;
            reward_ += discount * re;
            discount *= config::actor_mcts_reward_discount;

            if (isTerminal()) return true;
        }
    }
    if (direction == -1) {
        return false;
    } else {
        return true;
    }
}

bool GridWorldEnv::act(const std::vector<std::string>& action_string_args)
{
    return act(GridWorldAction(action_string_args));
}

void GridWorldEnv::updatePlayerPosition(int direction)
{
    board_[player_position_.first][player_position_.second] = ' ';
    // shift returned_featureMaps_ histrory, only works when kChangedFeatures=1
    for (int i = kFixedFeatures; i < (kFixedFeatures + past_n_moves_ - 1); ++i) {
        std::copy(returned_featureMaps_.begin() + (i + 1) * kGridWorldResolution, returned_featureMaps_.begin() + (i + 2) * kGridWorldResolution, returned_featureMaps_.begin() + i * kGridWorldResolution);
    }
    int last_map_start_pos = (kFixedFeatures + past_n_moves_ - 1) * kGridWorldResolution;
    assert(returned_featureMaps_[last_map_start_pos + player_position_.first * kMaxGridWorldBoardSizeWidth + player_position_.second] == 1.0f);
    returned_featureMaps_[last_map_start_pos + player_position_.first * kMaxGridWorldBoardSizeWidth + player_position_.second] = 0.0f;

    // Calculate the new player position
    player_position_.first += directions_[direction].first;
    player_position_.second += directions_[direction].second;
    returned_featureMaps_[last_map_start_pos + player_position_.first * kMaxGridWorldBoardSizeWidth + player_position_.second] = 1.0f;
    bool reach_goal = (player_position_.first == goal_position_.first) && (player_position_.second == goal_position_.second);
    board_[player_position_.first][player_position_.second] = (reach_goal ? '*' : '@');
    // add one feature plane
    std::vector<float> tmp(kGridWorldResolution, static_cast<float>(num_moves_made) / static_cast<float>(maximum_num_moves));
    int step_map_start_plane = (kFixedFeatures + past_n_moves_);
    std::copy(tmp.begin(), tmp.end(), returned_featureMaps_.begin() + step_map_start_plane * kGridWorldResolution);
}

bool GridWorldEnv::isLegalAction(const GridWorldAction& action) const
{
    const int direction = action.getActionID();
    const int newPlayerRow = player_position_.first + directions_[direction].first;
    const int newPlayerCol = player_position_.second + directions_[direction].second;
    return board_[newPlayerRow][newPlayerCol] != 'X';
}

std::vector<GridWorldAction> GridWorldEnv::getLegalActions() const
{
    std::vector<GridWorldAction> legal_moves;

    for (size_t dir = 0; dir < directions_.size(); ++dir) {
        GridWorldAction action(dir, turn_);
        if (isLegalAction(action)) {
            // if return true, then it's a legal move
            legal_moves.push_back(action);
        }
    }
    return legal_moves;
}

bool GridWorldEnv::isTerminal() const
{
    bool reach_goal = (player_position_.first == goal_position_.first) && (player_position_.second == goal_position_.second);
    return (reach_goal || (num_moves_made >= maximum_num_moves));
}

std::vector<float> GridWorldEnv::getFeatures(utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
{
    // Get the features: player, walls, boxes, goals, empty spaces
    // We represent their location as 1, the rest will be 0; it's stored as bitset
    return returned_featureMaps_;
}

std::vector<float> GridWorldEnv::getActionFeatures(const GridWorldAction& action, utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
{
    return getActionFeatures(std::vector<GridWorldAction>{action}, rotation);
}

std::vector<float> GridWorldEnv::getActionFeatures(const std::vector<GridWorldAction>& action, utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
{
    int hidden_size = kGridWorldHiddenChannelWidth * kGridWorldHiddenChannelHeight;
    std::vector<float> action_features(getNumActionFeatureChannels() * hidden_size, 0.0f);
    int ones = kGridWorldResolution / kGridWorldActionSize;
    for (size_t i = 0; i < action.size(); ++i) {
        int action_id = action[i].getActionID();
        std::fill(action_features.begin() + i * hidden_size + action_id * ones, action_features.begin() + i * hidden_size + (action_id + 1) * ones, 1.0f);
    }
    return action_features;
}

std::string GridWorldEnv::toString() const
{
    // std::ostringstream outputString;
    // ostringstream << "goal: ("<< goal_position_.first << ", " << goal_position_.second << ")" <<std::endl;
    // ostringstream << "player: ("<< player_position_.first << ", " << player_position_.second << ")" <<std::endl;
    // return outputString.str();
    std::ostringstream outputString;
    std::string currentBoard;

    // Define symbols for GridWorld elements (modify as needed)
    const char playerSymbol = '@';
    const char boxSymbol = 'O';
    const char goalSymbol = '#';
    const char wallSymbol = 'X';
    const char emptySymbol = ' ';
    const char arrivedGoalSymbol = '*';

    // Map GridWorld symbols to ANSI escape codes (modify as needed)
    std::unordered_map<char, std::string> symbolToColor{
        {playerSymbol, "\033[48;2;0;0;0m"},         // Black for player
        {boxSymbol, "\033[48;2;255;0;0m"},          // Red for box
        {goalSymbol, "\033[48;2;0;255;0m"},         // Green for goal
        {wallSymbol, "\033[48;2;139;69;19m"},       // Brown for wall
        {emptySymbol, "\033[48;2;255;255;255m"},    // White for empty space
        {arrivedGoalSymbol, "\033[48;2;255;255;0m"} // Yellow for arrived goal space
    };

    outputString << std::string(5, '-') + std::string((kMaxGridWorldBoardSizeWidth - 1) * 4, '-') + "\n";

    for (int row = 0; row < kMaxGridWorldBoardSizeHeight; ++row) {
        outputString << "|" + std::string(1, ' ');

        for (int col = 0; col < kMaxGridWorldBoardSizeWidth; ++col) {
            outputString << symbolToColor.at(board_[row][col]) + "  \033[m" + std::string(1, ' ') + "|";
        }
        outputString << "\n";
        outputString << std::string(5, '-') + std::string((kMaxGridWorldBoardSizeWidth - 1) * 4, '-') + "\n";
    }
    outputString << currentBoard;
    return outputString.str();
}

std::vector<float> GridWorldEnvLoader::getFeatures(const int pos, utils::Rotation rotation /* = utils::Rotation::kRotationNone */) const
{
    GridWorldEnv env;
    env.reset(getMap());
    for (int i = 0; i < std::min(pos, static_cast<int>(action_pairs_.size())); ++i) { env.act(action_pairs_[i].first); }
    return env.getFeatures(rotation);
}

std::vector<float> GridWorldEnvLoader::getActionFeatures(const int pos, utils::Rotation rotation /* = utils::Rotation::kRotationNone */) const
{
    const int step_length = getStepLength(pos);
    int max_length = config::option_seq_length;
    int hidden_size = kGridWorldHiddenChannelHeight * kGridWorldHiddenChannelWidth;
    std::vector<float> action_features(max_length * hidden_size, 0.0f);
    std::vector<float> tmp(hidden_size, 0.0f);
    int total_length = 0;
    // option action features
    for (int length = 0; length < step_length; ++length) {
        int action_id = ((pos + length < static_cast<int>(action_pairs_.size())) ? action_pairs_[pos + length].first.getActionID() : utils::Random::randInt() % kGridWorldActionSize);
        if (action_id < kGridWorldActionSize) {
            tmp = getPrimitiveActionFeatures(action_id, rotation);
            std::copy(tmp.begin(), tmp.end(), action_features.begin() + total_length * hidden_size);
            ++total_length;
        } else {
            const std::vector<int> option = action_pairs_[pos + length].first.getOption();
            for (const auto& action_id : option) {
                if (total_length >= max_length) { break; }
                tmp = getPrimitiveActionFeatures(action_id, rotation);
                std::copy(tmp.begin(), tmp.end(), action_features.begin() + total_length * hidden_size);
                ++total_length;
            }
        }
    }
    return action_features;
}

std::vector<float> GridWorldEnvLoader::getPrimitiveActionFeatures(const int action_id, utils::Rotation rotation /* = utils::Rotation::kRotationNone */) const
{
    int hidden_size = kGridWorldHiddenChannelWidth * kGridWorldHiddenChannelHeight;
    std::vector<float> action_features(hidden_size, 0.0f);
    int ones = kGridWorldResolution / kGridWorldActionSize;
    std::fill(action_features.begin() + action_id * ones, action_features.begin() + (action_id + 1) * ones, 1.0f);
    return action_features;
}

std::vector<float> GridWorldEnvLoader::getReward(const int pos) const
{
    const int step_length = getStepLength(pos);
    float total_reward = 0.0f;
    float discount = 1.0f;
    for (int length = 0; length < step_length; ++length) {
        float reward = (pos + length < static_cast<int>(action_pairs_.size()) ? BaseEnvLoader::getReward(pos + length)[0] : 0.0f);
        int sequence_length = ((pos + length >= static_cast<int>(action_pairs_.size()) || action_pairs_[pos + length].first.getActionID() < kGridWorldActionSize) ? 1 : action_pairs_[pos + length].first.getOption().size());
        total_reward = total_reward + discount * reward;
        discount *= std::pow(config::actor_mcts_reward_discount, sequence_length);
    }
    return toDiscreteValue(utils::transformValue(total_reward));
}

float GridWorldEnvLoader::calculateNStepValue(const int pos) const
{
    assert(pos < static_cast<int>(action_pairs_.size()));

    // calculate n-step return by using EpisodicLifeEnv (end-of-life == end-of-episode)
    // reference: https://github.com/DLR-RM/stable-baselines3/blob/472ff8edb815070c405da913dcbe64c1a06e0e7d/stable_baselines3/common/atari_wrappers.py#L98
    float discount = 1.0f;
    float value = 0.0f;
    size_t index = pos;
    int total_length = 0;
    for (int step = 0; step < config::learner_n_step_return; ++step) {
        const int step_length = getStepLength(index);
        for (int length = 0; length < step_length; ++length) {
            if (index >= action_pairs_.size() || (action_pairs_[index].second.count("L") && std::stoi(action_pairs_[index].second.at("L")) > 0)) { return value; }
            value += discount * BaseEnvLoader::getReward(index)[0]; // reward in option
            int sequence_length = (action_pairs_[index].first.getActionID() < kGridWorldActionSize ? 1 : action_pairs_[index].first.getOption().size());
            discount *= std::pow(config::actor_mcts_reward_discount, sequence_length);
            total_length += sequence_length;
            ++index;
        }
        if (total_length >= config::learner_n_step_return) { break; }
    }
    value += ((index < action_pairs_.size() && !action_pairs_[index].second.count("L")) ? discount * BaseEnvLoader::getValue(index)[0] : 0.0f);
    return value;
}

std::vector<float> GridWorldEnvLoader::toDiscreteValue(float value) const
{
    std::vector<float> discrete_value(kGridWorldDiscreteValueSize, 0.0f);
    int value_floor = floor(value);
    int value_ceil = ceil(value);
    int shift = kGridWorldDiscreteValueSize / 2;
    int value_floor_shift = std::min(std::max(value_floor + shift, 0), kGridWorldDiscreteValueSize - 1);
    int value_ceil_shift = std::min(std::max(value_ceil + shift, 0), kGridWorldDiscreteValueSize - 1);
    if (value_floor == value_ceil) {
        discrete_value[value_floor_shift] = 1.0f;
    } else {
        discrete_value[value_floor_shift] = value_ceil - value;
        discrete_value[value_ceil_shift] = value - value_floor;
    }
    return discrete_value;
}

} // namespace minizero::env::gridworld
