#pragma once

#include "base_env.h"
#include "configuration.h"
#include "random.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// Format for input data (GridWorld Map)
// XXXXXXXXXX
// X@    O  #
// X     #  X
// X   X   XX
// X O     XX
// X   O   XX
// X       XX
// X      XXX
// X #  XXXXX
// XXXXXXXXXX

namespace minizero::env::gridworld {

const std::string kGridWorldName = "gridworld";
const int kGridWorldNumPlayer = 1;
const int kMaxGridWorldBoardSize = 10;
const int kMaxGridWorldBoardSizeHeight = 10;
const int kMaxGridWorldBoardSizeWidth = 20;
const int kFixedFeatures = 2;
const int kChangedFeatures = 1;
const int kGridWorldResolution = kMaxGridWorldBoardSizeWidth * kMaxGridWorldBoardSizeHeight;
const int kGridWorldHiddenChannelHeight = kMaxGridWorldBoardSizeHeight;
const int kGridWorldHiddenChannelWidth = kMaxGridWorldBoardSizeWidth;
const int kGridWorldActionSize = 4;
const int kGridWorldDiscreteValueSize = 601;
constexpr float reward_step = -1.f;

const std::string kGridWorldActionName[] = {
    "up", "down", "left", "right"};

class GridWorldAction : public BaseAction {
public:
    GridWorldAction() : BaseAction() {}
    GridWorldAction(int action_id, Player player) : BaseAction(action_id, player) {}
    GridWorldAction(const std::vector<std::string>& action_string_args) : BaseAction()
    {
        player_ = Player::kPlayer1;
        if (action_string_args[1] == "up") {
            action_id_ = 0;
        } else if (action_string_args[1] == "down") {
            action_id_ = 1;
        } else if (action_string_args[1] == "left") {
            action_id_ = 2;
        } else if (action_string_args[1] == "right") {
            action_id_ = 3;
        }
    } // TODO
    inline Player nextPlayer() const override { return Player::kPlayer1; }
    inline std::string toConsoleString() const
    {
        if (option_.empty()) { return kGridWorldActionName[action_id_]; }
        std::string action_name = "";
        for (auto& action_id : option_) {
            action_name += (action_name == "" ? "" : "-") + kGridWorldActionName[action_id];
        }
        return action_name;
    }
};

class GridWorldEnv : public BaseEnv<GridWorldAction> {
public:
    GridWorldEnv()
    {
        assert(getBoardSize() <= kMaxGridWorldBoardSize);
        reset();
    }

    void reset() override;
    void reset(const std::string& map);
    bool act(const GridWorldAction& action) override;
    bool act(const std::vector<std::string>& action_string_args) override;
    std::vector<GridWorldAction> getLegalActions() const override;
    bool isLegalAction(const GridWorldAction& action) const override;
    bool isTerminal() const override;
    float getReward() const override { return reward_; }
    std::vector<float> getActionFeatures(const GridWorldAction& action, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getActionFeatures(const std::vector<GridWorldAction>& action, utils::Rotation rotation = utils::Rotation::kRotationNone) const;
    inline int getNumInputChannels() const override { return past_n_moves_ * kChangedFeatures + kFixedFeatures + 1; } // only need to record wall and goal one time
    inline int getNumActionFeatureChannels() const override { return config::option_seq_length; }
    inline int getInputChannelHeight() const override { return kMaxGridWorldBoardSizeHeight; }
    inline int getInputChannelWidth() const override { return kMaxGridWorldBoardSizeWidth; }
    inline int getHiddenChannelHeight() const override { return kGridWorldHiddenChannelHeight; }
    inline int getHiddenChannelWidth() const override { return kGridWorldHiddenChannelWidth; }
    inline int getPolicySize() const override { return kGridWorldActionSize; }
    inline int getDiscreteValueSize() const override { return kGridWorldDiscreteValueSize; }
    std::string toString() const override;
    inline std::string name() const override { return kGridWorldName + "_" + std::to_string(getBoardSize()) + "x" + std::to_string(getBoardSize()); }
    inline int getNumPlayer() const override { return kGridWorldNumPlayer; }
    inline int getRotatePosition(int position, utils::Rotation rotation) const override { return utils::getPositionByRotating(utils::Rotation::kRotationNone, position, getBoardSize()); }
    inline int getRotateAction(int action_id, utils::Rotation rotation) const override { return getRotatePosition(action_id, utils::Rotation::kRotationNone); };
    inline std::string getMap() const { return map_; }
    void updatePlayerPosition(int direction);
    void initializeFeatureMaps();
    float getEvalScore(bool is_resign = false) const override { return total_reward_; }
    inline int getNumMoves() const { return num_moves_made; }
    std::vector<float> getFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::string selectMap(const std::string& folder_name, int sub_m, bool randomly_select_m);
    int getBoardSize() const { return board_size_; }

private:
    const int board_size_ = 10;
    const int maximum_num_moves = 200;
    const int past_n_moves_ = 2;
    int max_width_;
    int max_height_;
    int num_moves_made;
    std::string map_;
    std::pair<int, int> player_position_;
    std::pair<int, int> goal_position_;
    // Define possible movement directions: up, down, left, right
    const std::vector<std::pair<int, int>> directions_ = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    std::vector<std::vector<char>> board_;
    std::vector<std::vector<float>> past_n_moves_features_;
    std::vector<float> returned_featureMaps_;
    float reward_;
    float total_reward_;
};

class GridWorldEnvLoader : public BaseEnvLoader<GridWorldAction, GridWorldEnv> {
public:
    void loadFromEnvironment(const GridWorldEnv& env, const std::vector<std::vector<std::pair<std::string, std::string>>>& action_info_history = {}) override
    {
        BaseEnvLoader::loadFromEnvironment(env, action_info_history);
        addTag("MAP", env.getMap());
    }
    inline std::string getMap() const { return getTag("MAP"); }
    std::vector<float> getFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getActionFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getPrimitiveActionFeatures(const int action_id, utils::Rotation rotation = utils::Rotation::kRotationNone) const;
    std::vector<float> getValue(const int pos) const override { return toDiscreteValue(pos < static_cast<int>(action_pairs_.size()) ? utils::transformValue(calculateNStepValue(pos)) : 0.0f); }
    std::vector<float> getReward(const int pos) const override;
    inline std::vector<float> getPrimitiveReward(const int pos) const { return toDiscreteValue(pos < static_cast<int>(action_pairs_.size()) ? utils::transformValue(BaseEnvLoader::getReward(pos)[0]) : 0.0f); }
    float getPriority(const int pos) const override { return fabs(calculateNStepValue(pos) - BaseEnvLoader::getValue(pos)[0]) + 1e-6; }

    inline std::string name() const override { return kGridWorldName + std::to_string(kMaxGridWorldBoardSize) + "x" + std::to_string(kMaxGridWorldBoardSize); }
    inline int getPolicySize() const override { return 4; }
    inline int getRotatePosition(int position, utils::Rotation rotation) const override { return position; }
    inline int getRotateAction(int action_id, utils::Rotation rotation) const override { return action_id; }

private:
    float calculateNStepValue(const int pos) const;
    std::vector<float> toDiscreteValue(float value) const;
};

} // namespace minizero::env::gridworld
