#pragma once

#include "alphazero_network.h"
#include "base_actor.h"
#include "gumbel_zero.h"
#include "mcts.h"
#include "muzero_network.h"
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace minizero::actor {

class MCTSSearchData {
public:
    std::string search_info_;
    std::string selection_path_;
    MCTSNode* selected_node_;
    std::vector<MCTSNode*> node_path_;
    void clear();
};

class ZeroActor : public BaseActor {
public:
    ZeroActor(uint64_t tree_node_size)
        : tree_node_size_(tree_node_size)
    {
        option_count_ = 0;
        alphazero_network_ = nullptr;
        muzero_network_ = nullptr;
    }

    void reset() override;
    void resetSearch() override;
    Action think(bool with_play = false, bool display_board = false) override;
    void beforeNNEvaluation() override;
    void afterNNEvaluation(const std::shared_ptr<network::NetworkOutput>& network_output) override;
    bool isSearchDone() const override { return getMCTS()->reachMaximumSimulation(); }
    Action getSearchAction() const override { return mcts_search_data_.selected_node_->getAction(); }
    bool isResign() const override { return enable_resign_ && getMCTS()->isResign(mcts_search_data_.selected_node_); }
    std::string getSearchInfo() const override { return mcts_search_data_.search_info_; }
    void setNetwork(const std::shared_ptr<network::Network>& network) override;
    std::shared_ptr<Search> createSearch() override { return std::make_shared<MCTS>(tree_node_size_); }
    std::shared_ptr<MCTS> getMCTS() { return std::static_pointer_cast<MCTS>(search_); }
    const std::shared_ptr<MCTS> getMCTS() const { return std::static_pointer_cast<MCTS>(search_); }

protected:
    std::vector<std::pair<std::string, std::string>> getActionInfo() const override;
    std::string getMCTSPolicy() const override { return (config::actor_use_gumbel ? gumbel_zero_.getMCTSPolicy(getMCTS()) : getMCTS()->getSearchDistributionString()); }
    std::string getMCTSValue() const override { return std::to_string(getMCTS()->getRootNode()->getMean()); }
    std::string getEnvReward() const override;

    virtual void step();
    virtual void handleSearchDone();
    virtual MCTSNode* decideActionNode();
    virtual void addNoiseToNodeChildren(MCTSNode* node);
    virtual std::vector<MCTSNode*> selection() { return (config::actor_use_gumbel ? gumbel_zero_.selection(getMCTS()) : getMCTS()->select()); }

    std::vector<MCTS::ActionCandidate> calculateAlphaZeroActionPolicy(const Environment& env_transition, const std::shared_ptr<network::AlphaZeroNetworkOutput>& alphazero_output, const utils::Rotation& rotation);
    std::vector<MCTS::ActionCandidate> calculateMuZeroActionPolicy(MCTSNode* leaf_node, const std::shared_ptr<network::MuZeroNetworkOutput>& muzero_output);
    virtual Environment getEnvironmentTransition(const std::vector<MCTSNode*>& node_path);

    bool enable_resign_;
    bool enable_option_;
    int option_count_;
    int p_max_id_;
    std::string used_options_;
    std::string depth_;
    std::string option_str_;
    GumbelZero gumbel_zero_;
    uint64_t tree_node_size_;
    MCTSSearchData mcts_search_data_;
    utils::Rotation feature_rotation_;
    std::shared_ptr<network::AlphaZeroNetwork> alphazero_network_;
    std::shared_ptr<network::MuZeroNetwork> muzero_network_;

private:
    void setAlphaZeroOptionInfo(MCTSNode* leaf_node, Environment& env_transition, const std::shared_ptr<network::AlphaZeroNetworkOutput>& alphazero_output, const utils::Rotation& rotation);
    void setMuZeroOptionInfo(MCTSNode* leaf_node, const std::shared_ptr<network::MuZeroNetworkOutput>& muzero_output);
    std::pair<std::vector<Action>, std::vector<std::vector<Action>>> calculateLegalOption(Environment& env_transition, env::Player turn, const std::vector<std::vector<float>> option, utils::Rotation rotation = utils::Rotation::kRotationNone);
    std::vector<Action> calculateOption(env::Player turn, const std::vector<std::vector<float>> option, utils::Rotation rotation = utils::Rotation::kRotationNone);
    void setOptionInfo(MCTSNode* leaf_node, const std::vector<std::vector<float>> option, const std::vector<Action> option_actions, const std::vector<std::vector<Action>> option_legal_actions, utils::Rotation rotation = utils::Rotation::kRotationNone);
};

} // namespace minizero::actor
