#include "zero_actor.h"
#include "random.h"
#include "time_system.h"
#include <algorithm>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace minizero::actor {

using namespace minizero;
using namespace network;

void MCTSSearchData::clear()
{
    search_info_ = "";
    selection_path_ = "";
    selected_node_ = nullptr;
    node_path_.clear();
}

void ZeroActor::reset()
{
    BaseActor::reset();
    enable_resign_ = (utils::Random::randReal() < config::zero_disable_resign_ratio ? false : true);
    enable_option_ = (utils::Random::randReal() < config::zero_disable_option_ratio ? false : true);
}

void ZeroActor::resetSearch()
{
    BaseActor::resetSearch();
    option_count_ = 0;
    p_max_id_ = -1;
    depth_ = "";
    used_options_ = "";
    option_str_ = "";
    mcts_search_data_.selection_path_ = "";
    mcts_search_data_.node_path_.clear();
    getMCTS()->getRootNode()->setAction(Action(-1, env::getPreviousPlayer(env_.getTurn(), env_.getNumPlayer())));
    getMCTS()->setNumAction(env_.getPolicySize());
}

Action ZeroActor::think(bool with_play /*= false*/, bool display_board /*= false*/)
{
    resetSearch();
    boost::posix_time::ptime start_ptime = utils::TimeSystem::getLocalTime();
    while (!isSearchDone()) {
        step();
        int spent_million_second = (utils::TimeSystem::getLocalTime() - start_ptime).total_milliseconds();
        if (config::actor_mcts_think_time_limit > 0 && spent_million_second >= config::actor_mcts_think_time_limit * 1000) { break; }
    }
    if (!isSearchDone()) { handleSearchDone(); }
    if (with_play) { act(getSearchAction()); }
    if (display_board) { std::cerr << env_.toString() << mcts_search_data_.search_info_ << std::endl; }
    return getSearchAction();
}

void ZeroActor::beforeNNEvaluation()
{
    mcts_search_data_.node_path_ = selection();

    std::ostringstream oss;
    for (size_t i = 1; i < mcts_search_data_.node_path_.size(); ++i) { oss << mcts_search_data_.node_path_[i]->getAction().getActionID() << "-"; }
    std::string selection_path_str = oss.str();
    if (!selection_path_str.empty()) {
        selection_path_str.pop_back();
        mcts_search_data_.selection_path_ += ":" + selection_path_str;
    }

    if (alphazero_network_) {
        Environment env_transition = getEnvironmentTransition(mcts_search_data_.node_path_);
        feature_rotation_ = config::actor_use_random_rotation_features ? static_cast<utils::Rotation>(utils::Random::randInt() % static_cast<int>(utils::Rotation::kRotateSize)) : utils::Rotation::kRotationNone;
        nn_evaluation_batch_id_ = alphazero_network_->pushBack(env_transition.getFeatures(feature_rotation_));
    } else if (muzero_network_) {
        if (getMCTS()->getNumSimulation() == 0) { // initial inference for root node
            nn_evaluation_batch_id_ = muzero_network_->pushBackInitialData(env_.getFeatures());
        } else { // for non-root nodes
            const std::vector<MCTSNode*>& node_path = mcts_search_data_.node_path_;
            MCTSNode* leaf_node = node_path.back();
            MCTSNode* parent_node = leaf_node->getParent();
            assert(parent_node && parent_node->getHiddenStateDataIndex() != -1);
            const std::vector<float>& hidden_state = getMCTS()->getTreeHiddenStateData().getData(parent_node->getHiddenStateDataIndex()).hidden_state_;
            assert(!leaf_node->getForwardActions().empty());
            nn_evaluation_batch_id_ = muzero_network_->pushBackRecurrentData(hidden_state, env_.getActionFeatures(leaf_node->getForwardActions()));
        }
    } else {
        assert(false);
    }
}

void ZeroActor::afterNNEvaluation(const std::shared_ptr<NetworkOutput>& network_output)
{
    const std::vector<MCTSNode*>& node_path = mcts_search_data_.node_path_;
    MCTSNode* leaf_node = node_path.back();
    if (alphazero_network_) {
        Environment env_transition = getEnvironmentTransition(node_path);
        if (!env_transition.isTerminal()) {
            std::shared_ptr<AlphaZeroNetworkOutput> alphazero_output = std::static_pointer_cast<AlphaZeroNetworkOutput>(network_output);
            getMCTS()->expand(leaf_node, calculateAlphaZeroActionPolicy(env_transition, alphazero_output, feature_rotation_));
            getMCTS()->backup(node_path, alphazero_output->value_, env_transition.getReward());
        } else {
            getMCTS()->backup(node_path, env_transition.getEvalScore(), env_transition.getReward());
        }
        if (leaf_node->getForwardActions().size() > 1) {
            ++option_count_;
            int depth = node_path.size();
            depth_ += (depth_ == "" ? "" : "-") + std::to_string(depth);
            used_options_ += (used_options_ == "" ? "" : ":");
            for (size_t i = 0; i < leaf_node->getForwardActions().size(); ++i) {
                used_options_ += std::to_string(leaf_node->getForwardActions()[i].getActionID()) + (i == leaf_node->getForwardActions().size() - 1 ? "" : "-");
            }
        }
        leaf_node->setHiddenStateDataIndex(0);
    } else if (muzero_network_) {
        std::shared_ptr<MuZeroNetworkOutput> muzero_output = std::static_pointer_cast<MuZeroNetworkOutput>(network_output);
        getMCTS()->expand(leaf_node, calculateMuZeroActionPolicy(leaf_node, muzero_output));
        assert(leaf_node == getMCTS()->getRootNode() || !leaf_node->getForwardActions().empty());
        if (leaf_node->getForwardActions().size() > 1) {
            ++option_count_;
            int depth = node_path.size();
            depth_ += (depth_ == "" ? "" : "-") + std::to_string(depth);
            used_options_ += (used_options_ == "" ? "" : ":");
            for (size_t i = 0; i < leaf_node->getForwardActions().size(); ++i) {
                used_options_ += std::to_string(leaf_node->getForwardActions()[i].getActionID()) + (i == leaf_node->getForwardActions().size() - 1 ? "" : "-");
            }
        }
        getMCTS()->backup(node_path, muzero_output->value_, muzero_output->reward_);
        leaf_node->setHiddenStateDataIndex(getMCTS()->getTreeHiddenStateData().store(HiddenStateData(muzero_output->hidden_state_)));
    } else {
        assert(false);
    }
    if (leaf_node == getMCTS()->getRootNode()) { addNoiseToNodeChildren(leaf_node); }
    if (isSearchDone()) { handleSearchDone(); }
    if (config::actor_use_gumbel) { gumbel_zero_.sequentialHalving(getMCTS()); }
}

void ZeroActor::setNetwork(const std::shared_ptr<network::Network>& network)
{
    assert(network);
    alphazero_network_ = nullptr;
    muzero_network_ = nullptr;
    if (network->getNetworkTypeName() == "alphazero") {
        alphazero_network_ = std::static_pointer_cast<AlphaZeroNetwork>(network);
    } else if (network->getNetworkTypeName() == "muzero" || network->getNetworkTypeName() == "muzero_atari" || network->getNetworkTypeName() == "muzero_gridworld") {
        muzero_network_ = std::static_pointer_cast<MuZeroNetwork>(network);
    } else {
        assert(false);
    }
    assert((alphazero_network_ && !muzero_network_) || (!alphazero_network_ && muzero_network_));
}

std::vector<std::pair<std::string, std::string>> ZeroActor::getActionInfo() const
{
    // ignore recording mcts action info if there is no search
    if (getMCTS()->getRootNode()->getCount() > 0) {
        auto action_info = BaseActor::getActionInfo();
        action_info.push_back({"C", std::to_string(option_count_)});
        action_info.push_back({"D", depth_});
        action_info.push_back({"OP", used_options_});
        action_info.push_back({"SP", utils::compressString(mcts_search_data_.selection_path_)});
        if (!enable_option_) { action_info.push_back({"DIS", "1"}); }
        action_info.push_back({"PMAX", std::to_string(p_max_id_)});
        std::vector<Action> option_actions = getMCTS()->getRootNode()->getOptionChildActions();
        if (!option_actions.empty()) {
            std::string s = "";
            for (auto& action : option_actions) { s += (s == "" ? "" : "-") + std::to_string(action.getActionID()); }
            action_info.push_back({"OP1", s});
        }
        return action_info;
    }
    return {};
}

std::string ZeroActor::getEnvReward() const
{
    std::ostringstream oss;
    oss << env_.getReward();
    return oss.str();
}

void ZeroActor::step()
{
    assert(alphazero_network_ || muzero_network_);
    int num_simulation = getMCTS()->getNumSimulation();
    int num_simulation_left = config::actor_num_simulation + 1 - num_simulation;
    int batch_size = std::min(config::actor_mcts_think_batch_size,
                              (alphazero_network_ || num_simulation > 0) ? num_simulation_left : 1 /* initial inference for root node */);
    assert(batch_size > 0);

    std::vector<std::tuple<int, utils::Rotation, decltype(mcts_search_data_.node_path_)>> batch_queries; // batch id, rotation, search path
    for (int batch_id = 0; batch_id < batch_size; batch_id++) {
        beforeNNEvaluation();
        assert(nn_evaluation_batch_id_ == batch_id);
        if (mcts_search_data_.node_path_.back()->getVirtualLoss() == 0) {
            batch_queries.emplace_back(nn_evaluation_batch_id_, feature_rotation_, mcts_search_data_.node_path_);
        }
        for (auto node : mcts_search_data_.node_path_) { node->addVirtualLoss(); }
    }
    auto network_output = alphazero_network_ ? alphazero_network_->forward()
                                             : (num_simulation == 0 ? muzero_network_->initialInference() : muzero_network_->recurrentInference());
    for (auto& query : batch_queries) {
        nn_evaluation_batch_id_ = std::get<0>(query);
        feature_rotation_ = std::get<1>(query);
        mcts_search_data_.node_path_ = std::get<2>(query);
        afterNNEvaluation(network_output[nn_evaluation_batch_id_]);
        auto virtual_loss = mcts_search_data_.node_path_.back()->getVirtualLoss();
        for (auto node : mcts_search_data_.node_path_) { node->removeVirtualLoss(virtual_loss); }
    }
}

void ZeroActor::handleSearchDone()
{
    mcts_search_data_.selected_node_ = decideActionNode();
    const Action action = getSearchAction();
    std::ostringstream oss;
    oss << "model file name: " << config::nn_file_name << std::endl
        << utils::TimeSystem::getTimeString("[Y/m/d H:i:s.f] ")
        << "move number: " << env_.getActionHistory().size()
        << ", action: " << action.toConsoleString()
        << " (" << action.getActionID() << ")"
        << ", reward: " << env_.getReward()
        << ", player: " << env::playerToChar(action.getPlayer())
        << ", option count: " << option_count_
        << ", depth: " << depth_
        << ", option: " << used_options_
        << ", enable: " << enable_option_ << std::endl
        << option_str_ << std::endl;
    if (config::actor_mcts_value_rescale) { oss << ", value bound: (" << getMCTS()->getTreeValueBound().begin()->first << ", " << getMCTS()->getTreeValueBound().rbegin()->first << ")"; }
    oss << std::endl
        << "p_max_id_: " << p_max_id_ << std::endl
        << "  root node info: " << getMCTS()->getRootNode()->toString() << std::endl
        << "action node info: " << mcts_search_data_.selected_node_->toString() << std::endl;
    mcts_search_data_.search_info_ = oss.str();
}

MCTSNode* ZeroActor::decideActionNode()
{
    if (config::actor_use_gumbel) {
        return gumbel_zero_.decideActionNode(getMCTS());
    } else {
        if (config::actor_select_action_by_count) {
            return getMCTS()->selectChildByMaxCount(getMCTS()->getRootNode());
        } else if (config::actor_select_action_by_softmax_count) {
            return getMCTS()->selectChildBySoftmaxCount(getMCTS()->getRootNode(), enable_option_, config::actor_select_action_softmax_temperature);
        }

        assert(false);
        return nullptr;
    }
}

void ZeroActor::addNoiseToNodeChildren(MCTSNode* node)
{
    assert(node && node->getNumChildren() > 0);
    if (config::actor_use_dirichlet_noise) {
        const float epsilon = config::actor_dirichlet_noise_epsilon;
        std::vector<float> dirichlet_noise = utils::Random::randDirichlet(config::actor_dirichlet_noise_alpha, node->getNumChildren());
        for (int i = 0; i < node->getNumChildren(); ++i) {
            MCTSNode* child = node->getChild(i);
            child->setPolicyNoise(dirichlet_noise[i]);
            child->setPolicy((1 - epsilon) * child->getPolicy() + epsilon * dirichlet_noise[i]);
        }
        if (enable_option_) {
            std::vector<float> option_dirichlet_noise = utils::Random::randDirichlet(config::actor_dirichlet_noise_alpha, 2);
            node->setOptionChildPolicyNoise(option_dirichlet_noise[0]);
            node->setOptionChildPolicy((1 - epsilon) * node->getOptionChildPolicy() + epsilon * node->getOptionChildPolicyNoise());
            node->setPrimitiveChildPolicyNoise(option_dirichlet_noise[1]);
            node->setPrimitiveChildPolicy(1.0f - node->getOptionChildPolicy());
        }
    } else if (config::actor_use_gumbel_noise) {
        std::vector<float> gumbel_noise = utils::Random::randGumbel(node->getNumChildren());
        for (int i = 0; i < node->getNumChildren(); ++i) {
            MCTSNode* child = node->getChild(i);
            child->setPolicyNoise(gumbel_noise[i]);
            child->setPolicyLogit(child->getPolicyLogit() + gumbel_noise[i]);
        }
    }
}

std::vector<MCTS::ActionCandidate> ZeroActor::calculateAlphaZeroActionPolicy(const Environment& env_transition, const std::shared_ptr<network::AlphaZeroNetworkOutput>& alphazero_output, const utils::Rotation& rotation)
{
    assert(alphazero_network_);
    std::vector<MCTS::ActionCandidate> action_candidates;
    for (size_t action_id = 0; action_id < alphazero_output->policy_.size(); ++action_id) {
        Action action(action_id, env_transition.getTurn());
        if (!env_transition.isLegalAction(action)) { continue; }
        int rotated_id = env_transition.getRotateAction(action_id, rotation);
        action_candidates.push_back(MCTS::ActionCandidate(action, alphazero_output->policy_[rotated_id], alphazero_output->policy_logits_[rotated_id]));
    }
    sort(action_candidates.begin(), action_candidates.end(), [](const MCTS::ActionCandidate& lhs, const MCTS::ActionCandidate& rhs) {
        return lhs.policy_ > rhs.policy_;
    });
    return action_candidates;
}

std::vector<MCTS::ActionCandidate> ZeroActor::calculateMuZeroActionPolicy(MCTSNode* leaf_node, const std::shared_ptr<network::MuZeroNetworkOutput>& muzero_output)
{
    assert(muzero_network_);
    setMuZeroOptionInfo(leaf_node, muzero_output);
    std::vector<MCTS::ActionCandidate> action_candidates;
    env::Player turn = leaf_node->getAction().nextPlayer();
    if (leaf_node == getMCTS()->getRootNode()) {
        std::vector<float>& policy = muzero_output->policy_;
        p_max_id_ = std::distance(policy.begin(), std::max_element(policy.begin(), policy.end()));
    }
    for (size_t action_id = 0; action_id < muzero_output->policy_.size(); ++action_id) {
        const Action action(action_id, turn);
        if (leaf_node == getMCTS()->getRootNode() && !env_.isLegalAction(action)) { continue; }
        action_candidates.push_back(MCTS::ActionCandidate(action, muzero_output->policy_[action_id], muzero_output->policy_logits_[action_id]));
    }
    sort(action_candidates.begin(), action_candidates.end(), [](const MCTS::ActionCandidate& lhs, const MCTS::ActionCandidate& rhs) {
        return lhs.policy_ > rhs.policy_;
    });
    return action_candidates;
}

Environment ZeroActor::getEnvironmentTransition(const std::vector<MCTSNode*>& node_path)
{
    Environment env = env_;
    for (size_t i = 1; i < node_path.size(); ++i) { env.act(node_path[i]->getAction()); }
    return env;
}

void ZeroActor::setMuZeroOptionInfo(MCTSNode* leaf_node, const std::shared_ptr<network::MuZeroNetworkOutput>& muzero_output)
{
    const std::vector<std::vector<float>> option = muzero_output->option_;
    std::vector<Action> option_actions = calculateOption(leaf_node->getAction().nextPlayer(), muzero_output->option_);
    if (option_actions.empty()) { return; }
    std::vector<Action> all_actions, opp_all_actions;
    env::Player turn = leaf_node->getAction().nextPlayer();
    env::Player opp_turn = leaf_node->getAction().getPlayer();
    for (int action_id = 0; action_id < env_.getPolicySize(); ++action_id) { // exclude stop
        const Action action(action_id, turn);
        const Action opp_action(action_id, opp_turn);
        all_actions.push_back(action);
        opp_all_actions.push_back(opp_action);
    }
    std::vector<std::vector<Action>> option_legal_actions;
    for (size_t i = 0; i < option_actions.size(); ++i) { option_legal_actions.push_back(i % 2 == 0 ? all_actions : opp_all_actions); }
    // check whether root node's option is legal
    if (leaf_node == getMCTS()->getRootNode()) {
        option_legal_actions[0] = env_.getLegalActions();
#if ATARI
        for (size_t i = 0; i < option_actions.size(); ++i) {
            if (!env_.isLegalAction(option_actions[i])) {
                option_actions.resize(i);
                option_legal_actions.resize(i);
                break;
            }
        }
#else
        Environment env = env_;
        for (size_t i = 0; i < option_actions.size(); ++i) {
            if (!env.isLegalAction(option_actions[i])) {
                option_actions.resize(i);
                option_legal_actions.resize(i);
                break;
            }
            env.act(option_actions[i]);
        }
#endif
    }
    setOptionInfo(leaf_node, muzero_output->option_, option_actions, option_legal_actions);
}

std::vector<Action> ZeroActor::calculateOption(env::Player turn, const std::vector<std::vector<float>> option, utils::Rotation rotation /* = utils::Rotation::kRotationNone */)
{
    std::vector<Action> option_actions;
    for (auto& p : option) {
        int max_index = std::distance(p.begin(), std::max_element(p.begin(), p.end()));
        max_index = env_.getRotateAction(max_index, utils::reversed_rotation[static_cast<int>(rotation)]);
        Action action(max_index, turn);
        if (max_index == env_.getPolicySize()) { break; }
        option_actions.push_back(action);
        turn = action.nextPlayer();
    }
    if (option_actions.size() <= 1) { option_actions.clear(); }
    return option_actions;
}

void ZeroActor::setOptionInfo(MCTSNode* leaf_node, const std::vector<std::vector<float>> option, const std::vector<Action> option_actions, const std::vector<std::vector<Action>> option_legal_actions, utils::Rotation rotation /* = utils::Rotation::kRotationNone */)
{
    if (leaf_node == getMCTS()->getRootNode()) {
        option_str_ += "cur option: ";
        for (size_t i = 0; i < option_actions.size(); ++i) {
            option_str_ += std::to_string(option_actions[i].getActionID()) + ":" + std::to_string(option[i][env_.getRotateAction(option_actions[i].getActionID(), rotation)]) + ",";
        }
    }
    if (option_actions.empty()) { return; }
    int rotated_option_id = env_.getRotateAction(option_actions.back().getActionID(), rotation);
    // int rotated_primitive_id = env_.getRotateAction(option_actions[0].getActionID(), rotation);
    // leaf_node->setOptionChildPolicy(std::min(option[option_actions.size() - 1][rotated_option_id] / option[0][rotated_primitive_id], 1.0f));
    leaf_node->setOptionChildPolicy(option[option_actions.size() - 1][rotated_option_id]);
    leaf_node->setPrimitiveChildPolicy(1.0f - leaf_node->getOptionChildPolicy());
    leaf_node->setOptionChildActions(option_actions);
    leaf_node->setOptionLegalActions(option_legal_actions);
}

} // namespace minizero::actor
