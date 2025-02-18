#include "mcts.h"

namespace minizero::actor {

void MCTSNode::reset()
{
    depth_ = 0;
    num_children_ = 0;
    hidden_state_data_index_ = -1;
    mean_ = 0.0f;
    count_ = 0.0f;
    total_count_ = 0.0f;
    virtual_loss_ = 0.0f;

    policy_ = 0.0f;
    policy_logit_ = 0.0f;
    policy_noise_ = 0.0f;
    option_child_policy_ = 0.0f;
    option_child_policy_noise_ = 0.0f;
    primitive_child_policy_ = 0.0f;
    primitive_child_policy_noise_ = 0.0f;
    value_ = 0.0f;
    reward_ = 0.0f;
    total_reward_ = 0.0f;
    forward_actions_.clear();
    option_child_actions_.clear();
    option_legal_actions_.clear();
    first_child_ = nullptr;
    option_child_ = nullptr;
    parent_ = nullptr;
    action_ = Action();
}

void MCTSNode::add(float value, float weight /* = 1.0f */)
{
    if (count_ + weight <= 0) {
        reset();
    } else {
        count_ += weight;
        mean_ += weight * (value - mean_) / count_;
    }
}

void MCTSNode::remove(float value, float weight /* = 1.0f */)
{
    if (count_ - weight <= 0) {
        reset();
    } else {
        count_ -= weight;
        mean_ -= weight * (value - mean_) / count_;
    }
}

float MCTSNode::getNormalizedMean(const std::map<float, int>& tree_value_bound, float parent_total_reward /* = 0.0f */, int parent_depth /* = 0 */) const
{
    float value = (mean_ - parent_total_reward) / std::pow(config::actor_mcts_reward_discount, parent_depth);
    if (config::actor_mcts_value_rescale) {
        if (tree_value_bound.size() < 2) { return 1.0f; }
        const float value_lower_bound = tree_value_bound.begin()->first;
        const float value_upper_bound = tree_value_bound.rbegin()->first;
        value = (value - value_lower_bound) / (value_upper_bound - value_lower_bound);
        value = fmin(1, fmax(-1, 2 * value - 1)); // normalize to [-1, 1]
    }
    value = (action_.getPlayer() == env::charToPlayer(config::actor_mcts_value_flipping_player) ? -value : value); // flip value according to player
    value = (value * count_ - virtual_loss_) / getCountWithVirtualLoss();                                          // value with virtual loss
    return value;
}

float MCTSNode::getNormalizedPUCTScore(int total_simulation, float normalized_mean, float policy, float count, float init_q_value /* = -1.0f */) const
{
    float puct_bias = config::actor_mcts_puct_init + log((1 + total_simulation + config::actor_mcts_puct_base) / config::actor_mcts_puct_base);
    float value_u = (puct_bias * policy * sqrt(total_simulation)) / (1 + count);
    float value_q = (count == 0 ? init_q_value : normalized_mean);
    return value_u + value_q;
}

std::string MCTSNode::toString() const
{
    std::ostringstream oss;
    oss.precision(4);
    oss << std::fixed << "p = " << policy_
        << ", p_logit = " << policy_logit_
        << ", p_noise = " << policy_noise_
        << ", option_p = " << option_child_policy_
        << ", option_p_noise = " << option_child_policy_noise_
        << ", v = " << value_
        << ", r = " << reward_
        << ", mean = " << mean_
        << ", count = " << count_
        << ", total_count = " << total_count_;
    return oss.str();
}

void MCTS::reset()
{
    Tree::reset();
    tree_hidden_state_data_.reset();
    tree_value_bound_.clear();
}

bool MCTS::isResign(const MCTSNode* selected_node) const
{
    float root_win_rate = getRootNode()->getNormalizedMean(tree_value_bound_);
    float action_win_rate = selected_node->getNormalizedMean(tree_value_bound_);
    return (-root_win_rate < config::actor_resign_threshold && action_win_rate < config::actor_resign_threshold);
}

MCTSNode* MCTS::selectChildByMaxCount(const MCTSNode* node) const
{
    assert(node && !node->isLeaf());
    float max_count = 0.0f;
    MCTSNode* selected = nullptr;
    for (int i = 0; i < node->getNumChildren(); ++i) {
        MCTSNode* child = node->getChild(i);
        if (child->getCount() <= max_count) { continue; }
        max_count = child->getCount();
        selected = child;
    }
    assert(selected != nullptr);
    return selected;
}

MCTSNode* MCTS::selectChildBySoftmaxCount(const MCTSNode* node, const bool enable_option, float temperature /* = 1.0f */, float value_threshold /* = 0.1f */) const
{
    assert(node && !node->isLeaf());
    MCTSNode* selected = nullptr;
    MCTSNode* best_child = selectChildByMaxCount(node);
    float best_mean = best_child->getNormalizedMean(tree_value_bound_, node->getTotalReward(), node->getDepth());
    float sum = 0.0f;
    int option_first_id = -1;
    float option_count = 0.0f;
    float option_mean = 0.0f;
    bool use_option = false;
    MCTSNode* option_child = node->getOptionChild();
    const std::vector<Action> option_child_actions = node->getOptionChildActions();
    if (!option_child_actions.empty()) {
        option_first_id = option_child_actions[0].getActionID();
        option_count = option_child->getCount();
        Action action(num_action_, option_child->getAction().getPlayer());
        std::vector<int> option;
        for (auto& a : option_child_actions) { option.push_back(a.getActionID()); }
        action.setOption(option);
        option_child->setAction(action);
        option_mean = option_child->getNormalizedMean(tree_value_bound_, node->getTotalReward(), node->getDepth());
        // flip value same as primitive child
        option_mean = (option_child->getAction().getPlayer() == node->getChild(0)->getAction().getPlayer() ? option_mean : -option_mean);
        float count = std::pow(option_count, 1 / temperature);
        if (count == 0 || (option_mean < best_mean - value_threshold) || !enable_option) {
            // do nothing
        } else {
            use_option = true;
            sum += count;
            float rand = utils::Random::randReal(sum);
            if (selected == nullptr || rand < count) { selected = option_child; }
        }
    }
    for (int i = 0; i < node->getNumChildren(); ++i) {
        MCTSNode* child = node->getChild(i);
        float count = child->getCount();
        child->setTotalCount(count);
        float mean = child->getNormalizedMean(tree_value_bound_, node->getTotalReward(), node->getDepth());
        if (child->getAction().getActionID() == option_first_id) {
            option_child->setTotalCount(count);
            if (use_option) {
                mean = (count - option_count) > 0.0f ? (mean * count - option_mean * option_count) / (count - option_count) : 0.0f;
                count -= option_count;
            }
        }
        count = std::pow(count, 1 / temperature);
        if (count == 0 || (mean < best_mean - value_threshold)) { continue; }
        sum += count;
        float rand = utils::Random::randReal(sum);
        if (selected == nullptr || rand < count) { selected = child; }
    }
    assert(selected != nullptr);
    return selected;
}

std::string MCTS::getSearchDistributionString() const
{
    const MCTSNode* root = getRootNode();
    std::ostringstream oss;
    for (int i = 0; i < root->getNumChildren(); ++i) {
        MCTSNode* child = root->getChild(i);
        if (child->getCount() == 0) { continue; }
        oss << (oss.str().empty() ? "" : ",")
            << child->getAction().getActionID() << ":" << child->getCount();
    }
    return oss.str();
}

std::vector<MCTSNode*> MCTS::selectFromNode(MCTSNode* start_node)
{
    assert(start_node);
    MCTSNode* node = start_node;
    std::vector<MCTSNode*> node_path{node};
    while (!node->isLeaf()) {
        MCTSNode* parent_node = node;
        MCTSNode* option_node = parent_node->getOptionChild();
        std::vector<Action> option_actions = parent_node->getOptionChildActions();
        node = selectChildByPUCTScore(node);
        node_path.push_back(node);
        if ((!parent_node->getOptionChildActions().empty() && option_actions[0].getActionID() == node->getAction().getActionID()) && selectOptionChild(parent_node, node, option_node)) {
            // pushback nodes until reach the option node
            for (size_t i = 1; i < option_actions.size(); ++i) {
                node = selectChildByAction(node, option_actions[i]);
                node_path.push_back(node);
            }
        } else {
            option_actions.clear();
            option_actions.push_back(node->getAction());
        }
        if (node->getHiddenStateDataIndex() == -1) {
            node->setParent(parent_node);
            node->setForwardActions(option_actions);
            break;
        }
    }
    return node_path;
}

void MCTS::expand(MCTSNode* leaf_node, const std::vector<ActionCandidate>& action_candidates)
{
    assert(leaf_node && action_candidates.size() > 0);
    const std::vector<Action>& option_actions = leaf_node->getOptionChildActions();
    const std::vector<std::vector<Action>>& option_legal_actions = leaf_node->getOptionLegalActions();
    if (leaf_node->isLeaf()) {
        leaf_node->setFirstChild(allocateNodes(action_candidates.size()));
        leaf_node->setNumChildren(action_candidates.size());
        for (size_t i = 0; i < action_candidates.size(); ++i) {
            const auto& candidate = action_candidates[i];
            MCTSNode* child = leaf_node->getChild(i);
            child->reset();
            child->setAction(candidate.action_);
            child->setPolicy(candidate.policy_);
            child->setPolicyLogit(candidate.policy_logit_);
            child->setDepth(leaf_node->getDepth() + 1);
        }
    } else {
        for (size_t i = 0; i < action_candidates.size(); ++i) {
            const auto& candidate = action_candidates[i];
            MCTSNode* child = selectChildByAction(leaf_node, candidate.action_);
            assert(candidate.action_.getActionID() == child->getAction().getActionID());
            child->setPolicy(candidate.policy_);
            child->setPolicyLogit(candidate.policy_logit_);
        }
    }
    if (!option_actions.empty()) {
        MCTSNode* cur_parent = selectChildByAction(leaf_node, option_actions[0]);
        // first action is at the child node, start from i=1
        for (size_t i = 1; i < option_actions.size(); ++i) {
            if (!cur_parent->isLeaf()) {
                cur_parent = selectChildByAction(cur_parent, option_actions[i]);
                continue;
            }
            const auto& expand_actions = option_legal_actions[i];
            cur_parent->setFirstChild(allocateNodes(expand_actions.size()));
            cur_parent->setNumChildren(expand_actions.size());
            for (size_t j = 0; j < expand_actions.size(); ++j) {
                MCTSNode* cur_child = cur_parent->getChild(j);
                cur_child->reset();
                cur_child->setAction(expand_actions[j]);
                cur_child->setDepth(cur_parent->getDepth() + 1);
            }
            cur_parent = selectChildByAction(cur_parent, option_actions[i]);
        }
        leaf_node->setOptionChild(cur_parent);
    }
}

void MCTS::backup(const std::vector<MCTSNode*>& node_path, const float value, const float reward /* = 0.0f */)
{
    assert(node_path.size() > 0);
    const float discount = config::actor_mcts_reward_discount;
    MCTSNode* backup_node = node_path.back();
    backup_node->setValue(value);
    backup_node->setReward(reward);
    if (backup_node->getParent() != nullptr) { // non-root nodes
        backup_node->setTotalReward(backup_node->getParent()->getTotalReward() + std::pow(discount, backup_node->getParent()->getDepth()) * reward);
    }
    float total_value = backup_node->getTotalReward() + std::pow(discount, backup_node->getDepth()) * backup_node->getValue();
    for (int i = static_cast<int>(node_path.size() - 1); i >= 0; --i) {
        MCTSNode* node = node_path[i];
        float old_mean = node->getMean();
        node->add(total_value);
        updateTreeValueBound(old_mean, node->getMean());
    }
}

MCTSNode* MCTS::selectChildByAction(const MCTSNode* node, const Action& action) const
{
    assert(node && !node->isLeaf());
    for (int i = 0; i < node->getNumChildren(); ++i) {
        MCTSNode* child = node->getChild(i);
        if (child->getAction().getActionID() == action.getActionID()) { return child; }
    }
    return nullptr;
}

bool MCTS::selectOptionChild(const MCTSNode* parent, const MCTSNode* primitive_child, const MCTSNode* option_child) const
{
    // compare with black's point of view
    assert(primitive_child && option_child);
    assert(option_child->getDepth() > primitive_child->getDepth());
    const float option_child_policy = parent->getOptionChildPolicy();
    const float primitive_child_policy = parent->getPrimitiveChildPolicy();

    int total_simulation = primitive_child->getCountWithVirtualLoss();
    float total_mean = (total_simulation > 0 ? primitive_child->getNormalizedMean(tree_value_bound_, parent->getTotalReward(), parent->getDepth()) : 0.0f);
    total_mean = (primitive_child->getAction().getPlayer() == env::charToPlayer(config::actor_mcts_value_flipping_player) ? -total_mean : total_mean); // flip value according to player
    float option_count = option_child->getCountWithVirtualLoss();
    float option_mean = (option_count > 0 ? option_child->getNormalizedMean(tree_value_bound_, parent->getTotalReward(), parent->getDepth()) : 0.0f);
    option_mean = (option_child->getAction().getPlayer() == env::charToPlayer(config::actor_mcts_value_flipping_player) ? -option_mean : option_mean); // flip value according to player
    float primitive_count = total_simulation - option_count;
    float primitive_mean = primitive_count > 0 ? (total_mean * total_simulation - option_mean * option_count) / primitive_count : 0.0f;
    // TODO: other init q-value?
    float init_q_value = (total_mean * total_simulation - 1) / (total_simulation + 1); // parent->getValue() is black's winrate

    float puct_bias = config::actor_mcts_puct_init + log((1 + total_simulation + config::actor_mcts_puct_base) / config::actor_mcts_puct_base);
    float value_u_option = (puct_bias * option_child_policy * sqrt(total_simulation)) / (1 + option_count);
    float value_q_option = (option_count == 0 ? init_q_value : option_mean);
    float value_u_primitive = (puct_bias * primitive_child_policy * sqrt(total_simulation)) / (1 + primitive_count);
    float value_q_primitive = (primitive_count == 0 ? init_q_value : primitive_mean);
    return (value_u_option + value_q_option > value_u_primitive + value_q_primitive);
}

MCTSNode* MCTS::selectChildByPUCTScore(const MCTSNode* node) const
{
    assert(node && !node->isLeaf());
    MCTSNode* selected = nullptr;
    int total_simulation = std::max(1.0f, node->getCountWithVirtualLoss()) - 1;
    float init_q_value = calculateInitQValue(node);
    float best_score = std::numeric_limits<float>::lowest(), best_policy = std::numeric_limits<float>::lowest();
    for (int i = 0; i < node->getNumChildren(); ++i) {
        MCTSNode* child = node->getChild(i);
        float score = child->getNormalizedPUCTScore(total_simulation, child->getNormalizedMean(tree_value_bound_, node->getTotalReward(), node->getDepth()), child->getPolicy(), child->getCountWithVirtualLoss(), init_q_value);
        if (score < best_score || (score == best_score && child->getPolicy() <= best_policy)) { continue; }
        best_score = score;
        best_policy = child->getPolicy();
        selected = child;
    }
    assert(selected != nullptr);
    return selected;
}

float MCTS::calculateInitQValue(const MCTSNode* node) const
{
    // init Q value = avg Q value of all visited children + one loss
    assert(node && !node->isLeaf());
    float sum_of_win = 0.0f, sum = 0.0f;
    for (int i = 0; i < node->getNumChildren(); ++i) {
        MCTSNode* child = node->getChild(i);
        if (child->getCountWithVirtualLoss() == 0) { continue; }
        sum_of_win += child->getNormalizedMean(tree_value_bound_, node->getTotalReward(), node->getDepth());
        sum += 1;
    }
#if ATARI
    // explore more in Atari games (TODO: check if this method also performs better in board games)
    return (sum > 0 ? sum_of_win / sum : 1.0f);
#else
    return (sum_of_win - 1) / (sum + 1);
#endif
}

void MCTS::updateTreeValueBound(float old_value, float new_value)
{
    if (!config::actor_mcts_value_rescale) { return; }
    if (tree_value_bound_.count(old_value)) {
        assert(tree_value_bound_[old_value] > 0);
        --tree_value_bound_[old_value];
        if (tree_value_bound_[old_value] == 0) { tree_value_bound_.erase(old_value); }
    }
    ++tree_value_bound_[new_value];
}

} // namespace minizero::actor
