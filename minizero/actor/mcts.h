#pragma once

#include "configuration.h"
#include "environment.h"
#include "random.h"
#include "search.h"
#include "tree.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace minizero::actor {

class MCTSNode : public TreeNode {
public:
    MCTSNode() { reset(); }

    void reset() override;
    virtual void add(float value, float weight = 1.0f);
    virtual void remove(float value, float weight = 1.0f);
    virtual float getNormalizedMean(const std::map<float, int>& tree_value_bound, float parent_total_reward = 0.0f, int parent_depth = 0) const;
    virtual float getNormalizedPUCTScore(int total_simulation, float normalized_mean, float policy, float count, float init_q_value = -1.0f) const;
    std::string toString() const override;
    bool displayInTreeLog() const override { return count_ > 0; }

    // setter
    inline void setDepth(int depth) { depth_ = depth; }
    inline void setHiddenStateDataIndex(int hidden_state_data_index) { hidden_state_data_index_ = hidden_state_data_index; }
    inline void setMean(float mean) { mean_ = mean; }
    inline void setCount(float count) { count_ = count; }
    inline void setTotalCount(float total_count) { total_count_ = total_count; }
    inline void addVirtualLoss(float num = 1.0f) { virtual_loss_ += num; }
    inline void removeVirtualLoss(float num = 1.0f) { virtual_loss_ -= num; }
    inline void setPolicy(float policy) { policy_ = policy; }
    inline void setPolicyLogit(float policy_logit) { policy_logit_ = policy_logit; }
    inline void setPolicyNoise(float policy_noise) { policy_noise_ = policy_noise; }
    inline void setOptionChildPolicy(float option_child_policy) { option_child_policy_ = option_child_policy; }
    inline void setOptionChildPolicyNoise(float option_child_policy_noise) { option_child_policy_noise_ = option_child_policy_noise; }
    inline void setPrimitiveChildPolicy(float primitive_child_policy) { primitive_child_policy_ = primitive_child_policy; }
    inline void setPrimitiveChildPolicyNoise(float primitive_child_policy_noise) { primitive_child_policy_noise_ = primitive_child_policy_noise; }
    inline void setValue(float value) { value_ = value; }
    inline void setReward(float reward) { reward_ = reward; }
    inline void setTotalReward(float total_reward) { total_reward_ = total_reward; }
    inline void setOptionChildActions(std::vector<Action> option_child_actions) { option_child_actions_ = option_child_actions; }
    inline void setOptionLegalActions(std::vector<std::vector<Action>> option_legal_actions) { option_legal_actions_ = option_legal_actions; }
    inline void setForwardActions(std::vector<Action> forward_actions) { forward_actions_ = forward_actions; }
    inline void setFirstChild(MCTSNode* first_child) { TreeNode::setFirstChild(first_child); }
    inline void setOptionChild(MCTSNode* option_child) { TreeNode::setOptionChild(option_child); }
    inline void setParent(MCTSNode* parent) { TreeNode::setParent(parent); }

    // getter
    inline int getDepth() const { return depth_; }
    inline int getHiddenStateDataIndex() const { return hidden_state_data_index_; }
    inline float getMean() const { return mean_; }
    inline float getCount() const { return count_; }
    inline float getTotalCount() const { return total_count_; }
    inline float getCountWithVirtualLoss() const { return count_ + virtual_loss_; }
    inline float getVirtualLoss() const { return virtual_loss_; }
    inline float getPolicy() const { return policy_; }
    inline float getPolicyLogit() const { return policy_logit_; }
    inline float getPolicyNoise() const { return policy_noise_; }
    inline float getOptionChildPolicy() const { return option_child_policy_; }
    inline float getOptionChildPolicyNoise() const { return option_child_policy_noise_; }
    inline float getPrimitiveChildPolicy() const { return primitive_child_policy_; }
    inline float getPrimitiveChildPolicyNoise() const { return primitive_child_policy_noise_; }
    inline float getValue() const { return value_; }
    inline float getReward() const { return reward_; }
    inline float getTotalReward() const { return total_reward_; }
    inline std::vector<Action> getOptionChildActions() const { return option_child_actions_; }
    inline std::vector<std::vector<Action>> getOptionLegalActions() const { return option_legal_actions_; }
    inline std::vector<Action> getForwardActions() const { return forward_actions_; }
    inline virtual MCTSNode* getChild(int index) const override { return (index < num_children_ ? static_cast<MCTSNode*>(first_child_) + index : nullptr); }
    inline MCTSNode* getOptionChild() const override { return static_cast<MCTSNode*>(option_child_); }
    inline MCTSNode* getParent() const override { return static_cast<MCTSNode*>(parent_); }

protected:
    int hidden_state_data_index_;
    int depth_;
    float mean_;
    float count_;
    float virtual_loss_;
    float policy_;
    float policy_logit_;
    float policy_noise_;
    float option_child_policy_;
    float option_child_policy_noise_;
    float primitive_child_policy_;
    float primitive_child_policy_noise_;
    float value_;
    float reward_;
    float total_reward_;
    float total_count_;
    std::vector<Action> option_child_actions_;
    std::vector<std::vector<Action>> option_legal_actions_;
    std::vector<Action> forward_actions_;
};

class HiddenStateData {
public:
    HiddenStateData(const std::vector<float>& hidden_state)
        : hidden_state_(hidden_state) {}
    std::vector<float> hidden_state_;
};
typedef TreeData<HiddenStateData> TreeHiddenStateData;

class MCTS : public Tree, public Search {
public:
    class ActionCandidate {
    public:
        Action action_;
        float policy_;
        float policy_logit_;
        float option_policy_;
        float option_policy_logit_;
        ActionCandidate(const Action& action, const float& policy, const float& policy_logit)
            : action_(action), policy_(policy), policy_logit_(policy_logit) {}
    };

    MCTS(uint64_t tree_node_size)
        : Tree(tree_node_size) {}

    void reset() override;
    virtual bool isResign(const MCTSNode* selected_node) const;
    virtual MCTSNode* selectChildByMaxCount(const MCTSNode* node) const;
    virtual MCTSNode* selectChildBySoftmaxCount(const MCTSNode* node, const bool enable_option, float temperature = 1.0f, float value_threshold = 0.1f) const;
    virtual std::string getSearchDistributionString() const;
    virtual std::vector<MCTSNode*> select() { return selectFromNode(getRootNode()); }
    virtual std::vector<MCTSNode*> selectFromNode(MCTSNode* start_node);
    virtual void expand(MCTSNode* leaf_node, const std::vector<ActionCandidate>& action_candidates);
    virtual void backup(const std::vector<MCTSNode*>& node_path, const float value, const float reward = 0.0f);
    virtual MCTSNode* selectChildByAction(const MCTSNode* node, const Action& action) const;
    virtual bool selectOptionChild(const MCTSNode* parent, const MCTSNode* node_primitive, const MCTSNode* node_option) const;

    inline MCTSNode* allocateNodes(int size) { return static_cast<MCTSNode*>(Tree::allocateNodes(size)); }
    inline int getNumSimulation() const { return getRootNode()->getCount(); }
    inline bool reachMaximumSimulation() const { return (getNumSimulation() == config::actor_num_simulation + 1); }
    inline MCTSNode* getRootNode() { return static_cast<MCTSNode*>(Tree::getRootNode()); }
    inline const MCTSNode* getRootNode() const { return static_cast<const MCTSNode*>(Tree::getRootNode()); }
    inline TreeHiddenStateData& getTreeHiddenStateData() { return tree_hidden_state_data_; }
    inline const TreeHiddenStateData& getTreeHiddenStateData() const { return tree_hidden_state_data_; }
    inline std::map<float, int>& getTreeValueBound() { return tree_value_bound_; }
    inline const std::map<float, int>& getTreeValueBound() const { return tree_value_bound_; }
    inline void setNumAction(int num_action) { num_action_ = num_action; }
    inline int getNumAction() const { return num_action_; }

protected:
    TreeNode* createTreeNodes(uint64_t tree_node_size) override { return new MCTSNode[tree_node_size]; }
    TreeNode* getNodeIndex(int index) override { return getRootNode() + index; }

    virtual MCTSNode* selectChildByPUCTScore(const MCTSNode* node) const;
    virtual float calculateInitQValue(const MCTSNode* node) const;
    virtual void updateTreeValueBound(float old_value, float new_value);

    int num_action_;
    std::map<float, int> tree_value_bound_;
    TreeHiddenStateData tree_hidden_state_data_;
};

} // namespace minizero::actor
