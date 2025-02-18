#pragma once

#include "configuration.h"
#include "rotation.h"
#include "sgf_loader.h"
#include "utils.h"
#include "vector_map.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace minizero::env {

using namespace minizero;

// only support up to two players currently
enum class Player {
    kPlayerNone = 0,
    kPlayer1 = 1,
    kPlayer2 = 2,
    kPlayerSize = 3
};

char playerToChar(Player p);
Player charToPlayer(char c);
Player getNextPlayer(Player player, int num_player);
Player getPreviousPlayer(Player player, int num_player);

class BaseAction {
public:
    BaseAction() : action_id_(-1), player_(Player::kPlayerNone) {}
    BaseAction(int action_id, Player player) : action_id_(action_id), player_(player) {}
    virtual ~BaseAction() = default;

    virtual Player nextPlayer() const = 0;
    virtual std::string toConsoleString() const = 0;

    inline int getActionID() const { return action_id_; }
    inline Player getPlayer() const { return player_; }
    inline std::vector<int> getOption() const { return option_; }

    inline void setActionID(int action_id) { action_id_ = action_id; }
    inline void setOption(std::vector<int> option) { option_ = option; }

protected:
    int action_id_;
    Player player_;

    std::vector<int> option_;
};

template <class T>
class GamePair {
public:
    GamePair() { reset(); }
    GamePair(T black, T white) : black_(black), white_(white) {}

    inline void reset() { black_ = white_ = T(); }
    inline T& get(Player p) { return (p == Player::kPlayer1 ? black_ : white_); }
    inline const T& get(Player p) const { return (p == Player::kPlayer1 ? black_ : white_); }
    inline void set(Player p, const T& value) { (p == Player::kPlayer1 ? black_ : white_) = value; }
    inline void set(const T& black, const T& white)
    {
        black_ = black;
        white_ = white;
    }
    inline bool operator==(const GamePair<T>& rhs) const { return (black_ == rhs.black_ && white_ == rhs.white_); }

private:
    T black_;
    T white_;
};

template <class Action>
class BaseEnv {
public:
    BaseEnv() {}
    virtual ~BaseEnv() = default;

    virtual void reset() = 0;
    virtual bool act(const Action& action) = 0;
    virtual bool act(const std::vector<std::string>& action_string_args) = 0;
    virtual std::vector<Action> getLegalActions() const = 0;
    virtual bool isLegalAction(const Action& action) const = 0;
    virtual bool isTerminal() const = 0;
    virtual float getReward() const = 0;
    virtual float getEvalScore(bool is_resign = false) const = 0;
    virtual std::vector<float> getFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const = 0;
    virtual std::vector<float> getActionFeatures(const Action& action, utils::Rotation rotation = utils::Rotation::kRotationNone) const = 0;
    virtual int getNumInputChannels() const = 0;
    virtual int getNumActionFeatureChannels() const = 0;
    virtual int getInputChannelHeight() const = 0;
    virtual int getInputChannelWidth() const = 0;
    virtual int getHiddenChannelHeight() const = 0;
    virtual int getHiddenChannelWidth() const = 0;
    virtual int getPolicySize() const = 0;
    virtual inline int getOptionPolicySize() const { return getPolicySize() + 1; }
    virtual int getDiscreteValueSize() const = 0;
    virtual int getRotatePosition(int position, utils::Rotation rotation) const = 0;
    virtual int getRotateAction(int action_id, utils::Rotation rotation) const = 0;
    virtual std::string toString() const = 0;
    virtual std::string name() const = 0;
    virtual int getNumPlayer() const = 0;
    virtual void setTurn(Player p) { turn_ = p; }
    // can't count moves by size of action if act option in env
    virtual int getNumMoves() const { return actions_.size(); }

    inline Player getTurn() const { return turn_; }
    inline const std::vector<Action>& getActionHistory() const { return actions_; }
    inline const std::vector<std::string>& getObservationHistory() const { return observations_; }

protected:
    Player turn_;
    std::vector<Action> actions_;
    std::vector<std::string> observations_;
};

template <class Action, class Env>
class BaseEnvLoader {
public:
    BaseEnvLoader() {}
    virtual ~BaseEnvLoader() = default;

    typedef minizero::utils::VectorMap<std::string, std::string> Tags;
    typedef minizero::utils::VectorMap<std::string, std::string> ActionInfo;

public:
    virtual void reset()
    {
        sgf_content_.clear();
        tags_.clear();
        tags_.insert({"GM", name()});
        tags_.insert({"RE", "0"});
        action_pairs_.clear();
    }

    virtual bool loadFromFile(const std::string& file_name)
    {
        std::ifstream fin(file_name.c_str());
        if (!fin) { return false; }

        std::string line;
        std::string sgf_content;
        while (std::getline(fin, line)) {
            if (line.back() == '\r') { line.pop_back(); }
            if (line.empty()) { continue; }
            sgf_content += line;
        }
        return loadFromString(sgf_content);
    }

    virtual bool loadFromString(const std::string& content)
    {
        reset();
        sgf_content_ = content;
        std::string key, value;
        int state = '(';
        bool accept_move = false;
        bool escape_next = false;
        int board_size = minizero::config::env_board_size;
        for (char c : content) {
            switch (state) {
                case '(': // wait until record start
                    if (!accept_move) {
                        accept_move = (c == '(');
                    } else {
                        state = (c == ';') ? c : 'x';
                        accept_move = false;
                    }
                    break;
                case ';': // store key
                    if (c == ';') {
                        accept_move = true;
                    } else if (c == '[' || c == ')') {
                        state = c;
                    } else if (std::isgraph(c)) {
                        key += c;
                    }
                    break;
                case '[': // store value
                    if (c == '\\' && !escape_next) {
                        escape_next = true;
                    } else if (c != ']' || escape_next) {
                        value += c;
                        escape_next = false;
                    } else { // ready to store key-value pair
                        if (accept_move) {
                            int action_id = value.size() && std::isdigit(value[0]) ? std::stoi(value) : minizero::utils::SGFLoader::sgfStringToActionID(value, board_size);
                            action_pairs_.emplace_back().first = Action(action_id, charToPlayer(key[0]));
                            accept_move = false;
                        } else if (action_pairs_.size()) {
                            action_pairs_.back().second[key] = std::move(value);
                            if (key == "OP1") {
                                std::string s = action_pairs_.back().second[key];
                                std::vector<int> option;
                                size_t sz = 0;
                                while (!s.empty()) {
                                    int tmp = stoi(s, &sz);
                                    option.push_back(abs(tmp));
                                    s = s.substr(sz);
                                }
                                action_pairs_.back().first.setOption(option);
                            }
                        } else {
                            if (key == "SZ") { board_size = std::stoi(value); }
                            tags_[key] = std::move(value);
                        }
                        key.clear();
                        value.clear();
                        state = ';';
                    }
                    break;
                case ')': // end of record, do nothing
                    break;
            }
        }
        return state == ')';
    }

    virtual void loadFromEnvironment(const Env& env, const std::vector<std::vector<std::pair<std::string, std::string>>>& action_info_history = {})
    {
        reset();
        for (size_t i = 0; i < env.getActionHistory().size(); ++i) {
            addActionPair(env.getActionHistory()[i], action_info_history.size() > i ? ActionInfo(action_info_history[i]) : ActionInfo());
        }
        addTag("RE", std::to_string(env.getEvalScore()));

        // add observations
        std::string observations;
        for (const auto& obs : env.getObservationHistory()) { observations += obs; }
        addTag("OBS", utils::compressString(observations));
        assert(observations == utils::decompressString(getTag("OBS")));
    }

    virtual std::string toString() const
    {
        std::ostringstream oss;
        oss << "(;";
        for (const auto& t : tags_) { oss << t.first << "[" << escapeSGFString(t.second) << "]"; }
        for (const auto& p : action_pairs_) {
            oss << ";" << playerToChar(p.first.getPlayer()) << "[" << p.first.getActionID() << "]";
            for (const auto& info : p.second) { oss << info.first << "[" << escapeSGFString(info.second) << "]"; }
        }
        oss << ")";
        return oss.str();
    }

    virtual std::vector<float> getFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const
    {
        // a slow but naive method which simply replays the game again to get features
        Env env;
        for (int i = 0; i < std::min(pos, static_cast<int>(action_pairs_.size())); ++i) { env.act(action_pairs_[i].first); }
        return env.getFeatures(rotation);
    }

    virtual std::vector<float> getPolicy(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const
    {
        std::vector<float> policy(getPolicySize(), 0.0f);
        if (pos < static_cast<int>(action_pairs_.size())) {
            const std::string policy_distribution = action_pairs_[pos].second["P"];
            if (policy_distribution.empty()) {
                policy[getRotateAction(action_pairs_[pos].first.getActionID(), rotation)] = 1.0f;
            } else {
                std::string tmp;
                float total = 0.0f;
                std::istringstream iss(policy_distribution);
                while (std::getline(iss, tmp, ',')) {
                    int position = getRotateAction(std::stoi(tmp.substr(0, tmp.find(":"))), rotation);
                    float count = std::stof(tmp.substr(tmp.find(":") + 1));
                    if (position < getPolicySize()) {
                        policy[position] += count;
                    } else {
                        policy[action_pairs_[pos].first.getOption()[0]] += count;
                    }
                    total += count;
                }
                for (auto& p : policy) { p /= total; }
            }
        } else { // absorbing states
            std::fill(policy.begin(), policy.end(), 1.0f / getPolicySize());
        }
        return policy;
    }

    virtual std::vector<Action> getOption(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const
    {
        std::vector<Action> option;
        if (pos >= static_cast<int>(action_pairs_.size())) { return option; }
        for (auto& action_id : action_pairs_[pos].first.getOption()) {
            Action action(action_id, action_pairs_[pos].first.getPlayer());
            option.push_back(action);
        }
        return option;
    }

    virtual int getOptionLength(const int pos) const
    {
        return (pos < static_cast<int>(action_pairs_.size()) ? std::max(1, static_cast<int>(action_pairs_[pos].first.getOption().size())) : 1);
    }

    virtual int getStepLength(const int pos) const
    {
        if (pos >= static_cast<int>(action_pairs_.size()) || !action_pairs_[pos].second.count("DIS")) { return 1; }
        const std::vector<int> option_id = action_pairs_[pos].first.getOption();
        if (option_id.empty()) { return 1; }
        // only check whether the actions match, directly compare without rotation
        for (int length = 0; length < static_cast<int>(option_id.size()); ++length) {
            if (pos + length >= static_cast<int>(action_pairs_.size()) || action_pairs_[pos + length].first.getActionID() != option_id[length]) { return 1; }
        }
        return option_id.size();
    }

    virtual std::vector<float> getOptionPolicy(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const
    {
        size_t max_size = config::option_seq_length * getOptionPolicySize();
        const bool enable_option = (pos < static_cast<int>(action_pairs_.size()) ? !action_pairs_[pos].second.count("DIS") : false);
        std::vector<float> option_policy, tmp;
        int step_length = 0;
        while (option_policy.size() != max_size) {
            if (pos + step_length >= static_cast<int>(action_pairs_.size())) { // terminal
                tmp = std::vector<float>(getOptionPolicySize(), 0.0f);
                tmp[getPolicySize()] = 1.0f; // stop
                while (option_policy.size() != max_size) { option_policy.insert(option_policy.end(), tmp.begin(), tmp.end()); }
                break;
            } else {
                const int action_id = getRotateAction(action_pairs_[pos + step_length].first.getActionID(), rotation);
                std::vector<int> option = action_pairs_[pos + step_length].first.getOption();
                for (auto& o : option) { o = getRotateAction(o, rotation); }
                const int max_action_id = getMaxActionID(pos + step_length, rotation);
                const std::vector<float> policy = getPolicy(pos + step_length, rotation);
                if (step_length == 0) { // first head, learn distribution
                    tmp = policy;
                    tmp.push_back(0.0f); // stop
                    option_policy.insert(option_policy.end(), tmp.begin(), tmp.end());
                } else {
                    tmp = std::vector<float>(getOptionPolicySize(), 0.0f);
                    tmp[max_action_id] = policy[max_action_id];
                    tmp[getPolicySize()] = 1.0f - policy[max_action_id];
                    option_policy.insert(option_policy.end(), tmp.begin(), tmp.end());
                }
                // other heads
                if (action_id != getPolicySize() && action_id != max_action_id) {
                    tmp = std::vector<float>(getOptionPolicySize(), 0.0f);
                    tmp[getPolicySize()] = 1.0f; // stop
                    while (option_policy.size() != max_size) { option_policy.insert(option_policy.end(), tmp.begin(), tmp.end()); }
                    break;
                }
                if (enable_option && !option.empty()) {
                    if (action_id == getPolicySize() && option[0] == max_action_id) {
                        for (size_t i = 1; i < option.size(); ++i) {
                            if (option_policy.size() == max_size) { break; }
                            tmp = std::vector<float>(getOptionPolicySize(), 0.0f);
                            tmp[option[i]] = 1.0f;
                            option_policy.insert(option_policy.end(), tmp.begin(), tmp.end());
                        }
                    } else {
                        tmp = std::vector<float>(getOptionPolicySize(), 0.0f);
                        tmp[getPolicySize()] = 1.0f; // stop
                        while (option_policy.size() != max_size) { option_policy.insert(option_policy.end(), tmp.begin(), tmp.end()); }
                        break;
                    }
                }
                ++step_length;
            }
        }
        return option_policy;
    }

    virtual std::vector<float> getOptionLossScale(const int pos) const
    {
        std::vector<float> option_loss_scale(config::option_seq_length, 0.0f);
        const bool enable_option = (pos < static_cast<int>(action_pairs_.size()) ? !action_pairs_[pos].second.count("DIS") : false);
        int total_length = 0;
        int step_length = 0;
        while (total_length < config::option_seq_length) {
            if (pos + step_length >= static_cast<int>(action_pairs_.size())) {
                option_loss_scale[total_length] = 1.0f;
                ++total_length;
                ++step_length;
            } else {
                int action_id = action_pairs_[pos + step_length].first.getActionID();
                // compare action and option without rotation
                int max_action_id = getMaxActionID(pos + step_length);
                const std::vector<int>& option = action_pairs_[pos + step_length].first.getOption();
                if (max_action_id == getPolicySize()) {
                    for (int i = 0; i + total_length < config::option_seq_length; ++i) { option_loss_scale[i + total_length] = 1.0f; }
                    break;
                }
                option_loss_scale[total_length] = 1.0f;
                ++total_length;
                if (action_id != getPolicySize() && action_id != max_action_id) {
                    if (step_length == 0) { break; }
                    for (int i = 0; i + total_length < config::option_seq_length; ++i) { option_loss_scale[i + total_length] = 1.0f; }
                    break;
                }
                if (enable_option && !option.empty()) {
                    if (action_id == getPolicySize() && option[0] == max_action_id) {
                        for (size_t i = 1; i < option.size(); ++i) {
                            if (total_length >= config::option_seq_length) { break; }
                            option_loss_scale[total_length] = 1.0f;
                            ++total_length;
                        }
                    } else {
                        for (int i = 0; i + total_length < config::option_seq_length; ++i) { option_loss_scale[i + total_length] = 1.0f; }
                        break;
                    }
                }
                ++step_length;
            }
        }
        return option_loss_scale;
    }

    virtual int getMaxActionID(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const
    {
        if (pos >= static_cast<int>(action_pairs_.size())) { return -1; }
        const std::vector<float> policy = getPolicy(pos, rotation);
        const int max_index = std::distance(policy.begin(), std::max_element(policy.begin(), policy.end()));
        return (action_pairs_[pos].second.count("PMAX") ? getRotateAction(std::stoi(action_pairs_[pos].second.at("PMAX")), rotation) : max_index);
    }

    virtual std::pair<int, int> getDataRange() const
    {
        const std::string dlen = getTag("DLEN");
        if (dlen.empty()) {
            return {0, std::max(0, static_cast<int>(getActionPairs().size()) - 1)};
        } else {
            return {std::stoi(dlen), std::stoi(dlen.substr(dlen.find("-") + 1))};
        }
    }

    virtual std::vector<float> getValue(const int pos) const { return (pos < static_cast<int>(action_pairs_.size()) ? std::vector<float>{std::stof(action_pairs_[pos].second["V"])} : std::vector<float>{0.0f}); }
    virtual std::vector<float> getReward(const int pos) const { return (pos < static_cast<int>(action_pairs_.size()) ? std::vector<float>{std::stof(action_pairs_[pos].second["R"])} : std::vector<float>{0.0f}); }
    virtual bool setActionPairInfo(const int pos, const std::string& tag, const std::string value)
    {
        if (pos >= static_cast<int>(action_pairs_.size())) { return false; }
        action_pairs_[pos].second[tag] = value;
        return true;
    }
    virtual float getPriority(const int pos) const { return 1.0f; }

    virtual std::vector<float> getActionFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const = 0;
    virtual std::string name() const = 0;
    virtual int getPolicySize() const = 0;
    virtual inline int getOptionPolicySize() const { return getPolicySize() + 1; }
    virtual int getRotatePosition(int position, utils::Rotation rotation) const = 0;
    virtual int getRotateAction(int action_id, utils::Rotation rotation) const = 0;
    virtual inline std::string getTag(const std::string& key) const { return tags_.count(key) ? tags_.at(key) : ""; }
    virtual inline void addTag(const std::string& key, const std::string& value) { tags_[key] = value; }

    inline std::string getSGFContent() const { return sgf_content_; }
    inline std::vector<std::pair<Action, ActionInfo>>& getActionPairs() { return action_pairs_; }
    inline const std::vector<std::pair<Action, ActionInfo>>& getActionPairs() const { return action_pairs_; }
    inline void addActionPair(const Action& action, const ActionInfo& action_info = {}) { action_pairs_.emplace_back(action, action_info); }
    inline float getReturn() const { return std::stof(getTag("RE")); }

protected:
    std::string escapeSGFString(const std::string& str) const
    {
        std::string special = "()[]\\";
        std::string escaped;
        escaped.reserve(str.size() + 10);
        for (char c : str) {
            if (special.find(c) != std::string::npos) escaped += '\\';
            escaped += c;
        }
        return escaped;
    }

protected:
    std::string sgf_content_;
    Tags tags_;
    std::vector<std::pair<Action, ActionInfo>> action_pairs_;
};

template <int kNumPlayer = 2>
class BaseBoardAction : public BaseAction {
public:
    BaseBoardAction() : BaseAction() {}
    BaseBoardAction(int action_id, Player player) : BaseAction(action_id, player) {}
    BaseBoardAction(const std::vector<std::string>& action_string_args, int board_size = minizero::config::env_board_size)
    {
        assert(action_string_args.size() == 2);
        assert(action_string_args[0].size() == 1);
        player_ = charToPlayer(action_string_args[0][0]);
        assert(static_cast<int>(player_) > 0 && static_cast<int>(player_) <= kNumPlayer); // assume kPlayer1 == 1, kPlayer2 == 2, ...
        action_id_ = minizero::utils::SGFLoader::boardCoordinateStringToActionID(action_string_args[1], board_size);
    }

    inline Player nextPlayer() const override { return getNextPlayer(getPlayer(), kNumPlayer); }
    inline std::string toConsoleString() const override { return minizero::utils::SGFLoader::actionIDToBoardCoordinateString(getActionID(), minizero::config::env_board_size); }
};

template <class Action>
class BaseBoardEnv : public BaseEnv<Action> {
public:
    BaseBoardEnv(int board_size = minizero::config::env_board_size) : board_size_(board_size) { assert(board_size_ > 0); }
    virtual ~BaseBoardEnv() = default;

    inline int getBoardSize() const { return board_size_; }
    inline int getNumActionFeatureChannels() const override { return 1; }
    inline int getInputChannelHeight() const override { return getBoardSize(); }
    inline int getInputChannelWidth() const override { return getBoardSize(); }
    inline int getHiddenChannelHeight() const override { return getBoardSize(); }
    inline int getHiddenChannelWidth() const override { return getBoardSize(); }
    inline int getDiscreteValueSize() const override { return 1; }

protected:
    int board_size_;
};

template <class Action, class Env>
class BaseBoardEnvLoader : public BaseEnvLoader<Action, Env> {
public:
    BaseBoardEnvLoader() : board_size_(minizero::config::env_board_size) {}
    virtual ~BaseBoardEnvLoader() = default;

    void loadFromEnvironment(const Env& env, const std::vector<std::vector<std::pair<std::string, std::string>>>& action_info_history = {}) override
    {
        BaseEnvLoader<Action, Env>::loadFromEnvironment(env, action_info_history);
        addTag("SZ", std::to_string(env.getBoardSize()));
    }

    inline void addTag(const std::string& key, const std::string& value) override
    {
        BaseEnvLoader<Action, Env>::addTag(key, value);
        if (key == "SZ") { board_size_ = std::stoi(value); }
    }

    inline int getBoardSize() const { return board_size_; }

protected:
    int board_size_;
};

} // namespace minizero::env
