// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/turn_battle/turn_battle.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace turn_battle {
namespace {

const GameType kGameType{
    /*short_name=*/"turn_battle",
    /*long_name=*/"Turn Battle",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{{"num_turns", GameParameter(kDefaultNumTurns)}},
    /*default_loadable=*/true,
    /*provides_factored_observation_string=*/false};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new TurnBattleGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
RegisterSingleTensorObserver single_tensor(kGameType.short_name);

std::vector<double> TeamReturns(int team1_health, int team2_health) {
  std::vector<double> returns(kNumPlayers, 0.0);
  if (team1_health > team2_health) {
    returns[0] = returns[1] = 1.0;
    returns[2] = returns[3] = -1.0;
  } else if (team2_health > team1_health) {
    returns[0] = returns[1] = -1.0;
    returns[2] = returns[3] = 1.0;
  }
  return returns;
}

}  // namespace

RoleTargets RoleTargetsFor(Player player) {
  const bool is_defender = (player == 0 || player == 2);
  if (is_defender) {
    return {true, player == 0 ? 1 : 3, player == 0 ? 2 : 0, player == 0 ? 3 : 1};
  }
  return {false, player == 1 ? 0 : 2, player == 1 ? 2 : 0, player == 1 ? 3 : 1};
}

class TurnBattleObserver : public Observer {
 public:
  explicit TurnBattleObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/true, /*has_tensor=*/true),
        iig_obs_type_(iig_obs_type) {}

  void WriteTensor(const State& observed_state, int player,
                   Allocator* allocator) const override {
    const auto& state = down_cast<const TurnBattleState&>(observed_state);
    const auto& game = down_cast<const TurnBattleGame&>(*state.GetGame());
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, game.NumPlayers());

    if (iig_obs_type_.public_info) {
      auto out = allocator->Get("public_info", {1 + kNumPlayers * 2});
      out.at(0) = state.CurrentTurn();
      for (int p = 0; p < kNumPlayers; ++p) {
        out.at(1 + p) = state.PlayerHealthPoints()[p];
        out.at(1 + kNumPlayers + p) =
            state.PlayerSpecialMoves()[p] ? 1.0f : 0.0f;
      }
    }

    auto private_out = allocator->Get("private_info", {2});
    private_out.at(0) = state.PlayerHealthPoints()[player];
    private_out.at(1) = state.PlayerSpecialMoves()[player] ? 1.0f : 0.0f;

    if (iig_obs_type_.perfect_recall) {
      const int max_history = game.MaxGameLength() * kNumPlayers;
      auto history = allocator->Get("action_history", {max_history});
      int idx = 0;
      for (const auto& turn_actions : state.ActionsHistory()) {
        for (Action action : turn_actions) {
          if (idx < max_history) history.at(idx++) = action;
        }
      }
    }
  }

  std::string StringFrom(const State& observed_state,
                         int player) const override {
    const auto& state = down_cast<const TurnBattleState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, kNumPlayers);
    return iig_obs_type_.perfect_recall ? state.ToString()
                                       : state.ObservationString(player);
  }

 private:
  IIGObservationType iig_obs_type_;
};

TurnBattleState::TurnBattleState(std::shared_ptr<const Game> game, int num_turns)
    : SimMoveState(game),
      num_turns_(num_turns),
      current_turn_(0),
      player_health_points_(kNumPlayers, kMaxHealthPoints),
      player_special_moves_(kNumPlayers, true) {}

Player TurnBattleState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : kSimultaneousPlayerId;
}

std::vector<Action> TurnBattleState::LegalActions(Player player) const {
  if (IsTerminal()) return {};
  if (player == kSimultaneousPlayerId) return LegalFlatJointActions();
  if (player_health_points_[player] <= 0) return {kDeadPlayerAction};

  const RoleTargets role = RoleTargetsFor(player);
  std::vector<Action> legal;
  if (player_health_points_[role.enemy_defender] > 0) {
    legal.push_back(kTargetEnemyDefender);
  }
  if (player_health_points_[role.enemy_attacker] > 0) {
    legal.push_back(kTargetEnemyAttacker);
  }
  legal.push_back(kSelfDefense);
  if (player_special_moves_[player]) {
    if (role.is_defender) {
      legal.push_back(kSpecialMove);
    } else if (player_health_points_[role.enemy_defender] > 0 ||
               player_health_points_[role.enemy_attacker] > 0) {
      // Attacker special hits living enemies only; skip if both are dead.
      legal.push_back(kSpecialMove);
    }
  }
  return legal;
}

void TurnBattleState::DoApplyAction(Action) {
  SPIEL_CHECK_TRUE(false);
}

void TurnBattleState::ApplyDamage(int target, std::vector<int>* health) {
  (*health)[target] = std::max(0, (*health)[target] - 1);
}

int TurnBattleState::TeamHealth(int team) const {
  return player_health_points_[team * kTeamSize] +
         player_health_points_[team * kTeamSize + 1];
}

void TurnBattleState::ResolveTurn(const std::vector<Action>& actions) {
  std::vector<int> new_health = player_health_points_;

  for (Player player = 0; player < kNumPlayers; ++player) {
    if (player_health_points_[player] <= 0) continue;

    const Action action = actions[player];
    const RoleTargets role = RoleTargetsFor(player);

    switch (action) {
      case kTargetEnemyDefender:
        if (player_health_points_[role.enemy_defender] > 0) {
          ApplyDamage(role.enemy_defender, &new_health);
        }
        break;
      case kTargetEnemyAttacker:
        if (player_health_points_[role.enemy_attacker] > 0) {
          ApplyDamage(role.enemy_attacker, &new_health);
        }
        break;
      case kSpecialMove:
        if (!player_special_moves_[player]) break;
        if (role.is_defender) break;
        if (player_health_points_[role.enemy_defender] > 0) {
          ApplyDamage(role.enemy_defender, &new_health);
        }
        if (player_health_points_[role.enemy_attacker] > 0) {
          ApplyDamage(role.enemy_attacker, &new_health);
        }
        player_special_moves_[player] = false;
        break;
      default:
        break;
    }
  }

  for (Player player = 0; player < kNumPlayers; ++player) {
    if (player_health_points_[player] <= 0) continue;
    const Action action = actions[player];
    const RoleTargets role = RoleTargetsFor(player);
    if (action == kSelfDefense) {
      new_health[player] = player_health_points_[player];
    } else if (action == kSpecialMove && role.is_defender &&
               player_special_moves_[player]) {
      new_health[player] = player_health_points_[player];
      new_health[role.teammate] = player_health_points_[role.teammate];
      player_special_moves_[player] = false;
    }
  }

  player_health_points_ = std::move(new_health);
}

void TurnBattleState::DoApplyActions(const std::vector<Action>& actions) {
  SPIEL_CHECK_EQ(actions.size(), kNumPlayers);
  actions_history_.push_back(actions);
  ResolveTurn(actions);
  ++current_turn_;
  if (IsTerminal()) UpdateWinners();
}

void TurnBattleState::UpdateWinners() {
  winners_.clear();
  if (!IsTerminal()) return;
  const int team1 = TeamHealth(0);
  const int team2 = TeamHealth(1);
  if (team1 > team2) {
    winners_.insert(0);
    winners_.insert(1);
  } else if (team2 > team1) {
    winners_.insert(2);
    winners_.insert(3);
  }
}

std::string TurnBattleState::ActionToString(Player player,
                                            Action action_id) const {
  if (player == kSimultaneousPlayerId) {
    return FlatJointActionToString(action_id);
  }
  if (player_health_points_[player] <= 0 && action_id == kDeadPlayerAction) {
    return "Dead";
  }
  const bool is_defender = RoleTargetsFor(player).is_defender;
  switch (action_id) {
    case kTargetEnemyDefender:
      return "Attack Enemy Defender";
    case kTargetEnemyAttacker:
      return "Attack Enemy Attacker";
    case kSelfDefense:
      return "Self-Defense";
    case kSpecialMove:
      return is_defender ? "Special: Defend Team" : "Special: Attack Both";
    case kDeadPlayerAction:
      return "Dead";
    default:
      return "Unknown";
  }
}

std::string TurnBattleState::ToString() const {
  std::string result =
      absl::StrCat("Turn: ", current_turn_, "/", num_turns_, "\n");
  const char* labels[] = {"P0 (Defender)", "P1 (Attacker)", "P2 (Defender)",
                          "P3 (Attacker)"};
  for (int p = 0; p < kNumPlayers; ++p) {
    const int team = p < kTeamSize ? 1 : 2;
    if (p == 0 || p == kTeamSize) {
      absl::StrAppend(&result, "Team ", team, ":\n");
    }
    absl::StrAppend(&result, "  ", labels[p], ": HP=",
                    player_health_points_[p], " Special=",
                    player_special_moves_[p] ? "Yes" : "No", "\n");
  }
  if (IsTerminal()) {
    absl::StrAppend(&result, "Game Over. ");
    if (winners_.empty()) {
      result += "Draw\n";
    } else {
      result += "Winners:";
      for (int winner : winners_) absl::StrAppend(&result, " ", winner);
      result += "\n";
    }
  }
  return result;
}

bool TurnBattleState::IsTerminal() const {
  if (current_turn_ >= num_turns_) return true;
  return TeamHealth(0) == 0 || TeamHealth(1) == 0;
}

std::vector<double> TurnBattleState::Returns() const {
  if (!IsTerminal()) return std::vector<double>(kNumPlayers, 0.0);
  return TeamReturns(TeamHealth(0), TeamHealth(1));
}

std::string TurnBattleState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  return ToString();
}

std::string TurnBattleState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  return ToString();
}

void TurnBattleState::InformationStateTensor(Player player,
                                             absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const auto& game = down_cast<const TurnBattleGame&>(*game_);
  game.info_state_observer_->WriteTensor(*this, player, &allocator);
}

void TurnBattleState::ObservationTensor(Player player,
                                        absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const auto& game = down_cast<const TurnBattleGame&>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

std::unique_ptr<State> TurnBattleState::Clone() const {
  return std::unique_ptr<State>(new TurnBattleState(*this));
}

TurnBattleGame::TurnBattleGame(const GameParameters& params)
    : SimMoveGame(kGameType, params),
      num_turns_(ParameterValue<int>("num_turns", kDefaultNumTurns)) {
  default_observer_ = std::make_shared<TurnBattleObserver>(
      IIGObservationType{/*.public_info=*/true, /*.perfect_recall=*/false,
                         /*.private_info=*/PrivateInfoType::kNone});
  info_state_observer_ = std::make_shared<TurnBattleObserver>(
      IIGObservationType{/*.public_info=*/true, /*.perfect_recall=*/true,
                         /*.private_info=*/PrivateInfoType::kSinglePlayer});
}

std::unique_ptr<State> TurnBattleGame::NewInitialState() const {
  return std::unique_ptr<State>(
      new TurnBattleState(shared_from_this(), num_turns_));
}

int TurnBattleGame::MaxChanceOutcomes() const { return 0; }

double TurnBattleGame::MinUtility() const { return -1.0; }
double TurnBattleGame::MaxUtility() const { return 1.0; }
absl::optional<double> TurnBattleGame::UtilitySum() const { return 0.0; }

std::vector<int> TurnBattleGame::InformationStateTensorShape() const {
  return {1 + kNumPlayers * 2 + 2 + num_turns_ * kNumPlayers};
}

std::vector<int> TurnBattleGame::ObservationTensorShape() const {
  return {1 + kNumPlayers * 2 + 2};
}

std::shared_ptr<Observer> TurnBattleGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  if (iig_obs_type.has_value()) {
    return std::make_shared<TurnBattleObserver>(*iig_obs_type);
  }
  return default_observer_;
}

}  // namespace turn_battle
}  // namespace open_spiel
