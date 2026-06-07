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
#include "open_spiel/spiel_utils.h"
#include "open_spiel/observer.h"

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
    /*parameter_specification=*/
    {
        {"num_turns", GameParameter(kDefaultNumTurns)},                       
    },
    /*default_loadable=*/true,
    /*provides_factored_observation_string=*/false};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new TurnBattleGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

}  // namespace

// ============================================================================
// Observer Implementation
// ============================================================================

class TurnBattleObserver : public Observer {
 public:
  TurnBattleObserver(IIGObservationType iig_obs_type) : Observer(/*has_string=*/true, /*has_tensor=*/true), iig_obs_type_(iig_obs_type) {}
 
  void WriteTensor(const State& observed_state, int player,
                   Allocator* allocator) const override {
    const TurnBattleState& state =
        open_spiel::down_cast<const TurnBattleState&>(observed_state);
    const TurnBattleGame& game =
        open_spiel::down_cast<const TurnBattleGame&>(*state.GetGame());
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, game.NumPlayers());

    if (iig_obs_type_.public_info) {
      // Public information: current turn, health points, special moves available
      auto out = allocator->Get("public_info", {1 + kNumPlayers * 2});
      out.at(0) = state.current_turn_;
      
      for (int p = 0; p < kNumPlayers; ++p) {
        out.at(1 + p) = state.player_health_points_[p];
        out.at(1 + kNumPlayers + p) = state.player_special_moves_[p] ? 1.0 : 0.0;
      }
    }
    
    // Private information: player's own health and special move
    auto out = allocator->Get("private_info", {2});
    out.at(0) = state.player_health_points_[player];
    out.at(1) = state.player_special_moves_[player] ? 1.0 : 0.0;

    if (iig_obs_type_.perfect_recall) {
      // Action history
      int max_history_size = game.MaxGameLength() * kNumPlayers;
      auto out = allocator->Get("action_history", {max_history_size});
      
      int idx = 0;
      for (const auto& turn_actions : state.actions_history_) {
        for (Action action : turn_actions) {
          if (idx < max_history_size) {
            out.at(idx++) = action;
          }
        }
      }
    }
  }

  std::string StringFrom(const State& observed_state,
                         int player) const override {
    const TurnBattleState& state =
        open_spiel::down_cast<const TurnBattleState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, kNumPlayers);
    std::string result;

    if (iig_obs_type_.perfect_recall) {
      return state.ToString();
    } else {
      return state.ObservationString(player);
    }
  }

 private:
  IIGObservationType iig_obs_type_;
};

// ============================================================================
// TurnBattleState Implementation
// ============================================================================

TurnBattleState::TurnBattleState(std::shared_ptr<const Game> game, int num_turns)
    : SimMoveState(game),
      num_turns_(num_turns),
      current_player_(0),
      current_turn_(0),
      point_card_(0),
      player_health_points_(kNumPlayers, kMaxHealthPoints),
      player_special_moves_(kNumPlayers, true) {
  // Initialize action history
  actions_history_.clear();
}

Player TurnBattleState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : kSimultaneousPlayerId;
}

std::vector<Action> TurnBattleState::LegalActions(Player player) const {
  if (IsTerminal()) return {};
  if (player == kSimultaneousPlayerId) return LegalFlatJointActions();

  std::vector<Action> legal_actions;

    // If player is dead, only action 4 (dead player's action) is legal
  if (player_health_points_[player] <= 0) {
    legal_actions.push_back(4);
    return legal_actions;
  }
  
  // Actions 0-1: Target opponents
  legal_actions.push_back(0);  // Target defender
  legal_actions.push_back(1);  // Target attacker
  
  // Action 2: Self-defense
  legal_actions.push_back(2);
  
  // Action 3: Special move (only if available)
  if (player_special_moves_[player]) {
    legal_actions.push_back(3);
  }
  
  return legal_actions;
}

void TurnBattleState::DoApplyAction(Action action_id) {
  // Not used in simultaneous move games
  SPIEL_CHECK_TRUE(false);
}

void TurnBattleState::DoApplyActions(const std::vector<Action>& actions) {
  SPIEL_CHECK_EQ(actions.size(), kNumPlayers);
  
  // Store actions in history
  actions_history_.push_back(actions);
  
  // Temporary health points for this turn
  std::vector<int> new_health_points = player_health_points_;
  
  // Process each player's action
  for (Player player = 0; player < kNumPlayers; ++player) {
    Action action = actions[player];
    
    // Skip if player is dead
    if (player_health_points_[player] <= 0) {
      continue;
    }
    
    bool is_defender = (player == 0 || player == 2);
    int teammate;
    int opponentDefender;
    int opponentAttacker;

    if (is_defender) {
      teammate = player == 0 ? 1 : 3;
      opponentDefender = player == 0 ? 2 : 0;
      opponentAttacker = player == 0 ? 3 : 1;
    } else {
      teammate = player == 1 ? 0 : 2;
      opponentDefender = player == 1 ? 2 : 0;
      opponentAttacker = player == 1 ? 3 : 1;
    }
    
    switch (action) {
      case 0: {  // Target opponent defender
        int target = opponentDefender;
        if (player_health_points_[target] > 0) {
          ApplyDamage(target, &new_health_points);
        }
        break;
      }
      case 1: {  // Target opponent attacker
        int target = opponentAttacker;
        if (player_health_points_[target] > 0) {
          ApplyDamage(target, &new_health_points);
        }
        break;
      }
      case 2: {  // Self-defense (reduce incoming damage this turn)
        // Defense is handled by checking if player chose action 2
        break;
      }
      case 3: {  // Special move
        if (player_special_moves_[player]) {
          if (is_defender) {
            // Defender: defend self and ally
            // This is handled by marking that special defense was used
          } else {
            // Attacker: attack both opponents
            if (player_health_points_[opponentDefender] > 0) {
              ApplyDamage(opponentDefender, &new_health_points);
            }
            if (player_health_points_[opponentAttacker] > 0) {
              ApplyDamage(opponentAttacker, &new_health_points);
            }
          }
          player_special_moves_[player] = false;  // Special move used
        }
        break;
      }
      case 4: {  // Dead player action (do nothing)
        break;
      }
    }
  }
  
  // Apply defense modifiers
  for (Player player = 0; player < kNumPlayers; ++player) {
    if (player_health_points_[player] <= 0) continue;
    
    Action action = actions[player];
    bool is_defender = (player == 0 || player == 2);
    int teammate = is_defender ? (player == 0 ? 1 : 3) : (player == 1 ? 0 : 2);
    
    // Self-defense: nullify damage
    if (action == 2) {
      new_health_points[player] = player_health_points_[player];
    }
    
    // Special defense (defender protects self and ally)
    if (action == 3 && is_defender) {
        new_health_points[player] = player_health_points_[player];
        new_health_points[teammate] = player_health_points_[teammate];
    }
  }
  
  // Update health points
  player_health_points_ = new_health_points;
  
  // Increment turn counter
  current_turn_++;

  if (IsTerminal()) {
    UpdateWinners();
  }
}

void TurnBattleState::UpdateWinners() {
  winners_.clear();
  if (!IsTerminal()) return;

  int team1_health = player_health_points_[0] + player_health_points_[1];
  int team2_health = player_health_points_[2] + player_health_points_[3];

  if (team1_health > team2_health) {
    winners_.insert(0);
    winners_.insert(1);
  } else if (team2_health > team1_health) {
    winners_.insert(2);
    winners_.insert(3);
  }
}

void TurnBattleState::ApplyDamage(const int target, 
                                   std::vector<int>* player_health_points) {
  (*player_health_points)[target] = std::max(0, (*player_health_points)[target] - 1);
}

std::string TurnBattleState::ActionToString(Player player, Action action_id) const {

  if (player == kSimultaneousPlayerId)
    return FlatJointActionToString(action_id);

  if (player_health_points_[player] <= 0 && action_id == 4) {
    return "Dead";
  }
  
  bool is_defender = (player == 0 || player == 2);
  
  switch (action_id) {
    case 0:
      return "Attack Enemy Defender";
    case 1:
      return "Attack Enemy Attacker";
    case 2:
      return "Self-Defense";
    case 3:
      return is_defender ? "Special: Defend Team" : "Special: Attack Both";
    case 4:
      return "Dead";
    default:
      return "Unknown";
  }
}

std::string TurnBattleState::ToString() const {
  std::string result = absl::StrCat("Turn: ", current_turn_, "/", num_turns_, "\n");
  result += "Team 1 (Players 0,1):\n";
  result += absl::StrCat("  P0 (Defender): HP=", player_health_points_[0], 
                         " Special=", player_special_moves_[0] ? "Yes" : "No", "\n");
  result += absl::StrCat("  P1 (Attacker): HP=", player_health_points_[1], 
                         " Special=", player_special_moves_[1] ? "Yes" : "No", "\n");
  result += "Team 2 (Players 2,3):\n";
  result += absl::StrCat("  P2 (Defender): HP=", player_health_points_[2], 
                         " Special=", player_special_moves_[2] ? "Yes" : "No", "\n");
  result += absl::StrCat("  P3 (Attacker): HP=", player_health_points_[3], 
                         " Special=", player_special_moves_[3] ? "Yes" : "No", "\n");
  
  if (IsTerminal()) {
    result += "Game Over. ";
    if (!winners_.empty()) {
      result += "Winners: ";
      for (int winner : winners_) {
        result += std::to_string(winner) + " ";
      }
    } else {
      result += "Draw";
    }
    result += "\n";
  }
  
  return result;
}

bool TurnBattleState::IsTerminal() const {
  // Game ends if max turns reached
  if (current_turn_ >= num_turns_) {
    return true;
  }
  
  // Game ends if one team is completely eliminated
  bool team1_alive = (player_health_points_[0] > 0 || player_health_points_[1] > 0);
  bool team2_alive = (player_health_points_[2] > 0 || player_health_points_[3] > 0);
  
  return !team1_alive || !team2_alive;
}

std::vector<double> TurnBattleState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(kNumPlayers, 0.0);
  }
  
  // Calculate team health
  int team1_health = player_health_points_[0] + player_health_points_[1];
  int team2_health = player_health_points_[2] + player_health_points_[3];
  
  std::vector<double> returns(kNumPlayers);
  
  if (team1_health > team2_health) {
    // Team 1 wins
    returns[0] = 1.0;
    returns[1] = 1.0;
    returns[2] = -1.0;
    returns[3] = -1.0;
  } else if (team2_health > team1_health) {
    // Team 2 wins
    returns[0] = -1.0;
    returns[1] = -1.0;
    returns[2] = 1.0;
    returns[3] = 1.0;
  } else {
    // Draw
    returns[0] = 0.0;
    returns[1] = 0.0;
    returns[2] = 0.0;
    returns[3] = 0.0;
  }
  
  return returns;
}


std::string TurnBattleState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  
  // In perfect information game, information state equals observation
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
  const TurnBattleGame& game =
      open_spiel::down_cast<const TurnBattleGame&>(*game_);
  game.info_state_observer_->WriteTensor(*this, player, &allocator);
}

void TurnBattleState::ObservationTensor(Player player,
                                        absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const TurnBattleGame& game =
      open_spiel::down_cast<const TurnBattleGame&>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

std::unique_ptr<State> TurnBattleState::Clone() const {
  return std::unique_ptr<State>(new TurnBattleState(*this));
}

void TurnBattleState::NextPlayer(int* count, Player* player) const {
  *count = (*count + 1) % kNumPlayers;
  *player = (*player + 1) % kNumPlayers;
}

// ============================================================================
// TurnBattleGame Implementation
// ============================================================================

TurnBattleGame::TurnBattleGame(const GameParameters& params)
    : SimMoveGame(kGameType, params),
      num_turns_(ParameterValue<int>("num_turns", kDefaultNumTurns)) {
  default_observer_ = std::make_shared<TurnBattleObserver>(
      IIGObservationType{/*.public_info=*/true, /*.perfect_recall=*/false,
                         /*.private_info=*/PrivateInfoType::kNone});
  info_state_observer_ = std::make_shared<TurnBattleObserver>(
      IIGObservationType{/*.public_info=*/true, /*.perfect_recall=*/true,
                         /*.private_info=*/PrivateInfoType::kSinglePlayer});
  public_observer_ = std::make_shared<TurnBattleObserver>(
      IIGObservationType{/*.public_info=*/true, /*.perfect_recall=*/false,
                         /*.private_info=*/PrivateInfoType::kNone});
  private_observer_ = std::make_shared<TurnBattleObserver>(
      IIGObservationType{/*.public_info=*/false, /*.perfect_recall=*/false,
                         /*.private_info=*/PrivateInfoType::kSinglePlayer});
}

std::unique_ptr<State> TurnBattleGame::NewInitialState() const {
  return std::unique_ptr<State>(new TurnBattleState(shared_from_this(), num_turns_));
}

int TurnBattleGame::MaxChanceOutcomes() const {
  return 0;  // Deterministic game
}

double TurnBattleGame::MinUtility() const {
  return -1.0;
}

double TurnBattleGame::MaxUtility() const {
  return 1.0;
}

absl::optional<double> TurnBattleGame::UtilitySum() const {
  return 0.0;  // Zero-sum game
}

std::vector<int> TurnBattleGame::InformationStateTensorShape() const {
  return {1 + kNumPlayers * 2 + 2 + num_turns_ * kNumPlayers};
}

std::vector<int> TurnBattleGame::ObservationTensorShape() const {
  return {1 + kNumPlayers * 2 + 2};  // public + private = 11
}

std::shared_ptr<Observer> TurnBattleGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  if (iig_obs_type.has_value()) {
    return std::make_shared<TurnBattleObserver>(*iig_obs_type);
  } else {
    return default_observer_;
  }
}


}  // namespace turn_battle
}  // namespace open_spiel
