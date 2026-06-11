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

#ifndef OPEN_SPIEL_GAMES_TURN_BATTLE_H_
#define OPEN_SPIEL_GAMES_TURN_BATTLE_H_

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace turn_battle {

inline constexpr int kDefaultNumTurns = 15;
inline constexpr int kNumPlayers = 4;
inline constexpr int kMaxPossibleMoves = 5;
inline constexpr int kMaxHealthPoints = 3;

// Teams: {0,1} vs {2,3}. Defenders: 0,2. Attackers: 1,3.
inline constexpr int kTeamSize = 2;
inline constexpr Action kTargetEnemyDefender = 0;
inline constexpr Action kTargetEnemyAttacker = 1;
inline constexpr Action kSelfDefense = 2;
inline constexpr Action kSpecialMove = 3;
inline constexpr Action kDeadPlayerAction = 4;

struct RoleTargets {
  bool is_defender;
  int teammate;
  int enemy_defender;
  int enemy_attacker;
};

RoleTargets RoleTargetsFor(Player player);

class TurnBattleObserver;

class TurnBattleState : public SimMoveState {
 public:
  explicit TurnBattleState(std::shared_ptr<const Game> game, int num_turns);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions(Player player) const override;

  int CurrentTurn() const { return current_turn_; }
  const std::vector<int>& PlayerHealthPoints() const {
    return player_health_points_;
  }
  const std::vector<bool>& PlayerSpecialMoves() const {
    return player_special_moves_;
  }
  const std::vector<std::vector<Action>>& ActionsHistory() const {
    return actions_history_;
  }

 protected:
  void DoApplyAction(Action action_id) override;
  void DoApplyActions(const std::vector<Action>& actions) override;

 private:
  friend class TurnBattleObserver;

  int TeamHealth(int team) const;
  void ApplyDamage(int target, std::vector<int>* health);
  void ResolveTurn(const std::vector<Action>& actions);
  void UpdateWinners();

  int num_turns_;
  int current_turn_;
  std::set<int> winners_;
  std::vector<int> player_health_points_;
  std::vector<bool> player_special_moves_;
  std::vector<std::vector<Action>> actions_history_;
};

class TurnBattleGame : public SimMoveGame {
 public:
  explicit TurnBattleGame(const GameParameters& params);

  int NumDistinctActions() const override { return kMaxPossibleMoves; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override;
  double MaxUtility() const override;
  absl::optional<double> UtilitySum() const override;
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override { return num_turns_; }
  int MaxChanceNodesInHistory() const override { return 0; }
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const override;

  int NumTurns() const { return num_turns_; }

  std::shared_ptr<Observer> default_observer_;
  std::shared_ptr<Observer> info_state_observer_;

 private:
  int num_turns_;
};

}  // namespace turn_battle
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_TURN_BATTLE_H_
