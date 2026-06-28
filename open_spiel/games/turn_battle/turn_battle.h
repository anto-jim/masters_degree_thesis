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

// turn_battle.h — Four-player two-team simultaneous-move battle game.
//
// The game simulates a turn-based fight between two teams of two:
//   Team 0: players 0 (Defender) and 1 (Attacker)
//   Team 1: players 2 (Defender) and 3 (Attacker)
//
// Each round every living player selects one of up to five actions
// (attack enemy defender, attack enemy attacker, self-defense, special move,
// or the mandatory "dead" no-op).  All actions are resolved simultaneously.
// The game is zero-sum: the team with more total HP at termination wins (+1/-1);
// a draw yields 0 for everyone.
//
// Key types defined here:
//   RoleTargets        — pre-computed target indices for a given player's role
//   TurnBattleState    — game state holding HP, special-move flags, history
//   TurnBattleGame     — game descriptor and observer factory

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

inline constexpr int kDefaultNumTurns = 10;
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

// Player-role descriptor: pre-computed team and target indices for one player.
// Avoids repeated role-logic branches inside hot loops.
struct RoleTargets {
  bool is_defender;    // true for players 0 and 2
  int teammate;        // index of the player on the same team
  int enemy_defender;  // index of the opposing team's Defender
  int enemy_attacker;  // index of the opposing team's Attacker
};

// Returns the RoleTargets descriptor for the given player (0–3).
RoleTargets RoleTargetsFor(Player player);

class TurnBattleObserver;

// Concrete state for a TurnBattle episode.
// Tracks per-player HP, one-shot special-move availability, the full action
// history, and the winner set populated at termination.
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
  // Writes the full-recall information-state tensor for `player` into `values`.
  // Layout: [current_turn | HP×4 | special×4 | action_history (flat)].
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  // Writes the current-turn observation tensor for `player` into `values`.
  // Layout: [current_turn | HP×4 | special×4 | private×2] (shape 11; no history).
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  // Returns the set of legal actions for `player`.
  // Dead players have exactly one legal action (kDeadPlayerAction).
  // Living players may attack either living enemy, defend, or use their
  // one-shot special move (if unused and the move is applicable to their role).
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
  // Always asserts false: simultaneous-move games must use DoApplyActions.
  void DoApplyAction(Action action_id) override;
  // Records actions, resolves the current turn, advances the turn counter,
  // and calls UpdateWinners() when the game reaches a terminal state.
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

// Game descriptor for TurnBattle.
// Owns the two shared Observer instances used by all states to serialise their
// tensors (default_observer_ for observations, info_state_observer_ for
// information-state tensors with full history).
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
