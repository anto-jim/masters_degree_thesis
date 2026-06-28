// Copyright 2026 OpenSpiel Authors
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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_TURN_BATTLE_TEAMS_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_TURN_BATTLE_TEAMS_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// turn_battle_teams — sequential 2-player team wrapper for turn_battle
//
// Converts the 4-player simultaneous turn_battle game into a 2-player
// sequential game compatible with AlphaZero-style self-play training.
// Each of the two "team" players (0 and 1) controls both members of its
// team: team 0 acts for underlying players 0 and 1; team 1 acts for
// underlying players 2 and 3.
//
// Within each battle turn the four member actions are gathered one at a
// time in order (P0 → P1 → P2 → P3) via a sub-phase counter.  Only when
// all four have been collected does the state flush them to the underlying
// simultaneous game as a joint action.
//
// Observation layout (flat 1-D tensor):
//   [underlying_obs | sub_phase_norm | member_parity | teammate_action_hint]
// where the three extra scalars provide sub-phase context to the learner.

namespace open_spiel {

inline constexpr int kTurnBattleTeamsNumMembers = 4;
inline constexpr int kTurnBattleTeamsExtraObs = 3;

// Wraps a turn_battle State to present a 2-player sequential interface.
// Internally tracks which of the four underlying members should act next
// (sub_phase_) and buffers their chosen actions in pending_actions_ until a
// full set of four can be committed to the underlying simultaneous state.
class TurnBattleTeamsState : public State {
 public:
  TurnBattleTeamsState(std::shared_ptr<const Game> game,
                       std::unique_ptr<State> underlying);
  TurnBattleTeamsState(const TurnBattleTeamsState& other);

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions(Player player) const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;

 protected:
  // Records action_id for the current sub-phase member and advances the
  // sub-phase counter.  When all four members have chosen, flushes the
  // buffered actions to the underlying state as a simultaneous joint action
  // and resets the counter to 0.
  void DoApplyAction(Action action_id) override;

 private:
  // Returns the index (0–3) of the underlying player whose action is
  // needed next, derived directly from sub_phase_.
  int UnderlyingMember() const;

  // Fills values with the underlying observation for the active member
  // followed by three sub-phase context scalars: normalised phase index,
  // member parity within the team, and the teammate's previous action
  // (normalised) or -1 when not yet available.
  void WriteObservation(Player player, absl::Span<float> values) const;

  std::unique_ptr<State> underlying_;
  int sub_phase_;
  std::array<Action, kTurnBattleTeamsNumMembers> pending_actions_;
};

// 2-player sequential zero-sum game that wraps the 4-player turn_battle game.
// Presents standard OpenSpiel Game metadata (NumPlayers, utility bounds,
// tensor shapes) and delegates action enumeration and state construction
// to the underlying turn_battle game instance.
class TurnBattleTeamsGame : public Game {
 public:
  explicit TurnBattleTeamsGame(const GameParameters& params);

  int NumPlayers() const override { return 2; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override { return 0; }
  int NumDistinctActions() const override {
    return underlying_->NumDistinctActions();
  }
  double MinUtility() const override { return -1.0; }
  double MaxUtility() const override { return 1.0; }
  absl::optional<double> UtilitySum() const override { return 0.0; }
  int MaxGameLength() const override;
  std::vector<int> ObservationTensorShape() const override;
  std::vector<int> InformationStateTensorShape() const override;

  int UnderlyingObservationSize() const { return underlying_obs_size_; }

 private:
  std::shared_ptr<const Game> underlying_;
  int num_turns_;
  int underlying_obs_size_;
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_TURN_BATTLE_TEAMS_H_
