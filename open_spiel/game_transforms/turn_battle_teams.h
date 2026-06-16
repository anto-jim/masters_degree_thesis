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

// A 2-player sequential wrapper around turn_battle where each player controls
// both members of their team. Team 0 acts for players 0 and 1; team 1 for 2
// and 3. Within each battle turn the order is P0, P1, P2, P3.

namespace open_spiel {

inline constexpr int kTurnBattleTeamsNumMembers = 4;
inline constexpr int kTurnBattleTeamsExtraObs = 3;

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
  void DoApplyAction(Action action_id) override;

 private:
  int UnderlyingMember() const;
  void WriteObservation(Player player, absl::Span<float> values) const;

  std::unique_ptr<State> underlying_;
  int sub_phase_;
  std::array<Action, kTurnBattleTeamsNumMembers> pending_actions_;
};

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
