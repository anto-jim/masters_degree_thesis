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

// Implementation of the TurnBattleTeams game transform.
//
// See turn_battle_teams.h for the design overview.  This file contains:
//   - kGameType descriptor and factory registration
//   - TurnBattleTeamsState method bodies (sub-phase bookkeeping, observation)
//   - TurnBattleTeamsGame constructor and tensor shape queries

#include "open_spiel/game_transforms/turn_battle_teams.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

const GameType kGameType{
    /*short_name=*/"turn_battle_teams",
    /*long_name=*/"Turn Battle (2-Team AlphaZero View)",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{{"num_turns", GameParameter(5)}},
    /*default_loadable=*/true,
    /*provides_factored_observation_string=*/false,
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new TurnBattleTeamsGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
RegisterSingleTensorObserver single_tensor(kGameType.short_name);

}  // namespace

TurnBattleTeamsState::TurnBattleTeamsState(
    std::shared_ptr<const Game> game, std::unique_ptr<State> underlying)
    : State(game), underlying_(std::move(underlying)), sub_phase_(0) {
  pending_actions_.fill(0);
}

TurnBattleTeamsState::TurnBattleTeamsState(const TurnBattleTeamsState& other)
    : State(other),
      underlying_(other.underlying_->Clone()),
      sub_phase_(other.sub_phase_),
      pending_actions_(other.pending_actions_) {}

std::unique_ptr<State> TurnBattleTeamsState::Clone() const {
  return std::unique_ptr<State>(new TurnBattleTeamsState(*this));
}

int TurnBattleTeamsState::UnderlyingMember() const { return sub_phase_; }

Player TurnBattleTeamsState::CurrentPlayer() const {
  if (underlying_->IsTerminal()) return kTerminalPlayerId;
  return sub_phase_ < 2 ? 0 : 1;
}

std::vector<Action> TurnBattleTeamsState::LegalActions(Player player) const {
  if (player != CurrentPlayer()) return {};
  return underlying_->LegalActions(UnderlyingMember());
}

std::vector<Action> TurnBattleTeamsState::LegalActions() const {
  return LegalActions(CurrentPlayer());
}

// Buffers action_id for the current underlying member (sub_phase_) and
// increments the sub-phase counter.  Once all kTurnBattleTeamsNumMembers
// actions have been collected, they are forwarded to the underlying
// simultaneous game as a single joint action and the sub-phase resets to 0.
void TurnBattleTeamsState::DoApplyAction(Action action_id) {
  SPIEL_CHECK_FALSE(underlying_->IsTerminal());
  const int member = UnderlyingMember();
  pending_actions_[member] = action_id;
  ++sub_phase_;
  if (sub_phase_ == kTurnBattleTeamsNumMembers) {
    underlying_->ApplyActions(
        std::vector<Action>(pending_actions_.begin(), pending_actions_.end()));
    sub_phase_ = 0;
    pending_actions_.fill(0);
  }
}

std::string TurnBattleTeamsState::ActionToString(Player player,
                                                 Action action_id) const {
  const int member = UnderlyingMember();
  return absl::StrCat("Team ", player, " member ", member, ": ",
                      underlying_->ActionToString(member, action_id));
}

std::string TurnBattleTeamsState::ToString() const {
  return absl::StrCat("Sub-phase: ", sub_phase_, "/", kTurnBattleTeamsNumMembers,
                      "\n", underlying_->ToString());
}

bool TurnBattleTeamsState::IsTerminal() const {
  return underlying_->IsTerminal();
}

std::vector<double> TurnBattleTeamsState::Returns() const {
  if (!underlying_->IsTerminal()) return {0.0, 0.0};
  const std::vector<double> underlying_returns = underlying_->Returns();
  SPIEL_CHECK_EQ(underlying_returns.size(), kTurnBattleTeamsNumMembers);
  const double team0 = underlying_returns[0];
  return {team0, -team0};
}

// Writes the observation tensor for the requesting team player.
// Layout: [underlying_obs (underlying_obs_size floats) |
//          sub_phase normalised to [0, 1] |
//          sub_phase parity (0 = first member of team, 1 = second) |
//          normalised teammate action, or -1 if not yet available]
void TurnBattleTeamsState::WriteObservation(Player player,
                                            absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, 2);
  const auto& game = down_cast<const TurnBattleTeamsGame&>(*game_);
  const int underlying_obs_size = game.UnderlyingObservationSize();
  SPIEL_CHECK_EQ(values.size(), underlying_obs_size + kTurnBattleTeamsExtraObs);

  const int member = UnderlyingMember();
  underlying_->ObservationTensor(
      member, absl::MakeSpan(values.data(), underlying_obs_size));

  const int extra_base = underlying_obs_size;
  values[extra_base] = static_cast<float>(sub_phase_) / 3.0f;
  values[extra_base + 1] = static_cast<float>(sub_phase_ % 2);
  if (sub_phase_ % 2 == 1) {
    values[extra_base + 2] =
        static_cast<float>(pending_actions_[sub_phase_ - 1]) / 4.0f;
  } else {
    values[extra_base + 2] = -1.0f;
  }
}

std::string TurnBattleTeamsState::InformationStateString(Player player) const {
  return ObservationString(player);
}

void TurnBattleTeamsState::InformationStateTensor(
    Player player, absl::Span<float> values) const {
  WriteObservation(player, values);
}

std::string TurnBattleTeamsState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, 2);
  return absl::StrCat("Team ", player, " acting for member ", UnderlyingMember(),
                      "\n", underlying_->ObservationString(UnderlyingMember()));
}

void TurnBattleTeamsState::ObservationTensor(Player player,
                                             absl::Span<float> values) const {
  WriteObservation(player, values);
}

TurnBattleTeamsGame::TurnBattleTeamsGame(const GameParameters& params)
    : Game(kGameType, params) {
  num_turns_ = ParameterValue<int>("num_turns");
  GameParameters underlying_params;
  underlying_params["num_turns"] = GameParameter(num_turns_);
  underlying_ = LoadGame("turn_battle", underlying_params);
  SPIEL_CHECK_EQ(underlying_->NumPlayers(), kTurnBattleTeamsNumMembers);
  SPIEL_CHECK_EQ(underlying_->GetType().dynamics,
                 GameType::Dynamics::kSimultaneous);
  underlying_obs_size_ = underlying_->ObservationTensorSize();
}

std::unique_ptr<State> TurnBattleTeamsGame::NewInitialState() const {
  return std::unique_ptr<State>(new TurnBattleTeamsState(
      shared_from_this(), underlying_->NewInitialState()));
}

int TurnBattleTeamsGame::MaxGameLength() const {
  return num_turns_ * kTurnBattleTeamsNumMembers;
}

// Returns the flat shape of the observation tensor: the underlying per-member
// observation size plus kTurnBattleTeamsExtraObs sub-phase context scalars.
std::vector<int> TurnBattleTeamsGame::ObservationTensorShape() const {
  return {underlying_obs_size_ + kTurnBattleTeamsExtraObs};
}

// Information-state tensor has the same layout as the observation tensor
// because this game provides perfect information.
std::vector<int> TurnBattleTeamsGame::InformationStateTensorShape() const {
  return ObservationTensorShape();
}

}  // namespace open_spiel
