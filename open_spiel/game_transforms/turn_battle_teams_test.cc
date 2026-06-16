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

#include "open_spiel/game_transforms/turn_battle_teams.h"

#include <memory>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace {

void BasicTurnBattleTeamsTests() {
  testing::LoadGameTest("turn_battle_teams(num_turns=3)");
  auto game = LoadGame("turn_battle_teams(num_turns=3)");
  SPIEL_CHECK_EQ(game->NumPlayers(), 2);
  SPIEL_CHECK_EQ(game->GetType().dynamics, GameType::Dynamics::kSequential);
  SPIEL_CHECK_EQ(game->GetType().utility, GameType::Utility::kZeroSum);
  SPIEL_CHECK_EQ(game->NumDistinctActions(), 5);

  auto state = game->NewInitialState();
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->LegalActions().size(), 4);

  for (int step = 0; step < game->MaxGameLength() && !state->IsTerminal(); ++step) {
    const Player player = state->CurrentPlayer();
    SPIEL_CHECK_TRUE(player == 0 || player == 1);
    const auto legal = state->LegalActions();
    SPIEL_CHECK_FALSE(legal.empty());
    state->ApplyAction(legal.front());
  }
  SPIEL_CHECK_TRUE(state->IsTerminal());
  const std::vector<double> returns = state->Returns();
  SPIEL_CHECK_EQ(returns.size(), 2);
  SPIEL_CHECK_FLOAT_EQ(returns[0], -returns[1]);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::BasicTurnBattleTeamsTests();
  std::cerr << "Turn Battle Teams tests passed." << std::endl;
}
