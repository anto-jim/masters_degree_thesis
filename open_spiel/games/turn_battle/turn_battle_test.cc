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

#include <iostream>

#include "open_spiel/games/turn_battle/turn_battle.h"

#include "open_spiel/game_parameters.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace turn_battle {
namespace {

namespace testing = open_spiel::testing;

void BasicTurnBattleTests() {
  std::shared_ptr<const Game> game = LoadGame("turn_battle");
  SPIEL_CHECK_EQ(game->GetType().short_name, "turn_battle");
  SPIEL_CHECK_EQ(game->NumPlayers(), kNumPlayers);
  std::cout << "BasicTurnBattleTests: game=" << game->GetType().short_name
            << " players=" << game->NumPlayers() << std::endl;
  testing::LoadGameTest("turn_battle");
  testing::RandomSimTest(*game, 100);
}

void LegalActionsValidAtEveryState() {
  std::shared_ptr<const Game> game = LoadGameAsTurnBased("turn_battle");
  SPIEL_CHECK_EQ(game->GetType().short_name, "turn_based_simultaneous_game");
  std::cout << "LegalActionsValidAtEveryState: wrapped turn_battle ("
            << game->NumPlayers() << " players)" << std::endl;
  testing::RandomSimTest(*game, /*num_sims=*/10);
}

}  // namespace
}  // namespace turn_battle
}  // namespace open_spiel

int main(int argc, char **argv) {
  std::cout << "=== turn_battle_test ===" << std::endl;
  open_spiel::turn_battle::BasicTurnBattleTests();
  open_spiel::turn_battle::LegalActionsValidAtEveryState();
  std::cout << "=== turn_battle_test passed ===" << std::endl;
}
