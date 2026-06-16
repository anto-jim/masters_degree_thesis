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

#include <algorithm>
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

bool ContainsAction(const std::vector<Action>& actions, Action action) {
  return std::find(actions.begin(), actions.end(), action) != actions.end();
}

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

void CannotAttackDeadTargets() {
  GameParameters params;
  params["num_turns"] = GameParameter(20);
  std::shared_ptr<const Game> game = LoadGame("turn_battle", params);
  std::unique_ptr<State> state = game->NewInitialState();
  auto* tb = down_cast<TurnBattleState*>(state.get());

  // P0 and P1 focus P2; P2 counters P1 once per turn before dying.
  for (int turn = 0; turn < kMaxHealthPoints; ++turn) {
    SPIEL_CHECK_FALSE(tb->IsTerminal());
    tb->ApplyActions({kTargetEnemyDefender, kTargetEnemyDefender,
                      kTargetEnemyAttacker, kSelfDefense});
  }

  SPIEL_CHECK_EQ(tb->PlayerHealthPoints()[2], 0);
  SPIEL_CHECK_GT(tb->PlayerHealthPoints()[1], 0);
  SPIEL_CHECK_GT(tb->PlayerHealthPoints()[3], 0);

  const std::vector<Action> p1_legal = tb->LegalActions(1);
  SPIEL_CHECK_FALSE(ContainsAction(p1_legal, kTargetEnemyDefender));
  SPIEL_CHECK_TRUE(ContainsAction(p1_legal, kTargetEnemyAttacker));
  SPIEL_CHECK_TRUE(ContainsAction(p1_legal, kSelfDefense));

  std::cout << "CannotAttackDeadTargets: P1 cannot target dead P2" << std::endl;
}

}  // namespace
}  // namespace turn_battle
}  // namespace open_spiel

int main(int argc, char **argv) {
  std::cout << "=== turn_battle_test ===" << std::endl;
  open_spiel::turn_battle::BasicTurnBattleTests();
  open_spiel::turn_battle::CannotAttackDeadTargets();
  open_spiel::turn_battle::LegalActionsValidAtEveryState();
  std::cout << "=== turn_battle_test passed ===" << std::endl;
}
