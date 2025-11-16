/**
 * Play 2048 manually to test game logic
 */

import * as game from './game'

console.log('🎮 2048 Game - Manual Test\n')

let state = game.init()

console.log('Initial state:')
game.printGrid(state.grid)
console.log(`Score: ${state.score}\n`)

// Make some moves
const moves: game.Direction[] = ['left', 'up', 'right', 'down']

for (let i = 0; i < 10 && !state.gameOver; i++) {
  const move = moves[Math.floor(Math.random() * moves.length)]!
  const newState = game.move(state, move)

  if (newState.grid === state.grid) {
    console.log(`Move ${move}: No change`)
    continue
  }

  state = newState
  console.log(`\nMove ${i + 1}: ${move}`)
  game.printGrid(state.grid)
  console.log(`Score: ${state.score}`)
  console.log(`Max tile: ${game.getMaxTile(state)}`)
}

if (state.gameOver) {
  console.log('\n❌ Game Over!')
} else {
  console.log('\n✅ Game still playable')
}

console.log(`\nFinal Score: ${state.score}`)
console.log(`Max Tile: ${game.getMaxTile(state)}`)
