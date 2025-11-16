/**
 * 2048 Game - Pure functional implementation
 * Perfect for RL training
 */

export type Grid = readonly (readonly number[])[]

export type GameState = {
  readonly grid: Grid
  readonly score: number
  readonly gameOver: boolean
}

export type Direction = 'up' | 'down' | 'left' | 'right'

/**
 * Initialize new game
 * Pure function - returns initial state
 */
export function init(): GameState {
  const grid = createEmptyGrid()
  const withTile1 = addRandomTile(grid)
  const withTile2 = addRandomTile(withTile1)

  return {
    grid: withTile2,
    score: 0,
    gameOver: false,
  }
}

/**
 * Create empty 4x4 grid
 */
function createEmptyGrid(): Grid {
  return Array(4)
    .fill(0)
    .map(() => Array(4).fill(0))
}

/**
 * Add random tile (2 or 4) to empty position
 * Pure function - returns new grid
 */
function addRandomTile(grid: Grid): Grid {
  const emptyCells: Array<[number, number]> = []

  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      if (grid[i]![j] === 0) {
        emptyCells.push([i, j])
      }
    }
  }

  if (emptyCells.length === 0) return grid

  const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)]!
  const value = Math.random() < 0.9 ? 2 : 4

  const newGrid = grid.map((row) => [...row])
  newGrid[randomCell[0]]![randomCell[1]] = value

  return newGrid
}

/**
 * Make a move
 * Pure function - returns new state
 */
export function move(state: GameState, direction: Direction): GameState {
  if (state.gameOver) return state

  const { newGrid, scoreGained, moved } = moveGrid(state.grid, direction)

  if (!moved) {
    // No change, check if game over
    if (isGameOver(state.grid)) {
      return { ...state, gameOver: true }
    }
    return state
  }

  // Add new tile after successful move
  const gridWithNewTile = addRandomTile(newGrid)

  // Check if game over
  const gameOver = isGameOver(gridWithNewTile)

  return {
    grid: gridWithNewTile,
    score: state.score + scoreGained,
    gameOver,
  }
}

/**
 * Move grid in direction
 * Pure function - returns new grid and score gained
 */
function moveGrid(
  grid: Grid,
  direction: Direction
): { newGrid: Grid; scoreGained: number; moved: boolean } {
  let rotatedGrid = rotateGridForDirection(grid, direction)
  const { newGrid: movedGrid, scoreGained } = moveLeft(rotatedGrid)
  const finalGrid = rotateGridBack(movedGrid, direction)

  // Check if anything changed
  const moved = !gridEquals(grid, finalGrid)

  return { newGrid: finalGrid, scoreGained, moved }
}

/**
 * Move all tiles left and merge
 * Pure function
 */
function moveLeft(grid: Grid): { newGrid: Grid; scoreGained: number } {
  let scoreGained = 0
  const newGrid = grid.map((row) => {
    const { newRow, score } = mergeLine(row)
    scoreGained += score
    return newRow
  })

  return { newGrid, scoreGained }
}

/**
 * Merge a line to the left
 * Pure function
 */
function mergeLine(line: readonly number[]): { newRow: number[]; score: number } {
  // Remove zeros
  const nonZero = line.filter((x) => x !== 0)

  // Merge adjacent same values
  const merged: number[] = []
  let score = 0
  let skip = false

  for (let i = 0; i < nonZero.length; i++) {
    if (skip) {
      skip = false
      continue
    }

    if (i < nonZero.length - 1 && nonZero[i] === nonZero[i + 1]) {
      const value = nonZero[i]! * 2
      merged.push(value)
      score += value
      skip = true
    } else {
      merged.push(nonZero[i]!)
    }
  }

  // Pad with zeros
  while (merged.length < 4) {
    merged.push(0)
  }

  return { newRow: merged, score }
}

/**
 * Rotate grid to make direction "left"
 */
function rotateGridForDirection(grid: Grid, direction: Direction): Grid {
  switch (direction) {
    case 'left':
      return grid
    case 'right':
      return grid.map((row) => [...row].reverse())
    case 'up':
      return transpose(grid)
    case 'down':
      return transpose(grid).map((row) => [...row].reverse())
  }
}

/**
 * Rotate grid back after move
 */
function rotateGridBack(grid: Grid, direction: Direction): Grid {
  switch (direction) {
    case 'left':
      return grid
    case 'right':
      return grid.map((row) => [...row].reverse())
    case 'up':
      return transpose(grid)
    case 'down':
      const reversed = grid.map((row) => [...row].reverse())
      return transpose(reversed)
  }
}

/**
 * Transpose grid (swap rows and columns)
 */
function transpose(grid: Grid): Grid {
  return grid[0]!.map((_, i) => grid.map((row) => row[i]!))
}

/**
 * Check if two grids are equal
 */
function gridEquals(a: Grid, b: Grid): boolean {
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      if (a[i]![j] !== b[i]![j]) return false
    }
  }
  return true
}

/**
 * Check if game is over (no more moves possible)
 */
function isGameOver(grid: Grid): boolean {
  // Check if any empty cells
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      if (grid[i]![j] === 0) return false
    }
  }

  // Check if any adjacent tiles can merge
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      const value = grid[i]![j]!
      // Check right
      if (j < 3 && grid[i]![j + 1] === value) return false
      // Check down
      if (i < 3 && grid[i + 1]![j] === value) return false
    }
  }

  return true
}

/**
 * Get available actions
 */
export function getAvailableActions(state: GameState): Direction[] {
  const actions: Direction[] = []
  const directions: Direction[] = ['up', 'down', 'left', 'right']

  for (const dir of directions) {
    const { moved } = moveGrid(state.grid, dir)
    if (moved) actions.push(dir)
  }

  return actions
}

/**
 * Get max tile value
 */
export function getMaxTile(state: GameState): number {
  let max = 0
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      max = Math.max(max, state.grid[i]![j]!)
    }
  }
  return max
}

/**
 * Convert grid to flat array for neural network input
 */
export function gridToArray(grid: Grid): number[] {
  return grid.flat()
}

/**
 * Print grid to console (for debugging)
 */
export function printGrid(grid: Grid): void {
  console.log('┌────────┬────────┬────────┬────────┐')
  for (let i = 0; i < 4; i++) {
    const row = grid[i]!
    const formatted = row.map((v) => (v === 0 ? '   ' : String(v).padStart(4, ' ')))
    console.log(`│${formatted.join('│')}│`)
    if (i < 3) {
      console.log('├────────┼────────┼────────┼────────┤')
    }
  }
  console.log('└────────┴────────┴────────┴────────┘')
}
