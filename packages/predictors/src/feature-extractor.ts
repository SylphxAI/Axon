export interface ClickContext {
  elementType?: string
  position?: { x: number; y: number }
  timestamp?: number
  pageDepth?: number
  timeOnPage?: number
  previousClicks?: number
  deviceType?: 'mobile' | 'tablet' | 'desktop'
  viewport?: { width: number; height: number }
}

export interface SequenceContext {
  actions: string[]
  timestamps?: number[]
  maxLength?: number
}

export function clickToVector(context: ClickContext, vectorSize = 16): Float32Array {
  const features = new Float32Array(vectorSize)
  let idx = 0

  if (context.position && context.viewport) {
    features[idx++] = context.position.x / context.viewport.width
    features[idx++] = context.position.y / context.viewport.height
  } else {
    idx += 2
  }

  if (context.timestamp && context.timeOnPage) {
    const hour = new Date(context.timestamp).getHours() / 24
    features[idx++] = hour
    features[idx++] = Math.min(context.timeOnPage / 60000, 1)
  } else {
    idx += 2
  }

  features[idx++] = context.pageDepth ?? 0
  features[idx++] = context.previousClicks ?? 0

  if (context.deviceType) {
    features[idx++] = context.deviceType === 'mobile' ? 1 : 0
    features[idx++] = context.deviceType === 'tablet' ? 1 : 0
    features[idx++] = context.deviceType === 'desktop' ? 1 : 0
  } else {
    idx += 3
  }

  const elementTypes = ['button', 'link', 'input', 'image', 'text']
  const elementIdx = context.elementType
    ? elementTypes.indexOf(context.elementType.toLowerCase())
    : -1
  for (let i = 0; i < elementTypes.length; i++) {
    features[idx++] = i === elementIdx ? 1 : 0
  }

  return features
}

export function sequenceToVector(context: SequenceContext, vectorSize = 32): Float32Array {
  const features = new Float32Array(vectorSize)
  const maxLength = context.maxLength ?? 10
  const actions = context.actions.slice(-maxLength)

  const actionMap = new Map<string, number>()
  let actionIdx = 0

  for (const action of actions) {
    if (!actionMap.has(action)) {
      actionMap.set(action, actionIdx++)
    }
  }

  for (let i = 0; i < actions.length && i < maxLength; i++) {
    const action = actions[i]!
    const idx = actionMap.get(action) ?? 0
    if (idx < vectorSize / 2) {
      features[idx] = 1
    }
  }

  if (context.timestamps && context.timestamps.length > 1) {
    const intervals: number[] = []
    for (let i = 1; i < context.timestamps.length; i++) {
      intervals.push(context.timestamps[i]! - context.timestamps[i - 1]!)
    }
    const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length
    features[vectorSize - 1] = Math.min(avgInterval / 1000, 1)
  }

  return features
}

export function normalizeFeatures(features: Float32Array): Float32Array {
  const normalized = new Float32Array(features.length)
  let sum = 0

  for (let i = 0; i < features.length; i++) {
    sum += features[i]! * features[i]!
  }

  const norm = Math.sqrt(sum)
  if (norm > 0) {
    for (let i = 0; i < features.length; i++) {
      normalized[i] = features[i]! / norm
    }
  }

  return normalized
}
