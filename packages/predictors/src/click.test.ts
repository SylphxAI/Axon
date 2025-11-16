import { describe, expect, test } from 'bun:test'
import type { ClickEvent } from './click'
import { ClickPredictor } from './click'

describe('ClickPredictor', () => {
  test('initialization', () => {
    const predictor = new ClickPredictor()
    const metrics = predictor.getMetrics()
    expect(metrics.predictions).toBe(0)
  })

  test('prediction returns probability', () => {
    const predictor = new ClickPredictor()
    const prob = predictor.predict({
      position: { x: 100, y: 100 },
      viewport: { width: 1920, height: 1080 },
    })

    expect(prob).toBeGreaterThanOrEqual(0)
    expect(prob).toBeLessThanOrEqual(1)
  })

  test('learning from click events', () => {
    const predictor = new ClickPredictor({
      learningRate: 0.1,
    })

    const clickedEvent: ClickEvent = {
      context: {
        position: { x: 100, y: 100 },
        viewport: { width: 1920, height: 1080 },
        elementType: 'button',
      },
      clicked: true,
    }

    const notClickedEvent: ClickEvent = {
      context: {
        position: { x: 1800, y: 1000 },
        viewport: { width: 1920, height: 1080 },
        elementType: 'text',
      },
      clicked: false,
    }

    for (let i = 0; i < 50; i++) {
      predictor.learn(clickedEvent)
      predictor.learn(notClickedEvent)
    }

    const clickedProb = predictor.predict(clickedEvent.context)
    const notClickedProb = predictor.predict(notClickedEvent.context)

    expect(clickedProb).toBeGreaterThan(notClickedProb)
  })

  test('willClick threshold', () => {
    const predictor = new ClickPredictor({
      threshold: 0.7,
    })

    const context = {
      position: { x: 100, y: 100 },
      viewport: { width: 1920, height: 1080 },
    }

    const prediction = predictor.predict(context)
    const willClick = predictor.willClick(context)

    expect(willClick).toBe(prediction > 0.7)
  })

  test('export and import', () => {
    const predictor = new ClickPredictor()

    predictor.learn({
      context: {
        position: { x: 100, y: 100 },
        viewport: { width: 1920, height: 1080 },
      },
      clicked: true,
    })

    const exported = predictor.export()
    const newPredictor = new ClickPredictor()
    newPredictor.import(exported)

    const context = {
      position: { x: 100, y: 100 },
      viewport: { width: 1920, height: 1080 },
    }

    expect(predictor.predict(context)).toBeCloseTo(newPredictor.predict(context), 5)
  })
})
