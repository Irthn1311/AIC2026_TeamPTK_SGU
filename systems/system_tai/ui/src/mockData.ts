import type { Candidate, EvidenceTab, TrakeChain } from './types'

const frames = (center: number, minute: number) => [-12, -6, 0, 6, 12].map((delta, index) => ({
  id: center + delta,
  timestamp: `00:${String(minute).padStart(2, '0')}:${String(6 + index * 3).padStart(2, '0')}`,
}))

export const candidates: Candidate[] = [
  { videoId: 'L21_V001', frameId: 4592, timestamp: '00:03:12', score: 0.94, badges: ['VISUAL', 'OBJECT'], neighbors: frames(4592, 3) },
  { videoId: 'L21_V014', frameId: 10244, timestamp: '00:06:50', score: 0.88, badges: ['VISUAL', 'OCR'], neighbors: frames(10244, 6) },
  { videoId: 'L22_V005', frameId: 321, timestamp: '00:00:15', score: 0.82, badges: ['ASR', 'VISUAL'], neighbors: frames(321, 0) },
  { videoId: 'L22_V019', frameId: 6804, timestamp: '00:04:31', score: 0.79, badges: ['VISUAL', 'METADATA'], neighbors: frames(6804, 4) },
  { videoId: 'L23_V002', frameId: 13102, timestamp: '00:09:02', score: 0.76, badges: ['OCR', 'OBJECT'], neighbors: frames(13102, 9) },
  { videoId: 'L24_V010', frameId: 2640, timestamp: '00:01:45', score: 0.71, badges: ['ASR', 'METADATA'], neighbors: frames(2640, 1) },
]

export const evidenceByTab: Record<EvidenceTab, string> = {
  Visual: 'Strong visual match: a person approaches a red city bus beside the curb.',
  OCR: 'Detected text: “CITY LOOP 14” and partial stop signage.',
  ASR: 'Transcript: “The next bus arrives at the central station.”',
  Object: 'Detected objects: person 0.97, bus 0.94, backpack 0.78, curb 0.71.',
  Metadata: 'Shot 42 · daylight · outdoor · urban transit · 30 FPS.',
}

export const qaHypotheses = [
  { answer: 'A red bus', confidence: 0.94 },
  { answer: 'A yellow taxi', confidence: 0.12 },
]

export const trakeChains: TrakeChain[] = [
  { videoId: 'L21_V001', frames: [4240, 4592, 5012], confidence: 0.93 },
  { videoId: 'L21_V001', frames: [4264, 4620, 5078], confidence: 0.87 },
  { videoId: 'L22_V019', frames: [6201, 6804, 7310], confidence: 0.81 },
]

export const trakeEvents = [
  'A person approaches a bus stop',
  'The person boards the bus',
  'The bus leaves the station',
]
