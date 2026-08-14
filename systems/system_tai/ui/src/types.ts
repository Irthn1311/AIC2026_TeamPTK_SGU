export type Task = 'KIS' | 'Q&A' | 'TRAKE'
export type EvidenceTab = 'Visual' | 'OCR' | 'ASR' | 'Object' | 'Metadata'

export interface Frame {
  id: number
  timestamp: string
}

export interface Candidate {
  videoId: string
  frameId: number
  timestamp: string
  score: number
  badges: string[]
  neighbors: Frame[]
}

export interface KisAnswer {
  videoId: string
  frameId: number
  confidence: number
}

export interface QaAnswer extends KisAnswer {
  answer: string
  validation: 'VALID' | 'INVALID'
}

export interface TrakeChain {
  videoId: string
  frames: [number, number, number]
  confidence: number
}
