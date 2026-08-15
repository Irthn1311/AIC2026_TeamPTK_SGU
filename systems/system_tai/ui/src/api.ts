/**
 * Typed API Client for system_tai Gateway (conforming to Sheet 09 Accepted V1 Contract).
 * Interacts directly with the live Backend REST API.
 */

import type { Candidate, QaAnswer, TrakeChain } from './types'

const API_BASE = '/api/v1'

interface ResponseEnvelope<T> {
  meta: {
    request_id: string
    api_contract_version: string
    dataset_batch: string
    index_version: string
    mapping_version: string
    latency_ms: number
    server_time: string
  }
  data: T
}

export async function apiSearchKis(
  query: string,
  filters: string[] = [],
  variants: string[] = []
): Promise<Candidate[]> {
  try {
    const res = await fetch(`${API_BASE}/kis/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        filters,
        query_variants: variants,
        top_k: 100,
      }),
    })
    if (!res.ok) return []
    const json: ResponseEnvelope<{ candidates: Candidate[] }> = await res.json()
    return json.data.candidates || []
  } catch {
    return []
  }
}

export async function apiRefineKis(
  videoId: string,
  centerFrameId: number,
  windowSeconds: number = 1.0
): Promise<{ recommendedFrame: number; neighbors: Array<{ id: number; timestamp: string }> }> {
  try {
    const res = await fetch(`${API_BASE}/kis/refine`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video_id: videoId,
        center_actual_frame_id: centerFrameId,
        window_seconds: windowSeconds,
      }),
    })
    if (!res.ok) return { recommendedFrame: centerFrameId, neighbors: [] }
    const json: ResponseEnvelope<{ recommended_frame: number; neighboring_frames: Array<{ id: number; timestamp: string }> }> = await res.json()
    return {
      recommendedFrame: json.data.recommended_frame,
      neighbors: json.data.neighboring_frames || [],
    }
  } catch {
    return { recommendedFrame: centerFrameId, neighbors: [] }
  }
}

export async function apiSearchQa(
  eventDescription: string,
  question: string,
  temporalRelation: string = 'during',
  suggestedAnswerType: string = 'auto'
): Promise<{ candidates: Candidate[]; answers: QaAnswer[] }> {
  try {
    const res = await fetch(`${API_BASE}/qa/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        event_description: eventDescription,
        question,
        temporal_relation: temporalRelation,
        suggested_answer_type: suggestedAnswerType,
        top_k: 100,
      }),
    })
    if (!res.ok) return { candidates: [], answers: [] }
    const json: ResponseEnvelope<{ candidates: Candidate[]; answers: QaAnswer[] }> = await res.json()
    return {
      candidates: json.data.candidates || [],
      answers: json.data.answers || [],
    }
  } catch {
    return { candidates: [], answers: [] }
  }
}

export async function apiSearchTrake(
  events: string[]
): Promise<TrakeChain[]> {
  try {
    const res = await fetch(`${API_BASE}/trake/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        events,
        top_k_chains: 100,
      }),
    })
    if (!res.ok) return []
    const json: ResponseEnvelope<{ chains?: TrakeChain[]; top_chains?: TrakeChain[] }> = await res.json()
    return json.data.chains || json.data.top_chains || []
  } catch {
    return []
  }
}

export async function apiValidateSubmission(
  taskType: 'KIS' | 'Q&A' | 'TRAKE',
  records: Record<string, unknown>[]
): Promise<{ valid: boolean; errors: string[]; warnings: string[] }> {
  try {
    const res = await fetch(`${API_BASE}/submissions/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        task_type: taskType,
        records,
      }),
    })
    if (!res.ok) return { valid: false, errors: [`HTTP error ${res.status}`], warnings: [] }
    const json: ResponseEnvelope<{ valid: boolean; errors: string[]; warnings: string[] }> = await res.json()
    return {
      valid: json.data.valid,
      errors: json.data.errors || [],
      warnings: json.data.warnings || [],
    }
  } catch (err: unknown) {
    return { valid: false, errors: [String(err)], warnings: [] }
  }
}

export async function apiExportSubmission(
  taskType: 'KIS' | 'Q&A' | 'TRAKE',
  records: Record<string, unknown>[]
): Promise<Blob> {
  const res = await fetch(`${API_BASE}/submissions/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      task_type: taskType,
      records,
    }),
  })
  if (!res.ok) throw new Error(`HTTP error ${res.status}`)
  return await res.blob()
}
