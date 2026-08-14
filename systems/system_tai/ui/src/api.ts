/**
 * Typed API Client for system_tai Gateway (conforming to Sheet 09 Accepted V1 Contract).
 * Interacts directly with the live Backend REST API.
 */

import type { Candidate, KisAnswer, QaAnswer, TrakeChain } from './types'

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
  if (!query.trim()) return []
  try {
    const res = await fetch(`${API_BASE}/kis/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query.trim(),
        filters,
        query_variants: variants,
        top_k: 100,
        interaction_mode: 'interactive',
      }),
    })
    if (res.ok) {
      const json: ResponseEnvelope<{ candidates: Candidate[] }> = await res.json()
      if (json.data && Array.isArray(json.data.candidates)) {
        return json.data.candidates
      }
    }
  } catch (err) {
    console.error('API KIS search error:', err)
  }
  return []
}

export async function apiSearchQa(
  event: string,
  question: string,
  temporal: string = 'during',
  answerType: string = 'automatic'
): Promise<{ candidates: Candidate[]; answers: QaAnswer[]; detectedType?: string }> {
  if (!event.trim() && !question.trim()) {
    return { candidates: [], answers: [], detectedType: undefined }
  }
  try {
    const res = await fetch(`${API_BASE}/qa/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        event_description: event.trim(),
        question: question.trim(),
        temporal_relation: temporal.toLowerCase(),
        answer_type: answerType.toLowerCase(),
        top_k: 100,
      }),
    })
    if (res.ok) {
      const json: ResponseEnvelope<{
        candidates: Candidate[]
        answers: QaAnswer[]
        detected_answer_type: string
      }> = await res.json()
      if (json.data) {
        return {
          candidates: json.data.candidates || [],
          answers: json.data.answers || [],
          detectedType: json.data.detected_answer_type,
        }
      }
    }
  } catch (err) {
    console.error('API QA search error:', err)
  }
  return { candidates: [], answers: [], detectedType: undefined }
}

export async function apiSearchTrake(events: string[]): Promise<TrakeChain[]> {
  const filteredEvents = events.map((e) => e.trim()).filter(Boolean)
  if (filteredEvents.length === 0) return []
  try {
    const res = await fetch(`${API_BASE}/trake/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        events: filteredEvents,
        top_k_chains: 100,
        beam_width: 100,
      }),
    })
    if (res.ok) {
      const json: ResponseEnvelope<{
        chains: Array<{ videoId: string; frames: number[]; confidence: number }>
      }> = await res.json()
      if (json.data && Array.isArray(json.data.chains)) {
        return json.data.chains.map((c) => ({
          videoId: c.videoId,
          frames: (c.frames.length >= 3
            ? c.frames.slice(0, 3)
            : [c.frames[0] || 0, c.frames[1] || 0, c.frames[2] || 0]) as [
            number,
            number,
            number,
          ],
          confidence: c.confidence,
        }))
      }
    }
  } catch (err) {
    console.error('API TRAKE search error:', err)
  }
  return []
}

export async function apiValidateSubmission(
  taskType: 'KIS' | 'Q&A' | 'TRAKE',
  records: Array<{ videoId: string; frameId?: number; frames?: number[]; answer?: string }>
): Promise<{ valid: boolean; errors: string[]; warnings: string[] }> {
  try {
    const res = await fetch(`${API_BASE}/submissions/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        task_type: taskType,
        records: records.map((r) => ({
          video_id: r.videoId,
          frame_id: r.frameId,
          frame_ids: r.frames,
          answer: r.answer,
        })),
      }),
    })
    if (res.ok) {
      const json: ResponseEnvelope<{
        valid: boolean
        errors: string[]
        warnings: string[]
      }> = await res.json()
      if (json.data) {
        return json.data
      }
    }
  } catch (err) {
    console.error('API validation error:', err)
  }
  return { valid: true, errors: [], warnings: [] }
}
