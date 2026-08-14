import type { KisAnswer, QaAnswer, TrakeChain } from './types'

export const normalizeAnswer = (value: string) => value
  .trim()
  .toLowerCase()
  .replace(/\s+/g, ' ')
  .replace(/[.!?,;:]+$/g, '')

export function validateKis(rows: KisAnswer[]) {
  const seen = new Set<string>()
  const errors: string[] = []
  if (rows.length > 100) errors.push('No more than 100 answers are allowed.')
  rows.forEach((row, index) => {
    if (!/^L\d{2}_V\d{3}$/.test(row.videoId)) errors.push(`Row ${index + 1}: invalid video ID.`)
    if (!Number.isInteger(row.frameId) || row.frameId < 0) errors.push(`Row ${index + 1}: invalid Actual Frame ID.`)
    const key = `${row.videoId}:${row.frameId}`
    if (seen.has(key)) errors.push(`Row ${index + 1}: duplicate tuple.`)
    seen.add(key)
  })
  return { valid: errors.length === 0, errors }
}

export function validateQa(rows: QaAnswer[]) {
  const base = validateKis(rows)
  const seen = new Set<string>()
  const errors = [...base.errors]
  rows.forEach((row, index) => {
    const normalized = normalizeAnswer(row.answer)
    if (!normalized) errors.push(`Row ${index + 1}: answer is empty.`)
    const key = `${row.videoId}:${row.frameId}:${normalized}`
    if (seen.has(key)) errors.push(`Row ${index + 1}: duplicate normalized answer.`)
    seen.add(key)
  })
  return { valid: errors.length === 0, errors }
}

export function validateChain(chain: TrakeChain) {
  const complete = chain.frames.every(Number.isInteger)
  const order = chain.frames[0] < chain.frames[1] && chain.frames[1] < chain.frames[2]
  const sameVideo = /^L\d{2}_V\d{3}$/.test(chain.videoId)
  const gap = chain.frames[1] - chain.frames[0] >= 30 && chain.frames[2] - chain.frames[1] >= 30
  return { valid: complete && order && sameVideo && gap, complete, order, sameVideo, gap }
}

const escapeCsv = (value: string | number) => `"${String(value).replace(/"/g, '""')}"`

export const kisCsv = (rows: KisAnswer[]) => [
  'video_id,frame_id',
  ...rows.map((row) => `${escapeCsv(row.videoId)},${row.frameId}`),
].join('\n')

export const qaCsv = (rows: QaAnswer[]) => [
  'video_id,frame_id,answer',
  ...rows.map((row) => `${escapeCsv(row.videoId)},${row.frameId},${escapeCsv(normalizeAnswer(row.answer))}`),
].join('\n')

export const trakeCsv = (rows: TrakeChain[]) => [
  'video_id,frame_id_1,frame_id_2,frame_id_3',
  ...rows.map((row) => `${escapeCsv(row.videoId)},${row.frames.join(',')}`),
].join('\n')

export function downloadCsv(name: string, csv: string) {
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = name
  anchor.click()
  URL.revokeObjectURL(url)
}
