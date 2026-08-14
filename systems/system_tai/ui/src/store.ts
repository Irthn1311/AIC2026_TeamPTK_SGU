import { create } from 'zustand'
import { apiSearchKis, apiSearchQa, apiSearchTrake } from './api'
import { trakeEvents } from './mockData'
import type { Candidate, EvidenceTab, KisAnswer, QaAnswer, Task, TrakeChain } from './types'
import { normalizeAnswer, validateChain } from './utils'

interface KisState {
  query: string
  filters: string[]
  variants: string[]
  searched: boolean
  loading: boolean
  results: Candidate[]
  selectedVideoId?: string
  selectedFrameId?: number
  evidenceTab: EvidenceTab
  answers: KisAnswer[]
  message?: string
}

interface QaState {
  event: string
  question: string
  temporal: string
  answerType: string
  searched: boolean
  loading: boolean
  results: Candidate[]
  selectedVideoId?: string
  selectedFrameId?: number
  intervalReady: boolean
  evidenceTab: EvidenceTab
  canonical: string
  verified: boolean
  answers: QaAnswer[]
  message?: string
}

interface TrakeState {
  query: string
  events: string[]
  searched: boolean
  loading: boolean
  selectedVideoId?: string
  chains: TrakeChain[]
  activeChain?: TrakeChain
  verified: boolean
  answers: TrakeChain[]
  message?: string
}

interface WorkspaceStore {
  activeTask: Task
  mode: 'Interactive' | 'Automatic'
  historyOpen: boolean
  history: string[]
  kis: KisState
  qa: QaState
  trake: TrakeState
  setTask: (task: Task) => void
  setMode: (mode: 'Interactive' | 'Automatic') => void
  toggleHistory: () => void
  updateKis: (patch: Partial<KisState>) => void
  searchKis: () => Promise<void>
  selectKisCandidate: (candidate: Candidate) => void
  addKisAnswer: () => void
  removeKisAnswer: (index: number) => void
  resetKis: () => void
  updateQa: (patch: Partial<QaState>) => void
  searchQa: () => Promise<void>
  selectQaCandidate: (candidate: Candidate) => void
  addQaAnswer: () => void
  removeQaAnswer: (index: number) => void
  resetQa: () => void
  updateTrake: (patch: Partial<TrakeState>) => void
  searchTrake: () => Promise<void>
  selectChain: (chain: TrakeChain) => void
  verifyChain: () => void
  addChain: () => void
  removeChain: (index: number) => void
  resetTrake: () => void
}

export const initialKis: KisState = {
  query: '',
  filters: ['Visual', 'OCR', 'ASR', 'Object', 'Metadata'],
  variants: [],
  searched: false,
  loading: false,
  results: [],
  evidenceTab: 'Visual',
  answers: [],
}

export const initialQa: QaState = {
  event: '',
  question: '',
  temporal: 'During',
  answerType: 'Automatic',
  searched: false,
  loading: false,
  results: [],
  intervalReady: false,
  evidenceTab: 'Visual',
  canonical: '',
  verified: false,
  answers: [],
}

export const initialTrake: TrakeState = {
  query: 'A person approaches a bus stop, boards the bus, then the bus leaves.',
  events: trakeEvents,
  searched: false,
  loading: false,
  chains: [],
  verified: false,
  answers: [],
}

export const useWorkspaceStore = create<WorkspaceStore>((set, get) => ({
  activeTask: 'KIS',
  mode: 'Interactive',
  historyOpen: false,
  history: [],
  kis: { ...initialKis },
  qa: { ...initialQa },
  trake: { ...initialTrake, events: [...trakeEvents] },

  setTask: (activeTask) => set({ activeTask }),
  setMode: (mode) => set({ mode }),
  toggleHistory: () => set((s) => ({ historyOpen: !s.historyOpen })),

  // --- KIS Actions ---
  updateKis: (patch) => set((s) => ({ kis: { ...s.kis, ...patch } })),
  searchKis: async () => {
    const { kis, mode } = get()
    const query = kis.query.trim()
    set((s) => ({ kis: { ...s.kis, loading: true, searched: true, message: undefined } }))

    const results = await apiSearchKis(query, kis.filters, kis.variants)

    set((s) => ({
      kis: {
        ...s.kis,
        loading: false,
        results,
        selectedVideoId: mode === 'Automatic' && results.length > 0 ? results[0].videoId : s.kis.selectedVideoId || (results.length > 0 ? results[0].videoId : undefined),
        selectedFrameId: mode === 'Automatic' && results.length > 0 ? results[0].frameId : s.kis.selectedFrameId || (results.length > 0 ? results[0].frameId : undefined),
      },
      history: query ? [`KIS · ${query}`, ...s.history].slice(0, 8) : s.history,
    }))
  },
  selectKisCandidate: (candidate) =>
    set((s) => ({
      kis: {
        ...s.kis,
        selectedVideoId: candidate.videoId,
        selectedFrameId: candidate.frameId,
        evidenceTab: 'Visual',
        message: undefined,
      },
    })),
  addKisAnswer: () => {
    const { kis } = get()
    const candidate = kis.results.find((item) => item.videoId === kis.selectedVideoId)
    if (!candidate || kis.selectedFrameId === undefined) return
    const duplicate = kis.answers.some(
      (row) => row.videoId === candidate.videoId && row.frameId === kis.selectedFrameId
    )
    if (duplicate)
      return set((s) => ({ kis: { ...s.kis, message: 'Duplicate video_id + frame_id rejected.' } }))
    if (kis.answers.length >= 100)
      return set((s) => ({
        kis: { ...s.kis, message: 'The 100-answer limit has been reached.' },
      }))
    set((s) => ({
      kis: {
        ...s.kis,
        answers: [
          ...s.kis.answers,
          {
            videoId: candidate.videoId,
            frameId: kis.selectedFrameId!,
            confidence: candidate.score,
          },
        ],
        message: 'Frame added to the ranked list.',
      },
    }))
  },
  removeKisAnswer: (index) =>
    set((s) => ({ kis: { ...s.kis, answers: s.kis.answers.filter((_, i) => i !== index) } })),
  resetKis: () => set({ kis: { ...initialKis } }),

  // --- Q&A Actions ---
  updateQa: (patch) => set((s) => ({ qa: { ...s.qa, ...patch } })),
  searchQa: async () => {
    const { qa, mode } = get()
    const query = `${qa.event} ${qa.question}`.trim()
    set((s) => ({ qa: { ...s.qa, loading: true, searched: true, message: undefined } }))

    const { candidates: results, answers } = await apiSearchQa(
      qa.event,
      qa.question,
      qa.temporal,
      qa.answerType
    )

    const defaultCanonical = answers.length > 0 ? answers[0].answer : 'A red bus'

    set((s) => ({
      qa: {
        ...s.qa,
        loading: false,
        results,
        selectedVideoId:
          mode === 'Automatic' && results.length > 0 ? results[0].videoId : s.qa.selectedVideoId || (results.length > 0 ? results[0].videoId : undefined),
        selectedFrameId:
          mode === 'Automatic' && results.length > 0 ? results[0].frameId : s.qa.selectedFrameId || (results.length > 0 ? results[0].frameId : undefined),
        intervalReady: mode === 'Automatic',
        canonical: mode === 'Automatic' ? defaultCanonical : s.qa.canonical || defaultCanonical,
        verified: mode === 'Automatic',
      },
      history: query ? [`Q&A · ${query}`, ...s.history].slice(0, 8) : s.history,
    }))
  },
  selectQaCandidate: (candidate) =>
    set((s) => ({
      qa: {
        ...s.qa,
        selectedVideoId: candidate.videoId,
        selectedFrameId: candidate.frameId,
        intervalReady: true,
        verified: true,
        canonical: s.qa.canonical || 'A red bus',
        message: undefined,
      },
    })),
  addQaAnswer: () => {
    const { qa } = get()
    const candidate = qa.results.find((item) => item.videoId === qa.selectedVideoId)
    const answer = normalizeAnswer(qa.canonical)
    if (!candidate || qa.selectedFrameId === undefined || !qa.intervalReady || !qa.verified || !answer)
      return
    const duplicate = qa.answers.some(
      (row) =>
        row.videoId === candidate.videoId &&
        row.frameId === qa.selectedFrameId &&
        normalizeAnswer(row.answer) === answer
    )
    if (duplicate)
      return set((s) => ({
        qa: { ...s.qa, message: 'Duplicate normalized Q&A tuple rejected.' },
      }))
    set((s) => ({
      qa: {
        ...s.qa,
        answers: [
          ...s.qa.answers,
          {
            videoId: candidate.videoId,
            frameId: qa.selectedFrameId!,
            answer,
            confidence: candidate.score,
            validation: 'VALID',
          },
        ],
        message: 'Verified answer added.',
      },
    }))
  },
  removeQaAnswer: (index) =>
    set((s) => ({ qa: { ...s.qa, answers: s.qa.answers.filter((_, i) => i !== index) } })),
  resetQa: () => set({ qa: { ...initialQa } }),

  // --- TRAKE Actions ---
  updateTrake: (patch) => set((s) => ({ trake: { ...s.trake, ...patch } })),
  searchTrake: async () => {
    const { trake, mode } = get()
    set((s) => ({ trake: { ...s.trake, loading: true, searched: true, message: undefined } }))

    const chains = await apiSearchTrake(trake.events)

    set((s) => ({
      trake: {
        ...s.trake,
        loading: false,
        chains,
        selectedVideoId:
          mode === 'Automatic' && chains.length > 0 ? chains[0].videoId : s.trake.selectedVideoId || (chains.length > 0 ? chains[0].videoId : undefined),
        activeChain:
          mode === 'Automatic' && chains.length > 0 ? chains[0] : s.trake.activeChain || (chains.length > 0 ? chains[0] : undefined),
        verified: mode === 'Automatic',
      },
      history: [`TRAKE · ${s.trake.query}`, ...s.history].slice(0, 8),
    }))
  },
  selectChain: (activeChain) =>
    set((s) => ({
      trake: {
        ...s.trake,
        activeChain: {
          ...activeChain,
          frames: [...activeChain.frames] as [number, number, number],
        },
        selectedVideoId: activeChain.videoId,
        verified: false,
        message: undefined,
      },
    })),
  verifyChain: () => {
    const chain = get().trake.activeChain
    set((s) => ({
      trake: {
        ...s.trake,
        verified: !!chain && validateChain(chain).valid,
        message:
          chain && validateChain(chain).valid
            ? 'Chain verification passed.'
            : 'Chain must be complete, same-video, ordered, and satisfy gaps.',
      },
    }))
  },
  addChain: () => {
    const { trake } = get()
    const chain = trake.activeChain
    if (!chain || !trake.verified || !validateChain(chain).valid) return
    const duplicate = trake.answers.some(
      (row) =>
        row.videoId === chain.videoId &&
        row.frames.every((frame, i) => frame === chain.frames[i])
    )
    if (duplicate)
      return set((s) => ({ trake: { ...s.trake, message: 'Duplicate chain rejected.' } }))
    set((s) => ({
      trake: {
        ...s.trake,
        answers: [...s.trake.answers, chain],
        message: 'Verified chain added.',
      },
    }))
  },
  removeChain: (index) =>
    set((s) => ({ trake: { ...s.trake, answers: s.trake.answers.filter((_, i) => i !== index) } })),
  resetTrake: () => set({ trake: { ...initialTrake, events: [...trakeEvents] } }),
}))
