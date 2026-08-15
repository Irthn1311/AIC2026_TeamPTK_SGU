import '@testing-library/jest-dom'
import { act, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'
import { initialKis, initialQa, initialTrake, useWorkspaceStore } from './store'
import { kisCsv, normalizeAnswer, qaCsv, trakeCsv, validateChain, validateKis } from './utils'

const mockCandidates = [
  {
    videoId: 'L21_V001',
    frameId: 4592,
    timestamp: '00:03:12',
    score: 0.94,
    badges: ['VISUAL', 'OBJECT'],
    neighbors: [
      { id: 4580, timestamp: '00:03:09' },
      { id: 4592, timestamp: '00:03:12' },
      { id: 4604, timestamp: '00:03:15' },
    ],
  },
  {
    videoId: 'L21_V014',
    frameId: 10244,
    timestamp: '00:06:50',
    score: 0.88,
    badges: ['VISUAL', 'OCR'],
    neighbors: [
      { id: 10230, timestamp: '00:06:47' },
      { id: 10244, timestamp: '00:06:50' },
      { id: 10260, timestamp: '00:06:53' },
    ],
  },
]

function populateKisResults() {
  useWorkspaceStore.setState((s) => ({
    kis: {
      ...s.kis,
      searched: true,
      results: mockCandidates,
      selectedVideoId: 'L21_V001',
      selectedFrameId: 4592,
    },
  }))
}

describe('AIC retrieval workspace', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    useWorkspaceStore.setState({
      activeTask: 'KIS',
      mode: 'Interactive',
      history: [],
      historyOpen: false,
      kis: { ...initialKis },
      qa: { ...initialQa },
      trake: { ...initialTrake, events: [...initialTrake.events] },
    })
  })

  it('renders candidates when results exist', () => {
    render(<App />)
    act(() => populateKisResults())
    expect(screen.getByTestId('candidate-L21_V001')).toBeInTheDocument()
  })

  it('selecting a candidate updates the inspector', () => {
    render(<App />)
    act(() => populateKisResults())
    fireEvent.click(screen.getByTestId('candidate-L21_V014'))
    expect(screen.getByTestId('inspector-frame-id')).toHaveTextContent('10244')
  })

  it('selecting a neighboring frame updates Actual Frame ID', () => {
    render(<App />)
    act(() => populateKisResults())
    fireEvent.click(screen.getByTestId('candidate-L21_V001'))
    fireEvent.click(screen.getByTestId('neighbor-4580'))
    expect(screen.getByTestId('inspector-frame-id')).toHaveTextContent('4580')
  })

  it('evidence tabs change content', () => {
    render(<App />)
    act(() => populateKisResults())
    fireEvent.click(screen.getByTestId('candidate-L21_V001'))
    fireEvent.click(screen.getByRole('tab', { name: 'OCR' }))
    expect(screen.getByTestId('evidence-content')).toHaveTextContent('OCR detection stream indexed')
  })

  it('KIS add answer works', () => {
    render(<App />)
    act(() => populateKisResults())
    fireEvent.click(screen.getByTestId('candidate-L21_V001'))
    fireEvent.click(screen.getByRole('button', { name: 'Select Exact Frame & Add to List' }))
    expect(useWorkspaceStore.getState().kis.answers).toHaveLength(1)
  })

  it('KIS duplicate is rejected', () => {
    render(<App />)
    act(() => populateKisResults())
    fireEvent.click(screen.getByTestId('candidate-L21_V001'))
    const add = screen.getByRole('button', { name: 'Select Exact Frame & Add to List' })
    fireEvent.click(add)
    fireEvent.click(add)
    expect(screen.getByText(/Duplicate video_id/)).toBeInTheDocument()
    expect(useWorkspaceStore.getState().kis.answers).toHaveLength(1)
  })

  it('Q&A Add Answer remains disabled in invalid state', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Q&A' }))
    expect(screen.getByRole('button', { name: 'Add Answer' })).toBeDisabled()
  })

  it('Q&A canonical answer normalization handles case, spaces and punctuation', () => {
    expect(normalizeAnswer('  A   RED Bus!!! ')).toBe('a red bus')
  })

  it('Q&A duplicate is rejected after normalization', () => {
    const state = useWorkspaceStore.getState()
    state.updateQa({
      searched: true,
      results: [
        {
          videoId: 'L21_V001',
          frameId: 4592,
          timestamp: '00:03:12',
          score: 0.94,
          badges: [],
          neighbors: [],
        },
      ],
      selectedVideoId: 'L21_V001',
      selectedFrameId: 4592,
      intervalReady: true,
      verified: true,
      canonical: 'Red Bus',
    })
    state.addQaAnswer()
    state.updateQa({ canonical: '  red bus!!!' })
    state.addQaAnswer()
    expect(useWorkspaceStore.getState().qa.answers).toHaveLength(1)
    expect(useWorkspaceStore.getState().qa.message).toMatch(/Duplicate/)
  })

  it('TRAKE only accepts complete same-video ordered chains', () => {
    const state = useWorkspaceStore.getState()
    state.updateTrake({
      activeChain: { videoId: 'L21_V001', frames: [400, 300, 600], confidence: 0.8 },
    })
    state.verifyChain()
    state.addChain()
    expect(useWorkspaceStore.getState().trake.answers).toHaveLength(0)

    state.updateTrake({
      activeChain: { videoId: 'L21_V001', frames: [300, 400, 600], confidence: 0.8 },
    })
    state.verifyChain()
    state.addChain()
    expect(useWorkspaceStore.getState().trake.answers).toHaveLength(1)
  })

  it('validation detects invalid output', () => {
    expect(validateKis([{ videoId: 'wrong', frameId: 1.5, confidence: 0.2 }]).valid).toBe(false)
    expect(
      validateChain({ videoId: 'L21_V001', frames: [3, 2, 1], confidence: 0.2 }).valid
    ).toBe(false)
  })

  it('CSV contains only official output columns', () => {
    expect(kisCsv([])).toBe('video_id,frame_id')
    expect(qaCsv([])).toBe('video_id,frame_id,answer')
    expect(trakeCsv([])).toBe('video_id,frame_id_1,frame_id_2,frame_id_3')
  })

  it('KIS, Q&A and TRAKE preserve independent state', async () => {
    vi.useRealTimers()
    const user = userEvent.setup()
    render(<App />)
    await user.type(
      screen.getByPlaceholderText(/Enter visual event description/),
      'KIS query'
    )
    await user.click(screen.getByRole('button', { name: 'Q&A' }))
    await user.type(screen.getByPlaceholderText(/Enter event description/), 'QA event')
    await user.click(screen.getByRole('button', { name: 'TRAKE' }))
    await user.click(screen.getByRole('button', { name: 'KIS' }))
    expect(screen.getByPlaceholderText(/Enter visual event description/)).toHaveValue(
      'KIS query'
    )
    await user.click(screen.getByRole('button', { name: 'Q&A' }))
    expect(screen.getByPlaceholderText(/Enter event description/)).toHaveValue('QA event')
  })
})
