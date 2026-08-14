import { useEffect, useMemo, useState } from 'react'
import { candidates, evidenceByTab, qaHypotheses } from './mockData'
import { useWorkspaceStore } from './store'
import type { Candidate, EvidenceTab, Task } from './types'
import { downloadCsv, kisCsv, qaCsv, trakeCsv, validateChain, validateKis, validateQa } from './utils'

const tabs: Task[] = ['KIS', 'Q&A', 'TRAKE']
const evidenceTabs: EvidenceTab[] = ['Visual', 'OCR', 'ASR', 'Object', 'Metadata']
const workflows: Record<Task, string[]> = {
  KIS: ['Query', 'Search', 'Inspect', 'Open video', 'Refine frame', 'Verify evidence', 'Add answer'],
  'Q&A': ['Query', 'Retrieve candidates', 'Localize evidence', 'Select frame', 'Inspect evidence', 'Select answer', 'Verify consistency', 'Add answer'],
  TRAKE: ['Decompose', 'Retrieve events', 'Rank videos', 'Align timeline', 'Edit chain', 'Verify', 'Add chain'],
}

function Header() {
  const { activeTask, setTask, mode, setMode, historyOpen, toggleHistory, history } = useWorkspaceStore()
  const queryId = activeTask === 'KIS' ? 'Q-1042' : activeTask === 'Q&A' ? 'QA-501' : 'TR-208'
  return <>
    <header className="global-header">
      <h1>AIC 2026 Retrieval Workspace</h1>
      <nav className="task-tabs" aria-label="Task selector">
        {tabs.map((tab) => <button key={tab} className={activeTask === tab ? 'active' : ''} onClick={() => setTask(tab)}>{tab}</button>)}
      </nav>
      <div className="header-controls">
        <label>Query ID: <span className="field-like">{queryId}</span></label>
        <span className="field-like">Dataset Batch: <b>B-04</b></span>
        <label className="sr-only" htmlFor="mode">Mode</label>
        <select id="mode" value={mode} onChange={(event) => setMode(event.target.value as 'Interactive' | 'Automatic')}>
          <option>Interactive</option><option>Automatic</option>
        </select>
        <button onClick={toggleHistory}>Search History</button>
      </div>
    </header>
    {historyOpen && <aside className="history-popover" aria-label="Search history">
      <strong>Recent searches</strong>
      {history.length ? history.map((item, index) => <div key={`${item}-${index}`}>{item}</div>) : <p>No searches yet.</p>}
    </aside>}
  </>
}

function Stepper() {
  const task = useWorkspaceStore((s) => s.activeTask)
  return <div className="stepper">{workflows[task].map((step, index) => <span key={step}><b>{index + 1}.</b> {step}{index < workflows[task].length - 1 && <i>→</i>}</span>)}</div>
}

function Panel({ title, children, className = '' }: { title: string; children: React.ReactNode; className?: string }) {
  return <section className={`panel ${className}`}><div className="panel-title">{title}</div>{children}</section>
}

function Placeholder({ label = 'Video frame', compact = false }: { label?: string; compact?: boolean }) {
  return <div className={`placeholder ${compact ? 'compact' : ''}`} aria-label={label}><span>{label}</span></div>
}

function Badge({ children }: { children: React.ReactNode }) { return <span className="badge">{children}</span> }

function EvidenceTabs({ value, onChange }: { value: EvidenceTab; onChange: (tab: EvidenceTab) => void }) {
  return <div className="evidence-block">
    <div className="evidence-tabs" role="tablist">{evidenceTabs.map((tab) => <button role="tab" aria-selected={tab === value} className={tab === value ? 'active' : ''} key={tab} onClick={() => onChange(tab)}>{tab}</button>)}</div>
    <div className="evidence-content" data-testid="evidence-content">
      <b>{value} evidence</b><p>{evidenceByTab[value]}</p>
      <dl><dt>Frame confidence</dt><dd>{value === 'Visual' ? '0.94' : value === 'OCR' ? '0.82' : '0.76'}</dd><dt>Source status</dt><dd>Available</dd></dl>
    </div>
  </div>
}

function CandidateCard({ candidate, selected, onSelect }: { candidate: Candidate; selected: boolean; onSelect: () => void }) {
  return <button className={`candidate-card ${selected ? 'selected' : ''}`} onClick={onSelect} data-testid={`candidate-${candidate.videoId}`}>
    {selected && <span className="selected-label">SELECTED</span>}
    <Placeholder label={`Keyframe ${candidate.videoId}`} />
    <div className="candidate-meta"><strong>{candidate.videoId}</strong><span>Score: {candidate.score.toFixed(2)}</span></div>
    <div className="mono">Actual Frame ID: {candidate.frameId}</div>
    <div className="card-footer"><span>{candidate.badges.map((badge) => <Badge key={badge}>{badge}</Badge>)}</span><b>{candidate.timestamp}</b></div>
  </button>
}

function KisWorkspace() {
  const { kis, updateKis, searchKis, selectKisCandidate, addKisAnswer, resetKis } = useWorkspaceStore()
  const selected = kis.results.find((candidate) => candidate.videoId === kis.selectedVideoId)
  const exactFrame = selected?.neighbors.find((frame) => frame.id === kis.selectedFrameId)
  const runSearch = () => { updateKis({ loading: true, message: undefined }); window.setTimeout(searchKis, 180) }
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.ctrlKey && event.key === 'Enter') addKisAnswer()
      if (selected && event.key === 'ArrowLeft') {
        const index = selected.neighbors.findIndex((frame) => frame.id === kis.selectedFrameId)
        if (index > 0) updateKis({ selectedFrameId: selected.neighbors[index - 1].id })
      }
      if (selected && event.key === 'ArrowRight') {
        const index = selected.neighbors.findIndex((frame) => frame.id === kis.selectedFrameId)
        if (index < selected.neighbors.length - 1) updateKis({ selectedFrameId: selected.neighbors[index + 1].id })
      }
    }
    window.addEventListener('keydown', handler); return () => window.removeEventListener('keydown', handler)
  }, [addKisAnswer, kis.selectedFrameId, selected, updateKis])
  return <div className="workspace kis-layout">
    <Panel title="Textual KIS" className="query-panel">
      <div className="panel-body form-stack">
        <label>Describe the event<textarea value={kis.query} onChange={(event) => updateKis({ query: event.target.value })} placeholder="A person boards a red city bus" /></label>
        <small>Describe visible actions, people, objects, text, or speech.</small>
        <fieldset><legend>Query Variants</legend><div className="chips">{['Action focused', 'Object focused', 'OCR focused'].map((variant) => <button key={variant} className={kis.variants.includes(variant) ? 'pressed' : ''} onClick={() => updateKis({ variants: kis.variants.includes(variant) ? kis.variants.filter((item) => item !== variant) : [...kis.variants, variant] })}>{variant}</button>)}</div></fieldset>
        <fieldset><legend>Evidence Source Filters</legend>{['Visual', 'OCR', 'ASR', 'Object', 'Metadata'].map((filter) => <label className="check" key={filter}><input type="checkbox" checked={kis.filters.includes(filter)} onChange={() => updateKis({ filters: kis.filters.includes(filter) ? kis.filters.filter((item) => item !== filter) : [...kis.filters, filter] })} />{filter}</label>)}</fieldset>
      </div>
      <div className="panel-actions"><button className="primary" onClick={runSearch}>{kis.loading ? 'Searching…' : 'Search'}</button><button onClick={resetKis}>Reset</button></div>
    </Panel>
    <Panel title="Results Grouped by Video" className="results-panel">
      <div className="results-toolbar"><span>{kis.searched ? `${kis.results.length} candidate videos` : 'Run a search to retrieve candidates'}</span><label>Group by: <select><option>Video</option><option>Shot</option></select></label><label>Sort by: <select><option>Combined Score</option><option>Visual Score</option></select></label></div>
      <div className="result-grid">
        {kis.loading && <div className="state-box">Loading multimodal candidates…</div>}
        {!kis.loading && !kis.searched && <div className="state-box">No results yet. Enter an event description and search.</div>}
        {!kis.loading && kis.searched && kis.results.map((candidate) => <CandidateCard key={candidate.videoId} candidate={candidate} selected={kis.selectedVideoId === candidate.videoId} onSelect={() => selectKisCandidate(candidate)} />)}
      </div>
    </Panel>
    <Panel title={selected ? selected.videoId : 'Frame Inspector'} className="inspector-panel">
      {selected ? <div className="inspector-scroll">
        <div className="inspector-summary"><b>{exactFrame?.timestamp ?? selected.timestamp}</b><span>Actual Frame ID: <b data-testid="inspector-frame-id">{kis.selectedFrameId}</b></span></div>
        <Placeholder label={`Video ${selected.videoId}`} />
        <div className="player-controls"><button onClick={() => shiftFrame(selected, kis.selectedFrameId, -1, updateKis)}>−5 SEC</button><button className="primary">▶ PLAY VIDEO</button><button onClick={() => shiftFrame(selected, kis.selectedFrameId, 1, updateKis)}>+5 SEC</button></div>
        <button className="wide">Search in this video</button>
        <div className="subheading">Nearby Frames</div>
        <div className="filmstrip">{selected.neighbors.map((frame, index) => <button key={frame.id} className={frame.id === kis.selectedFrameId ? 'current' : ''} onClick={() => updateKis({ selectedFrameId: frame.id })} data-testid={`neighbor-${frame.id}`}><small>{index < 2 ? 'BEFORE' : index === 2 ? 'CURRENT' : 'AFTER'}</small><Placeholder label={`Frame ${frame.id}`} compact /><b>{frame.id}</b><span>{frame.timestamp}</span></button>)}</div>
        <button className="primary wide add-frame" onClick={addKisAnswer}>Select Exact Frame & Add to List</button>
        {kis.message && <div className={kis.message.includes('rejected') ? 'message error' : 'message'}>{kis.message}</div>}
        <EvidenceTabs value={kis.evidenceTab} onChange={(evidenceTab) => updateKis({ evidenceTab })} />
      </div> : <div className="state-box">Select a result to inspect its exact frames and evidence.</div>}
    </Panel>
  </div>
}

function shiftFrame(candidate: Candidate, current: number | undefined, amount: number, update: (patch: { selectedFrameId: number }) => void) {
  const currentIndex = Math.max(0, candidate.neighbors.findIndex((frame) => frame.id === current))
  const nextIndex = Math.min(candidate.neighbors.length - 1, Math.max(0, currentIndex + amount))
  update({ selectedFrameId: candidate.neighbors[nextIndex].id })
}

function QaWorkspace() {
  const { qa, updateQa, searchQa, selectQaCandidate, addQaAnswer, resetQa } = useWorkspaceStore()
  const selected = qa.results.find((candidate) => candidate.videoId === qa.selectedVideoId)
  const canAdd = !!selected && qa.selectedFrameId !== undefined && qa.intervalReady && qa.verified && !!qa.canonical.trim()
  const runSearch = () => { updateQa({ loading: true, message: undefined }); window.setTimeout(searchQa, 180) }
  return <div className="workspace qa-layout">
    <Panel title="Video Q&A" className="qa-left">
      <div className="panel-body form-stack">
        <label>EVENT DESCRIPTION<textarea value={qa.event} onChange={(event) => updateQa({ event: event.target.value })} placeholder='e.g. “A man drops a package”' /></label>
        <label>QUESTION<textarea value={qa.question} onChange={(event) => updateQa({ question: event.target.value })} placeholder='e.g. “What color was the package?”' /></label>
        <div className="two-cols"><label>TEMPORAL RELATION<select value={qa.temporal} onChange={(event) => updateQa({ temporal: event.target.value })}>{['Before', 'During', 'After'].map((value) => <option key={value}>{value}</option>)}</select></label><label>SUGGESTED ANSWER TYPE<select value={qa.answerType} onChange={(event) => updateQa({ answerType: event.target.value })}>{['Automatic', 'Action', 'Object', 'Person', 'Count', 'Color', 'Location', 'OCR Text', 'ASR Speech', 'Yes / No'].map((value) => <option key={value}>{value}</option>)}</select></label></div>
      </div>
      <div className="panel-actions"><button className="primary" onClick={runSearch}>{qa.loading ? 'Retrieving…' : 'Retrieve Evidence'}</button><button onClick={resetQa}>Reset</button></div>
      <div className="subheading">Candidate Videos</div><div className="qa-candidates">
        {!qa.searched && <div className="state-box">No evidence candidates yet.</div>}
        {qa.results.map((candidate) => <button key={candidate.videoId} className={`qa-candidate ${candidate.videoId === qa.selectedVideoId ? 'selected' : ''}`} onClick={() => selectQaCandidate(candidate)}><span><b>{candidate.videoId}</b><small>Anchor: {candidate.timestamp}</small></span><span><b>Score: {candidate.score.toFixed(2)}</b><small>Frame: {candidate.frameId}</small></span><span>{candidate.badges.map((badge) => <Badge key={badge}>{badge}</Badge>)}</span></button>)}
      </div>
    </Panel>
    <Panel title="Evidence Workspace" className="qa-evidence">
      {selected ? <div className="qa-evidence-scroll">
        <div className="candidate-summary"><b>{selected.videoId}</b><span>Anchor Event: {selected.timestamp}</span><span>Actual Frame ID: <b data-testid="qa-frame-id">{qa.selectedFrameId}</b></span></div>
        <Placeholder label="Representative semantic frame" />
        <div className="timeline"><div className="before">BEFORE</div><div className="during">DURING · ANCHOR</div><div className="after">AFTER</div><i style={{ left: '49%' }} /></div>
        <div className="interval-labels"><span>Evidence interval starts −00:04</span><b>Representative frame {qa.selectedFrameId}</b><span>ends +00:06</span></div>
        <div className="semantic-frames">{[-6, 0, 6].map((delta, index) => <button key={delta} onClick={() => updateQa({ selectedFrameId: selected.frameId + delta })}><small>{index === 0 ? 'PREVIOUS' : index === 1 ? 'REPRESENTATIVE' : 'NEXT'}</small><Placeholder label={`Frame ${selected.frameId + delta}`} compact /><b>{selected.frameId + delta}</b></button>)}</div>
        <EvidenceTabs value={qa.evidenceTab} onChange={(evidenceTab) => updateQa({ evidenceTab })} />
      </div> : <div className="state-box">Retrieve evidence and select a candidate video. The workspace will show the anchor event, interval, and representative frame.</div>}
    </Panel>
    <Panel title="Answer Panel" className="answer-panel">
      <div className="panel-body answer-content"><div className="subheading plain">Answer Hypotheses</div>{qaHypotheses.map((hypothesis) => <button className="hypothesis" key={hypothesis.answer} disabled={!selected} onClick={() => updateQa({ canonical: hypothesis.answer })}><b>{hypothesis.answer}</b><span>Conf: {hypothesis.confidence.toFixed(2)}</span></button>)}
        <hr /><label>CANONICAL ANSWER<input value={qa.canonical} onChange={(event) => updateQa({ canonical: event.target.value })} placeholder="Enter grounded answer" /></label><hr />
        <div className="status-row"><span>Consistency Status:</span><b className={qa.verified ? 'status valid' : 'status'}>{qa.verified ? 'Verified' : 'Not evaluated'}</b></div>
        <div className="status-row"><span>Evidence Confidence:</span><b>{qa.verified ? selected?.score.toFixed(2) : '—'}</b></div>
        {qa.message && <div className={qa.message.includes('rejected') ? 'message error' : 'message'}>{qa.message}</div>}
      </div>
      <div className="answer-actions"><button className="primary wide" disabled={!canAdd} onClick={addQaAnswer}>Add Answer</button><small>Requires selected candidate, localized interval, representative frame, canonical answer, and verification.</small></div>
    </Panel>
  </div>
}

function TrakeWorkspace() {
  const { trake, updateTrake, searchTrake, selectChain, verifyChain, addChain, resetTrake } = useWorkspaceStore()
  const validation = trake.activeChain ? validateChain(trake.activeChain) : undefined
  const runSearch = () => { updateTrake({ loading: true, message: undefined }); window.setTimeout(searchTrake, 180) }
  return <div className="workspace trake-layout">
    <Panel title="Multi-event Query" className="trake-query">
      <div className="panel-body form-stack"><label>ORIGINAL QUERY<textarea value={trake.query} onChange={(event) => updateTrake({ query: event.target.value })} /></label><div className="event-cards">{trake.events.map((eventText, index) => <label key={index}><b>E{index + 1}</b><textarea value={eventText} onChange={(event) => updateTrake({ events: trake.events.map((item, i) => i === index ? event.target.value : item) })} /></label>)}</div><div className="constraint">Temporal constraint: <b>E1 &lt; E2 &lt; E3</b></div></div>
      <div className="panel-actions"><button className="primary" onClick={runSearch}>{trake.loading ? 'Retrieving…' : 'Retrieve All Events'}</button><button onClick={resetTrake}>Reset</button></div>
    </Panel>
    <Panel title="Video Ranking" className="trake-ranking"><div className="ranking-list">{!trake.searched && <div className="state-box">Retrieve all events to rank same-video coverage.</div>}{candidates.slice(0, 4).map((candidate, index) => trake.searched && <button key={candidate.videoId} className={candidate.videoId === trake.selectedVideoId ? 'selected' : ''} onClick={() => { const chain = trake.chains.find((item) => item.videoId === candidate.videoId); if (chain) selectChain(chain) }}><b>#{index + 1} {candidate.videoId}</b><span>Coverage {index < 2 ? '3/3' : '2/3'}</span><span>Retrieval {candidate.score.toFixed(2)}</span><Badge>{index < 2 ? 'ORDERED' : 'PARTIAL'}</Badge></button>)}</div></Panel>
    <Panel title="Multi-event Timeline & Candidate Chains" className="trake-main">
      {trake.searched ? <div className="trake-scroll">
        <div className="timeline trake-timeline"><span>E1</span><span>E2</span><span>E3</span>{trake.activeChain && <small>{trake.activeChain.frames[0]} <i>gap {trake.activeChain.frames[1] - trake.activeChain.frames[0]}</i> {trake.activeChain.frames[1]} <i>gap {trake.activeChain.frames[2] - trake.activeChain.frames[1]}</i> {trake.activeChain.frames[2]}</small>}</div>
        <div className="event-columns">{trake.events.map((eventText, eventIndex) => <div key={eventText}><h3>E{eventIndex + 1} Candidate</h3><Placeholder label={`E${eventIndex + 1} evidence`} compact /><p>{eventText}</p>{trake.activeChain && <label>Actual Frame ID<input type="number" value={trake.activeChain.frames[eventIndex]} onChange={(event) => { const frames = [...trake.activeChain!.frames] as [number, number, number]; frames[eventIndex] = Number(event.target.value); updateTrake({ activeChain: { ...trake.activeChain!, frames }, verified: false }) }} /></label>}<span>Confidence: {trake.activeChain ? (trake.activeChain.confidence - eventIndex * .03).toFixed(2) : '—'}</span><Badge>VISUAL + OBJECT</Badge></div>)}</div>
        <div className="subheading">Top-N Chains</div><div className="chain-list">{trake.chains.map((chain, index) => <button key={`${chain.videoId}-${chain.frames.join('-')}`} className={trake.activeChain?.videoId === chain.videoId && trake.activeChain.frames[0] === chain.frames[0] ? 'selected' : ''} onClick={() => selectChain(chain)}><b>Chain {index + 1}</b><span>{chain.videoId}</span><span>{chain.frames.join(' → ')}</span><span>Conf. {chain.confidence.toFixed(2)}</span><u>Edit</u></button>)}</div>
      </div> : <div className="state-box">No aligned chains yet. Retrieve all events to generate same-video candidates.</div>}
    </Panel>
    <Panel title="Chain Validation" className="chain-validation"><div className="panel-body"><ValidationRow label="Same Video" ok={validation?.sameVideo} /><ValidationRow label="Complete Events" ok={validation?.complete} /><ValidationRow label="Correct Order" ok={validation?.order} /><ValidationRow label="Gap Constraints" ok={validation?.gap} /><ValidationRow label="Evidence Consistency" ok={trake.verified} />{trake.message && <div className={trake.message.includes('rejected') || trake.message.includes('must') ? 'message error' : 'message'}>{trake.message}</div>}</div><div className="answer-actions"><button className="wide" disabled={!trake.activeChain} onClick={verifyChain}>Verify Chain</button><button className="primary wide" disabled={!trake.verified} onClick={addChain}>Add Chain</button></div></Panel>
  </div>
}

function ValidationRow({ label, ok }: { label: string; ok?: boolean }) { return <div className="validation-row"><span>{label}</span><b className={ok ? 'pass' : ''}>{ok ? 'PASS' : '—'}</b></div> }

function AnswerDrawer() {
  const { activeTask, kis, qa, trake, removeKisAnswer, removeQaAnswer, removeChain } = useWorkspaceStore()
  const [summaries, setSummaries] = useState<Record<Task, string>>({ KIS: 'Not validated', 'Q&A': 'Not validated', TRAKE: 'Not validated' })
  const summary = summaries[activeTask]
  const setSummary = (value: string) => setSummaries((current) => ({ ...current, [activeTask]: value }))
  const count = activeTask === 'KIS' ? kis.answers.length : activeTask === 'Q&A' ? qa.answers.length : trake.answers.length
  const validate = () => {
    if (activeTask === 'KIS') { const result = validateKis(kis.answers); setSummary(result.valid ? 'Validation passed: KIS format valid, no duplicates.' : result.errors[0]) }
    else if (activeTask === 'Q&A') { const result = validateQa(qa.answers); setSummary(result.valid ? 'Validation passed: Q&A format valid, no duplicates.' : result.errors[0]) }
    else { const invalid = trake.answers.find((row) => !validateChain(row).valid); setSummary(invalid ? 'Validation failed: invalid chain output.' : 'Validation passed: TRAKE chains valid, no duplicates.') }
  }
  const exportRows = () => {
    if (activeTask === 'KIS' && kis.answers.length) downloadCsv('kis_answers.csv', kisCsv(kis.answers))
    if (activeTask === 'Q&A' && qa.answers.length) downloadCsv('qa_answers.csv', qaCsv(qa.answers))
    if (activeTask === 'TRAKE' && trake.answers.length) downloadCsv('trake_answers.csv', trakeCsv(trake.answers))
  }
  return <section className="answer-drawer"><div className="drawer-head"><h2>{activeTask === 'KIS' ? 'KIS Ranked Answer List' : activeTask === 'Q&A' ? 'Ranked Q&A Answer List' : 'TRAKE Ranked Chain List'}</h2><span className={summary.startsWith('Validation passed') ? 'validation-summary valid' : 'validation-summary'}>{summary}</span><b>{count} / 100 answers</b></div><div className="drawer-grid"><div className="table-scroll"><table><thead><tr><th>Rank</th><th>Video ID</th>{activeTask === 'TRAKE' ? <><th>Frame ID 1</th><th>Frame ID 2</th><th>Frame ID 3</th></> : <th>Actual Frame ID</th>}{activeTask === 'Q&A' && <th>Canonical Answer</th>}<th>Confidence</th>{activeTask === 'Q&A' && <th>Validation</th>}<th>Action</th></tr></thead><tbody>
    {activeTask === 'KIS' && kis.answers.map((row, index) => <tr key={`${row.videoId}-${row.frameId}`}><td>{index + 1}</td><td>{row.videoId}</td><td>{row.frameId}</td><td>{row.confidence.toFixed(2)}</td><td><button className="remove" onClick={() => removeKisAnswer(index)}>REMOVE</button></td></tr>)}
    {activeTask === 'Q&A' && qa.answers.map((row, index) => <tr key={`${row.videoId}-${row.frameId}-${row.answer}`}><td>{index + 1}</td><td>{row.videoId}</td><td>{row.frameId}</td><td>{row.answer}</td><td>{row.confidence.toFixed(2)}</td><td><span className="status valid">{row.validation}</span></td><td><button className="remove" onClick={() => removeQaAnswer(index)}>REMOVE</button></td></tr>)}
    {activeTask === 'TRAKE' && trake.answers.map((row, index) => <tr key={`${row.videoId}-${row.frames.join('-')}`}><td>{index + 1}</td><td>{row.videoId}</td>{row.frames.map((frame) => <td key={frame}>{frame}</td>)}<td>{row.confidence.toFixed(2)}</td><td><button className="remove" onClick={() => removeChain(index)}>REMOVE</button></td></tr>)}
    {count === 0 && <tr><td className="empty-row" colSpan={activeTask === 'TRAKE' ? 7 : activeTask === 'Q&A' ? 7 : 5}>No ranked answers. Add a verified {activeTask === 'TRAKE' ? 'chain' : 'frame'} from the workspace.</td></tr>}
  </tbody></table></div><div className="drawer-actions"><button className="primary" onClick={validate}>Validate</button><button disabled={count === 0} onClick={exportRows}>Export CSV</button><small>Official columns only</small></div></div></section>
}

export default function App() {
  const activeTask = useWorkspaceStore((s) => s.activeTask)
  const workspace = useMemo(() => activeTask === 'KIS' ? <KisWorkspace /> : activeTask === 'Q&A' ? <QaWorkspace /> : <TrakeWorkspace />, [activeTask])
  return <main className="app-shell"><Header /><Stepper />{workspace}<AnswerDrawer /></main>
}
