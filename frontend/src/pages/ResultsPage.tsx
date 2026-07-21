import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import { experimentApi } from '../api'
import PageHeader from '../components/PageHeader'

export default function ResultsPage() {
  const experiments = useQuery({ queryKey: ['experiments'], queryFn: experimentApi.list, refetchInterval: 3000 })
  const completed = experiments.data?.filter((run) => run.status === 'completed') ?? []
  const [search] = useSearchParams()
  const [runId, setRunId] = useState(search.get('run') ?? '')
  useEffect(() => { if (!runId && completed[0]) setRunId(completed[0].id) }, [runId, completed])
  const selected = completed.find((run) => run.id === runId)
  const predictions = useQuery({ queryKey: ['predictions', runId], queryFn: () => experimentApi.predictions(runId), enabled: Boolean(runId) })
  const columns = predictions.data?.items[0] ? Object.keys(predictions.data.items[0]) : []
  return <>
    <PageHeader eyebrow="Saved artifacts" title="Результаты" text="Classification показывает клиентский риск, forecasting — actual/forecast/bounds, categorization — предсказанный класс." action={runId ? <a className="button button--secondary" href={`/api/experiments/${runId}/predictions.csv`}>Выгрузить CSV</a> : undefined} />
    <section className="panel result-controls"><label>Experiment<select value={runId} onChange={(event) => setRunId(event.target.value)}>{completed.map((run) => <option key={run.id} value={run.id}>{run.id} · {run.task} · {run.model_name}</option>)}</select></label><div>{selected?.task ?? '—'}<small>{predictions.data ? `Показано ${predictions.data.items.length} из ${predictions.data.total}` : ''}</small></div></section>
    <section className="panel table-wrap"><table><thead><tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{predictions.data?.items.map((row, index) => <tr key={index}>{columns.map((column) => <td key={column}>{typeof row[column] === 'number' ? Number(row[column]).toFixed(column.includes('score') || column.includes('forecast') ? 4 : 2) : String(row[column] ?? '—')}</td>)}</tr>)}</tbody></table>{!completed.length && <div className="empty">Нет завершённых runs.</div>}</section>
  </>
}
