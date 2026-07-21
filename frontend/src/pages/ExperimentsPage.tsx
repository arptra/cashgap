import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { errorMessage, experimentApi } from '../api'
import PageHeader from '../components/PageHeader'
import StatusBadge from '../components/StatusBadge'

function metric(run: {
  task: string
  metrics?: { pr_auc?: number | null; wape?: number | null; f1_macro?: number | null }
}) {
  if (!run.metrics) return '—'
  if (run.task === 'cash_gap_classification') return `PR-AUC ${run.metrics.pr_auc?.toFixed(3) ?? '—'}`
  if (run.task === 'flow_forecasting') return `WAPE ${run.metrics.wape != null ? (run.metrics.wape * 100).toFixed(1) + '%' : '—'}`
  return `F1 macro ${run.metrics.f1_macro?.toFixed(3) ?? '—'}`
}

export default function ExperimentsPage() {
  const navigate = useNavigate()
  const client = useQueryClient()
  const experiments = useQuery({ queryKey: ['experiments'], queryFn: experimentApi.list, refetchInterval: 2500 })
  const [selected, setSelected] = useState<string[]>(() => JSON.parse(localStorage.getItem('cashgap_compare_runs') || '[]'))
  const remove = useMutation({ mutationFn: experimentApi.remove, onSuccess: () => client.invalidateQueries({ queryKey: ['experiments'] }) })
  const toggle = (id: string) => setSelected((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id])
  const compare = () => { localStorage.setItem('cashgap_compare_runs', JSON.stringify(selected)); navigate('/comparison') }
  return <>
    <PageHeader eyebrow="Experiment registry" title="Эксперименты" text="Classification, forecasting и categorization хранятся раздельно и сравниваются по своим метрикам." action={<button className="button button--secondary" disabled={!selected.length} onClick={compare}>Сравнить ({selected.length})</button>} />
    <section className="panel table-wrap"><table><thead><tr><th></th><th>Run / дата</th><th>Dataset</th><th>Задача</th><th>Модель</th><th>Статус</th><th>Главная метрика</th><th>Время</th><th></th></tr></thead><tbody>{experiments.data?.map((run) => <tr key={run.id}><td><input type="checkbox" disabled={run.status !== 'completed'} checked={selected.includes(run.id)} onChange={() => toggle(run.id)} /></td><td><code>{run.id}</code><small>{new Date(run.created_at).toLocaleString('ru')}</small></td><td><code>{run.dataset_id}</code></td><td>{run.task}</td><td>{run.model_name}</td><td><StatusBadge status={run.status} />{run.error && <details><summary>Ошибка</summary><pre>{run.error}</pre></details>}</td><td><b>{metric(run)}</b></td><td>{run.duration_seconds ? `${run.duration_seconds.toFixed(2)} с` : '—'}</td><td><button className="icon-button" disabled={['queued', 'running'].includes(run.status)} onClick={() => window.confirm('Удалить run?') && remove.mutate(run.id)}>×</button></td></tr>)}</tbody></table>{!experiments.data?.length && <div className="empty">Экспериментов пока нет.</div>}</section>
    {remove.error && <p className="error-message">{errorMessage(remove.error)}</p>}
  </>
}
