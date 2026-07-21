import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useLocation } from 'react-router-dom'
import { Link } from 'react-router-dom'
import { datasetApi, errorMessage } from '../api'
import PageHeader from '../components/PageHeader'
import StatusBadge from '../components/StatusBadge'
import type { Dataset } from '../types'

function bool(value?: boolean) { return value ? 'да' : 'нет' }
function bytes(value?: number) { if (!value) return '—'; return value > 1e6 ? `${(value / 1e6).toFixed(1)} MB` : `${(value / 1e3).toFixed(1)} KB` }

export default function DatasetsPage() {
  const client = useQueryClient()
  const location = useLocation()
  const datasets = useQuery({ queryKey: ['datasets'], queryFn: datasetApi.list, refetchInterval: 2500 })
  const [selectedId, setSelectedId] = useState(() => (location.state as { datasetId?: string } | null)?.datasetId ?? '')
  const [selectedFile, setSelectedFile] = useState('')
  useEffect(() => { if (!selectedId && datasets.data?.[0]) setSelectedId(datasets.data[0].id) }, [selectedId, datasets.data])
  useEffect(() => { setSelectedFile('') }, [selectedId])
  const selected = datasets.data?.find((item) => item.id === selectedId)
  const preview = useQuery({ queryKey: ['preview', selectedId], queryFn: () => datasetApi.preview(selectedId), enabled: Boolean(selectedId && selected?.summary?.stage === 'normalized') })
  const normalize = useMutation({ mutationFn: (payload: { id: string; selectedFile?: string }) => datasetApi.normalize(payload.id, payload.selectedFile ? { selected_file: payload.selectedFile } : {}), onSuccess: () => client.invalidateQueries({ queryKey: ['datasets'] }) })
  const remove = useMutation({ mutationFn: (id: string) => datasetApi.remove(id), onSuccess: () => { setSelectedId(''); void client.invalidateQueries({ queryKey: ['datasets'] }) } })
  return <>
    <PageHeader eyebrow="Canonical layer" title="Datasets" text="Raw-файлы остаются неизменными; normalized-слой приводится к единой месячной схеме." action={<Link className="button" to="/generator">Создать синтетику</Link>} />
    <div className="dataset-layout">
      <section className="panel dataset-list"><div className="section-title"><h2>Наборы</h2><span>{datasets.data?.length ?? 0}</span></div>{datasets.data?.map((dataset) => <button key={dataset.id} className={selectedId === dataset.id ? 'selected' : ''} onClick={() => setSelectedId(dataset.id)}><div><code>{dataset.id}</code><small>{dataset.summary?.source_id ?? dataset.config.source_id ?? 'synthetic'}</small></div><StatusBadge status={dataset.status} /></button>)}</section>
      <section className="panel dataset-detail">{selected ? <>
        <div className="section-title"><div><p className="eyebrow">{selected.summary?.stage ?? selected.config.stage ?? 'raw'}</p><h2>{selected.id}</h2></div><StatusBadge status={selected.status} /></div>
        <div className="metric-grid six"><span><small>Строк</small><b>{(selected.summary?.rows ?? selected.summary?.monthly_rows ?? 0).toLocaleString('ru')}</b></span><span><small>Клиентов</small><b>{selected.summary?.clients ?? selected.summary?.n_clients ?? '—'}</b></span><span><small>Месяцев</small><b>{selected.summary?.months ?? selected.summary?.n_months ?? '—'}</b></span><span><small>Период</small><b>{selected.summary?.first_month ?? '—'} — {selected.summary?.last_month ?? '—'}</b></span><span><small>Debit / Credit</small><b>{bool(selected.summary?.has_debit)} / {bool(selected.summary?.has_credit)}</b></span><span><small>Balance / Target</small><b>{bool(selected.summary?.has_balance)} / {bool(selected.summary?.has_cash_gap_target)}</b></span><span><small>Несколько валют</small><b>{bool(selected.summary?.multiple_currencies)}</b></span><span><small>Пропуски</small><b>{selected.summary?.missing_percent?.toFixed(2) ?? '—'}%</b></span><span><small>Размер</small><b>{bytes(selected.summary?.size_bytes)}</b></span></div>
        {selected.summary?.quality_flags?.length ? <p className="warning">Data quality: {selected.summary.quality_flags.join(', ')}</p> : null}
        {selected.summary?.compatibility && <div className="compatibility"><h3>Совместимость</h3><div className="tag-row"><span className={selected.summary.compatibility.classification_eligible ? 'ok' : ''}>Classification</span><span className={selected.summary.compatibility.forecasting_eligible ? 'ok' : ''}>Forecasting</span><span className={selected.summary.compatibility.proxy_eligible ? 'ok' : ''}>Proxy</span><span className={selected.summary.compatibility.categorization_eligible ? 'ok' : ''}>Categorization</span></div>{selected.summary.compatibility.reasons.map((reason) => <p key={reason}>• {reason}</p>)}</div>}
        {selected.summary?.file_list?.length ? <details open><summary>Raw files ({selected.summary.file_list.length})</summary><div className="file-list">{selected.summary.file_list.map((file) => <label key={file.name}><input type="radio" name="raw-file" checked={selectedFile === file.name} onChange={() => setSelectedFile(file.name)} /><code>{file.name}</code><span>{bytes(file.size_bytes)}</span></label>)}</div><p className="hint">Для IBM AML выберите CSV явно; без выбора адаптер возьмёт самый маленький совместимый файл.</p></details> : null}
        <div className="card-actions">{selected.summary?.stage !== 'normalized' && selected.status === 'completed' && <button className="button" onClick={() => normalize.mutate({ id: selected.id, selectedFile })}>Нормализовать</button>}<button className="button button--danger" disabled={['queued', 'running', 'downloading', 'normalizing'].includes(selected.status)} onClick={() => window.confirm('Удалить dataset?') && remove.mutate(selected.id)}>Удалить</button></div>
        {selected.error && <pre className="error-box">{selected.error}</pre>}
        {selected.summary?.mapping && <details><summary>Mapping source → canonical</summary><pre>{JSON.stringify(selected.summary.mapping, null, 2)}</pre></details>}
        {preview.data?.items.length ? <div className="preview"><h3>Preview</h3><div className="table-wrap"><table><thead><tr>{preview.data.columns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{preview.data.items.slice(0, 20).map((row, index) => <tr key={index}>{preview.data.columns.map((column) => <td key={column}>{String(row[column] ?? '—')}</td>)}</tr>)}</tbody></table></div></div> : null}
      </> : <div className="empty">Выберите dataset</div>}</section>
    </div>
    {(normalize.error || remove.error) && <p className="error-message">{errorMessage(normalize.error || remove.error)}</p>}
  </>
}
