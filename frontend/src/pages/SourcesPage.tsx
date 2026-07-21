import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { errorMessage, jobApi, sourceApi } from '../api'
import PageHeader from '../components/PageHeader'
import StatusBadge from '../components/StatusBadge'
import type { Source } from '../types'

function SourceCard({ source }: { source: Source }) {
  const client = useQueryClient()
  const navigate = useNavigate()
  const [accepted, setAccepted] = useState(false)
  const [pathOrUrl, setPathOrUrl] = useState('')
  const [includeMerchants, setIncludeMerchants] = useState(false)
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: jobApi.list, refetchInterval: 2000 })
  const latest = jobs.data?.find((job) => job.source_id === source.id)
  const options: Record<string, unknown> = {
    base_date: '2023-01-01',
    include_merchants_as_clients: includeMerchants,
  }
  if (source.provider === 'local') options.path = pathOrUrl
  if (source.provider === 'http') options.url = pathOrUrl
  const check = useMutation({
    mutationFn: () => sourceApi.check(source.id, options),
    onSuccess: () => { void client.invalidateQueries({ queryKey: ['jobs'] }); void client.invalidateQueries({ queryKey: ['sources'] }) },
  })
  const download = useMutation({
    mutationFn: () => sourceApi.download(source.id, options),
    onSuccess: () => { void client.invalidateQueries({ queryKey: ['jobs'] }); void client.invalidateQueries({ queryKey: ['datasets'] }) },
  })
  const error = check.error || download.error
  return <article className="source-card">
    <div className="card-top"><div><small>{source.provider}</small><h2>{source.title}</h2></div><StatusBadge status={latest?.status ?? source.access_status} /></div>
    <code>{source.remote_id ?? source.id}</code>
    <div className="tag-row">{source.supported_tasks.map((task) => <span key={task}>{task}</span>)}</div>
    <dl className="source-meta"><div><dt>Adapter</dt><dd>{source.adapter}</dd></div><div><dt>Cash-gap target</dt><dd>{source.cash_gap_target ? 'да' : 'нет'}</dd></div><div><dt>Auth</dt><dd>{source.requires_auth ? 'требуется' : 'публичный'}</dd></div></dl>
    {source.license_note && <p className="license">Лицензия: {source.license_note}</p>}
    {source.usage_note && <p className="hint">{source.usage_note}</p>}
    {source.limitations?.length ? <p className="warning">Ограничения: {source.limitations.join(', ')}</p> : null}
    {(source.provider === 'local' || source.provider === 'http') && <label>{source.provider === 'local' ? 'Локальный путь' : 'HTTP / GitHub URL'}<input value={pathOrUrl} onChange={(event) => setPathOrUrl(event.target.value)} placeholder={source.provider === 'local' ? '/data/transactions.csv' : 'https://.../data.csv'} /></label>}
    {source.adapter === 'paysim' && <label className="check-line"><input type="checkbox" checked={includeMerchants} onChange={(event) => setIncludeMerchants(event.target.checked)} />Учитывать merchant M* как клиентов</label>}
    <label className="check-line terms"><input type="checkbox" checked={accepted} onChange={(event) => setAccepted(event.target.checked)} />Я проверил условия использования и правила источника</label>
    {latest && <div className="job-progress"><span style={{ width: `${latest.progress * 100}%` }} /><small>{latest.message}</small></div>}
    <div className="card-actions"><button className="button button--secondary" disabled={check.isPending} onClick={() => check.mutate()}>Проверить доступ</button><button className="button" disabled={!accepted || download.isPending || ((source.provider === 'local' || source.provider === 'http') && !pathOrUrl)} onClick={() => download.mutate()}>Скачать</button>{latest?.dataset_id && latest.status === 'completed' && <button className="text-button" onClick={() => navigate('/datasets', { state: { datasetId: latest.dataset_id } })}>Открыть dataset</button>}{latest && !['completed', 'failed', 'cancelled'].includes(latest.status) && <button className="text-button" onClick={() => jobApi.cancel(latest.id).then(() => client.invalidateQueries({ queryKey: ['jobs'] }))}>Отменить</button>}</div>
    {error && <p className="error-message">{errorMessage(error)}</p>}
    {latest?.error && <details><summary>Ошибка job</summary><pre>{latest.error}</pre></details>}
  </article>
}

export default function SourcesPage() {
  const sources = useQuery({ queryKey: ['sources'], queryFn: sourceApi.list, refetchInterval: 5000 })
  return <>
    <PageHeader eyebrow="Data ingestion" title="Источники данных" text="Загрузка начинается только после проверки условий. Fraud и AML labels никогда не становятся cash-gap target." />
    {sources.isLoading && <div className="panel empty">Загрузка реестра…</div>}
    {sources.error && <div className="panel error-message">{errorMessage(sources.error)}</div>}
    <div className="source-grid">{sources.data?.map((source) => <SourceCard source={source} key={source.id} />)}</div>
  </>
}
