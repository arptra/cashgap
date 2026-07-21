import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link, useSearchParams } from 'react-router-dom'
import { benchmarkApi, errorMessage } from '../api'
import PageHeader from '../components/PageHeader'
import StatusBadge from '../components/StatusBadge'

function value(number?: number | null, percent = false) {
  if (number == null) return '—'
  return percent ? `${(number * 100).toFixed(1)}%` : number.toFixed(4)
}

export default function ComparisonPage() {
  const [search, setSearch] = useSearchParams()
  const benchmarks = useQuery({ queryKey: ['benchmarks'], queryFn: benchmarkApi.list, refetchInterval: 2500 })
  const [benchmarkId, setBenchmarkId] = useState(search.get('benchmark') ?? '')
  useEffect(() => { if (!benchmarkId && benchmarks.data?.[0]) { setBenchmarkId(benchmarks.data[0].id); setSearch({ benchmark: benchmarks.data[0].id }, { replace: true }) } }, [benchmarkId, benchmarks.data, setSearch])
  const comparison = useQuery({ queryKey: ['benchmark-comparison', benchmarkId], queryFn: () => benchmarkApi.comparison(benchmarkId), enabled: Boolean(benchmarkId), refetchInterval: 2000 })
  const contract = comparison.data?.comparison_contract
  const classification = contract?.task === 'cash_gap_classification'
  return <>
        <PageHeader eyebrow="Same data · same horizon · same test" title="Сравнение моделей" text="Forecasting и classification никогда не смешиваются: каждая таблица привязана к одному benchmark contract." action={<Link className="button" to="/run">Новый benchmark</Link>} />
    <section className="panel comparison-contract"><label>Benchmark<select value={benchmarkId} onChange={(event) => { setBenchmarkId(event.target.value); setSearch({ benchmark: event.target.value }) }}><option value="">Выберите</option>{benchmarks.data?.map((benchmark) => <option key={benchmark.id} value={benchmark.id}>{benchmark.id} · {benchmark.task} · {benchmark.status}</option>)}</select></label>{contract && <div className="contract-row"><span><small>Dataset</small><b>{contract.dataset_id}</b></span><span><small>Target</small><b>{contract.target}</b></span><span><small>Horizon</small><b>{contract.horizon}</b></span><span><small>Уровень</small><b>{contract.series_level}</b></span><StatusBadge status={comparison.data?.benchmark.status ?? 'queued'} /></div>}</section>
    {comparison.error && <p className="error-message">{errorMessage(comparison.error)}</p>}
        <section className="panel table-wrap"><table><thead><tr><th>Run</th><th>Модель</th><th>Статус</th>{classification ? <><th>PR-AUC</th><th>ROC-AUC</th><th>Precision</th><th>Recall</th><th>F1</th><th>Brier</th><th>Recall@10%</th></> : <><th>MAE</th><th>RMSE</th><th>WAPE</th><th>MASE</th><th>Рядов</th><th>Ошибки</th></>}<th>Время</th><th></th></tr></thead><tbody>{comparison.data?.runs.map((run) => <tr key={run.id}><td><code>{run.id}</code></td><td>{run.model_name}</td><td><StatusBadge status={run.status} />{run.error && <details><summary>Ошибка</summary><pre>{run.error}</pre></details>}</td>{classification ? <><td>{value(run.metrics?.pr_auc)}</td><td>{value(run.metrics?.roc_auc)}</td><td>{value(run.metrics?.precision)}</td><td>{value(run.metrics?.recall)}</td><td>{value(run.metrics?.f1)}</td><td>{value(run.metrics?.brier_score)}</td><td>{value(run.metrics?.recall_at_top_10)}</td></> : <><td>{value(run.metrics?.mae)}</td><td>{value(run.metrics?.rmse)}</td><td>{value(run.metrics?.wape, true)}</td><td>{value(run.metrics?.mase)}</td><td>{run.metrics?.processed_series ?? '—'}</td><td>{value(run.metrics?.error_rate, true)}</td></>}<td>{run.duration_seconds?.toFixed(2) ?? '—'} с</td><td>{run.status === 'completed' && <Link to={`/results?run=${encodeURIComponent(run.id)}`}>Результаты / CSV</Link>}</td></tr>)}</tbody></table>{!benchmarkId && <div className="empty">Сначала запустите benchmark.</div>}</section>
  </>
}
