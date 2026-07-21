import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQueries, useQuery } from '@tanstack/react-query'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { benchmarkApi, datasetApi, errorMessage, modelApi } from '../api'
import PageHeader from '../components/PageHeader'

export default function RunModelPage() {
  const navigate = useNavigate()
  const [search] = useSearchParams()
  const models = useQuery({ queryKey: ['models'], queryFn: modelApi.list })
  const datasets = useQuery({ queryKey: ['datasets'], queryFn: datasetApi.list })
  const availableDatasets = datasets.data?.filter((dataset) => dataset.status === 'completed' && dataset.summary?.stage === 'normalized') ?? []
  const [datasetId, setDatasetId] = useState('')
  const [task, setTask] = useState<'cash_flow_forecasting' | 'cash_gap_classification'>('cash_flow_forecasting')
  const [target, setTarget] = useState('net_flow')
  const [selectedModels, setSelectedModels] = useState<string[]>([])
  const [seriesLevel, setSeriesLevel] = useState<'client' | 'client_category'>('client')
  const [horizon, setHorizon] = useState(1)
  const [minHistory, setMinHistory] = useState(8)
  const [parameters, setParameters] = useState({ batch_size: 16, iterations: 120, depth: 5, learning_rate: .05, season_length: 12 })
  useEffect(() => { if (!datasetId && availableDatasets[0]) setDatasetId(availableDatasets[0].id) }, [availableDatasets, datasetId])
  useEffect(() => { const id = search.get('model'); const model = models.data?.find((item) => item.id === id); if (model) { setTask(model.task); setTarget(model.task === 'cash_gap_classification' ? 'cash_gap_next_month' : 'net_flow'); setSelectedModels([model.id]) } }, [models.data, search])
  useEffect(() => { const ids = (search.get('models') ?? '').split(',').filter(Boolean); const selected = models.data?.filter((model) => ids.includes(model.id)) ?? []; if (selected.length) { const chosenTask = selected[0].task; setTask(chosenTask); setTarget(chosenTask === 'cash_gap_classification' ? 'cash_gap_next_month' : 'net_flow'); setSelectedModels(selected.filter((model) => model.task === chosenTask).map((model) => model.id)) } }, [models.data, search])
  const candidates = useMemo(() => models.data?.filter((model) => model.task === task) ?? [], [models.data, task])
  const compatibility = useQueries({ queries: selectedModels.map((modelId) => ({ queryKey: ['wizard-compatibility', modelId, datasetId, target, seriesLevel, horizon, minHistory], queryFn: () => modelApi.compatibility(modelId, { dataset_id: datasetId, target, series_level: seriesLevel, horizon, min_history: minHistory }), enabled: Boolean(datasetId), retry: false })) })
  const selectedDefinitions = selectedModels.map((id) => models.data?.find((model) => model.id === id)).filter(Boolean)
  const environmentsReady = selectedDefinitions.every((model) => model?.type === 'local_trainable_model' || model?.environment.status === 'INSTALLED')
  const compatible = compatibility.length > 0 && compatibility.every((query) => query.data?.compatible)
  const estimate = compatibility.find((query) => query.data)?.data
  const start = useMutation({ mutationFn: () => benchmarkApi.start({ dataset_id: datasetId, task, target, model_ids: selectedModels, series_level: seriesLevel, horizon, min_history: minHistory, parameters: Object.fromEntries(selectedModels.map((id) => [id, parameters])) }), onSuccess: (result) => navigate(`/comparison?benchmark=${result.benchmark_id}`) })
  const changeTask = (value: 'cash_flow_forecasting' | 'cash_gap_classification') => { setTask(value); setTarget(value === 'cash_gap_classification' ? 'cash_gap_next_month' : 'net_flow'); setSelectedModels([]) }
  const toggle = (id: string) => setSelectedModels((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id])
  return <>
    <PageHeader eyebrow="Six-step benchmark" title="Запуск модели" text="Все выбранные модели получают один dataset, target, horizon и test-период — результаты можно сравнивать честно." />
    <section className="wizard-grid">
      <article className="panel wizard-step"><b>1</b><h2>Dataset</h2><select value={datasetId} onChange={(event) => setDatasetId(event.target.value)}><option value="">Выберите dataset</option>{availableDatasets.map((dataset) => <option value={dataset.id} key={dataset.id}>{dataset.id} · {dataset.summary?.clients ?? dataset.summary?.n_clients} клиентов</option>)}</select>{!availableDatasets.length && <p className="warning">Нет normalized datasets. <Link to="/generator">Создать синтетику</Link></p>}</article>
      <article className="panel wizard-step"><b>2</b><h2>Задача</h2><div className="task-tabs"><button className={task === 'cash_flow_forecasting' ? 'active' : ''} onClick={() => changeTask('cash_flow_forecasting')}>Cash flow forecasting</button><button className={task === 'cash_gap_classification' ? 'active' : ''} onClick={() => changeTask('cash_gap_classification')}>Cash-gap classification</button></div></article>
      <article className="panel wizard-step"><b>3</b><h2>Target</h2>{task === 'cash_flow_forecasting' ? <select value={target} onChange={(event) => setTarget(event.target.value)}><option value="total_credit_sum">Credit</option><option value="total_debit_sum">Debit</option><option value="net_flow">Net flow</option></select> : <><input value="cash_gap_next_month" disabled /><p className="hint">Требуется исторический target; fraud/AML и balance proxy запрещены.</p></>}</article>
      <article className="panel wizard-step wizard-models"><b>4</b><h2>Модели</h2><div className="wizard-model-grid">{candidates.map((model) => { const ready = model.type === 'local_trainable_model' || model.environment.status === 'INSTALLED'; return <label className={`${selectedModels.includes(model.id) ? 'selected' : ''} ${!ready ? 'disabled' : ''}`} key={model.id}><input type="checkbox" checked={selectedModels.includes(model.id)} disabled={!ready} onChange={() => toggle(model.id)} /><span>{model.name}<small>{model.requires_training ? 'обучение' : 'zero-shot'} · {model.environment.status}</small></span></label> })}</div></article>
      <article className="panel wizard-step"><b>5</b><h2>Параметры</h2>{task === 'cash_flow_forecasting' && <div className="form-grid compact"><label>Уровень<select value={seriesLevel} onChange={(event) => setSeriesLevel(event.target.value as 'client' | 'client_category')}><option value="client">Клиент</option><option value="client_category">Клиент + категория</option></select></label><label>Horizon<select value={horizon} onChange={(event) => setHorizon(Number(event.target.value))}><option value="1">1 месяц</option><option value="2">2 месяца</option><option value="3">3 месяца</option></select></label><label>Мин. история<input type="number" min="6" value={minHistory} onChange={(event) => setMinHistory(Number(event.target.value))} /></label><label>Batch size<input type="number" min="1" value={parameters.batch_size} onChange={(event) => setParameters({ ...parameters, batch_size: Number(event.target.value) })} /></label></div>}</article>
      <article className="panel wizard-step"><b>6</b><h2>Проверка и запуск</h2><div className="run-estimate"><span><small>Рядов</small><strong>{estimate?.estimated_series ?? '—'}</strong></span><span><small>Память</small><strong>{estimate ? `${estimate.estimated_memory_mb} MB` : '—'}</strong></span><span><small>Устройство</small><strong>CPU</strong></span><span><small>Horizon</small><strong>{task === 'cash_gap_classification' ? 1 : horizon}</strong></span></div>{compatibility.map((query, index) => query.data && <p className={query.data.compatible ? 'notice' : 'warning'} key={selectedModels[index]}>{selectedModels[index]}: {query.data.compatible ? 'совместима' : query.data.reasons.join('; ')}</p>)}{!environmentsReady && <p className="warning">Сначала установите веса или подключите competition source на странице моделей.</p>}<button className="button large-action" disabled={!datasetId || !selectedModels.length || !compatible || !environmentsReady || start.isPending} onClick={() => start.mutate()}>{start.isPending ? 'Запускаю benchmark…' : 'Запустить benchmark'}</button>{start.error && <p className="error-message">{errorMessage(start.error)}</p>}</article>
    </section>
  </>
}
