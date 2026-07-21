import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { datasetApi, errorMessage, experimentApi } from '../api'
import PageHeader from '../components/PageHeader'

const taskModels: Record<string, { id: string; label: string }[]> = {
  cash_gap_classification: [
    { id: 'dummy', label: 'Dummy' }, { id: 'logistic_regression', label: 'Logistic Regression' },
    { id: 'random_forest', label: 'Random Forest' }, { id: 'catboost', label: 'CatBoost' }, { id: 'lightgbm', label: 'LightGBM' },
  ],
  flow_forecasting: [
    { id: 'seasonal_naive', label: 'Seasonal Naive' }, { id: 'auto_ets', label: 'AutoETS' },
    { id: 'auto_arima', label: 'AutoARIMA' }, { id: 'lightgbm_forecast', label: 'LightGBM lags' },
  ],
  transaction_categorization: [{ id: 'tfidf_logistic_regression', label: 'TF-IDF + LogisticRegression' }],
}

type Parameters = {
  C: number; max_iter: number; n_estimators: number; max_depth: number
  iterations: number; learning_rate: number; season_length: number; horizon: number; max_features: number
}

function parametersFor(model: string, values: Parameters): Record<string, number> {
  if (model === 'logistic_regression') return { C: values.C, max_iter: values.max_iter }
  if (model === 'random_forest') return { n_estimators: values.n_estimators, max_depth: values.max_depth }
  if (model === 'catboost') return { iterations: values.iterations, depth: values.max_depth, learning_rate: values.learning_rate }
  if (model === 'lightgbm' || model === 'lightgbm_forecast') return { n_estimators: values.n_estimators, max_depth: values.max_depth, learning_rate: values.learning_rate, horizon: values.horizon }
  if (['seasonal_naive', 'auto_ets', 'auto_arima'].includes(model)) return { season_length: values.season_length, horizon: values.horizon }
  if (model === 'tfidf_logistic_regression') return { max_features: values.max_features, max_iter: values.max_iter }
  return {}
}

export default function TrainingPage() {
  const client = useQueryClient()
  const datasets = useQuery({ queryKey: ['datasets'], queryFn: datasetApi.list, refetchInterval: 3000 })
  const [datasetId, setDatasetId] = useState('')
  const [task, setTask] = useState('cash_gap_classification')
  const [models, setModels] = useState<string[]>(['logistic_regression'])
  const [ratios, setRatios] = useState({ train: .6, validation: .2 })
  const [parameters, setParameters] = useState<Parameters>({ C: 1, max_iter: 500, n_estimators: 200, max_depth: 8, iterations: 250, learning_rate: .05, season_length: 12, horizon: 3, max_features: 30000 })
  const selected = datasets.data?.find((dataset) => dataset.id === datasetId)
  const eligibleTasks = useMemo(() => {
    const compatibility = selected?.summary?.compatibility
    if (!compatibility) return []
    return [
      compatibility.classification_eligible && 'cash_gap_classification',
      compatibility.forecasting_eligible && 'flow_forecasting',
      compatibility.categorization_eligible && 'transaction_categorization',
    ].filter(Boolean) as string[]
  }, [selected])
  useEffect(() => { if (!datasetId && datasets.data?.[0]) setDatasetId(datasets.data[0].id) }, [datasetId, datasets.data])
  useEffect(() => { if (eligibleTasks.length && !eligibleTasks.includes(task)) { setTask(eligibleTasks[0]); setModels([taskModels[eligibleTasks[0]][0].id]) } }, [eligibleTasks, task])
  const start = useMutation({ mutationFn: () => experimentApi.start({ dataset_id: datasetId, task, models, parameters: Object.fromEntries(models.map((model) => [model, parametersFor(model, parameters)])), train_ratio: ratios.train, validation_ratio: ratios.validation }), onSuccess: () => client.invalidateQueries({ queryKey: ['experiments'] }) })
  const toggle = (id: string) => setModels((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id])
  const parameter = (key: keyof Parameters, step = 1) => <input type="number" step={step} value={parameters[key]} onChange={(event) => setParameters({ ...parameters, [key]: Number(event.target.value) })} />
  return <>
    <PageHeader eyebrow="CPU experiments" title="Обучение" text="UI показывает только задачи, разрешённые compatibility report выбранного dataset." />
    <section className="panel training-layout">
      <div><h2>1. Dataset</h2><label className="wide-label">Normalized dataset<select value={datasetId} onChange={(event) => setDatasetId(event.target.value)}><option value="">Выберите</option>{datasets.data?.filter((item) => item.status === 'completed' && item.summary?.stage === 'normalized').map((item) => <option key={item.id} value={item.id}>{item.id} · {item.summary?.source_id}</option>)}</select></label>{selected?.summary?.compatibility?.reasons.map((reason) => <p className="hint" key={reason}>{reason}</p>)}</div>
      <div><h2>2. Задача</h2><div className="task-tabs">{Object.keys(taskModels).map((name) => <button key={name} disabled={!eligibleTasks.includes(name)} className={task === name ? 'active' : ''} onClick={() => { setTask(name); setModels([taskModels[name][0].id]) }}>{name}</button>)}</div></div>
      <div><h2>3. Модели</h2><div className="model-grid">{taskModels[task].map((model) => <label className={`model-option ${models.includes(model.id) ? 'selected' : ''}`} key={model.id}><input type="checkbox" checked={models.includes(model.id)} onChange={() => toggle(model.id)} /><span>{model.label}</span><small>CPU</small></label>)}</div></div>
      <div><h2>4. Основные параметры</h2><div className="form-grid compact">{task === 'cash_gap_classification' && <><label>Logistic C{parameter('C', .1)}</label><label>Max iterations{parameter('max_iter')}</label><label>Trees / estimators{parameter('n_estimators')}</label><label>Tree depth{parameter('max_depth')}</label><label>CatBoost iterations{parameter('iterations')}</label><label>Learning rate{parameter('learning_rate', .01)}</label></>}{task === 'flow_forecasting' && <><label>Horizon, месяцев{parameter('horizon')}</label><label>Season length{parameter('season_length')}</label><label>LightGBM estimators{parameter('n_estimators')}</label><label>Learning rate{parameter('learning_rate', .01)}</label></>}{task === 'transaction_categorization' && <><label>TF-IDF max features{parameter('max_features')}</label><label>Max iterations{parameter('max_iter')}</label></>}</div></div>
      {task === 'cash_gap_classification' && <div><h2>5. Временные периоды</h2><div className="form-grid compact"><label>Train ratio<input type="number" min="0.4" max="0.8" step="0.05" value={ratios.train} onChange={(event) => setRatios({ ...ratios, train: Number(event.target.value) })} /></label><label>Validation ratio<input type="number" min="0.1" max="0.3" step="0.05" value={ratios.validation} onChange={(event) => setRatios({ ...ratios, validation: Number(event.target.value) })} /></label><div className="split-preview"><span style={{ width: `${ratios.train * 100}%` }}>train</span><span style={{ width: `${ratios.validation * 100}%` }}>val</span><span style={{ width: `${(1 - ratios.train - ratios.validation) * 100}%` }}>test</span></div></div></div>}
      <button className="button" disabled={!datasetId || !eligibleTasks.includes(task) || !models.length || start.isPending} onClick={() => start.mutate()}>Запустить обучение</button>
      {start.data && <p className="notice">Созданы runs: {start.data.run_ids.join(', ')}</p>}{start.error && <p className="error-message">{errorMessage(start.error)}</p>}
    </section>
  </>
}
