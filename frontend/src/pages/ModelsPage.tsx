import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { datasetApi, errorMessage, modelApi } from '../api'
import PageHeader from '../components/PageHeader'
import StatusBadge from '../components/StatusBadge'
import type { ModelDefinition } from '../types'

const typeNames = {
  competition_recipe: 'Решение соревнования',
  pretrained_model: 'Готовая модель',
  local_trainable_model: 'Локальная модель',
}

function bytes(value: number) {
  if (!value) return '—'
  if (value > 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GB`
  return `${(value / 1024 ** 2).toFixed(1)} MB`
}

function ModelCard({ model, datasetId, compared, toggle }: { model: ModelDefinition; datasetId: string; compared: boolean; toggle: () => void }) {
  const navigate = useNavigate()
  const client = useQueryClient()
  const target = model.task === 'cash_gap_classification' ? 'cash_gap_next_month' : 'net_flow'
  const compatibility = useQuery({ queryKey: ['model-compatibility', model.id, datasetId, target], queryFn: () => modelApi.compatibility(model.id, { dataset_id: datasetId, target, series_level: 'client', horizon: 1, min_history: 6 }), enabled: Boolean(datasetId), retry: false })
  const check = useMutation({ mutationFn: () => modelApi.check(model.id), onSuccess: () => client.invalidateQueries({ queryKey: ['models'] }) })
  const install = useMutation({ mutationFn: () => modelApi.install(model.id), onSuccess: () => { void client.invalidateQueries({ queryKey: ['model-jobs'] }); void client.invalidateQueries({ queryKey: ['models'] }) } })
  const remove = useMutation({ mutationFn: () => modelApi.uninstall(model.id), onSuccess: () => client.invalidateQueries({ queryKey: ['models'] }) })
  const configure = () => navigate(`/run?model=${encodeURIComponent(model.id)}`)
  const error = check.error || install.error || remove.error || compatibility.error
  const isCompetition = model.type === 'competition_recipe'
  return <article className={`model-card model-card--${model.type}`}>
    <div className="card-top"><div><p className="eyebrow">{typeNames[model.type]}</p><h2>{model.name}</h2></div><StatusBadge status={model.environment.status} /></div>
    <p className="model-description">{model.description}</p>
    <div className="tag-row"><span>{model.provider}</span><span>{model.task}</span><span>{model.requires_training ? 'нужно обучение' : 'zero-shot'}</span><span>{model.cpu_supported ? 'CPU' : 'GPU'}</span></div>
    <dl className="source-meta"><div><dt>Совместимость</dt><dd>{!datasetId ? 'выберите dataset' : compatibility.isLoading ? 'проверяется' : compatibility.data?.compatible ? 'совместима' : 'нет'}</dd></div><div><dt>Размер</dt><dd>{bytes(model.environment.size_bytes)}</dd></div><div><dt>Лицензия</dt><dd>{model.license ?? 'не определена'}</dd></div></dl>
    <p className="hint">{model.environment.message}</p>
    {model.environment.revision && <p className="revision">Revision: <code>{model.environment.revision}</code></p>}
    {compatibility.data && !compatibility.data.compatible && <p className="warning">{compatibility.data.reasons.join('; ')}</p>}
    {model.limitations.length > 0 && <details><summary>Ограничения</summary><p className="hint">{model.limitations.join(', ')}</p></details>}
    {model.source_url && <a className="source-link" href={model.source_url} target="_blank" rel="noreferrer">Открыть источник ↗</a>}
    <div className="model-actions">
      <button className="button button--secondary" onClick={() => check.mutate()}>Проверить окружение</button>
      {!isCompetition && !model.environment.weights_cached && model.type === 'pretrained_model' && <button className="button" onClick={() => install.mutate()}>Установить</button>}
      {!isCompetition && model.environment.weights_cached && <button className="text-button danger" onClick={() => remove.mutate()}>Удалить веса</button>}
      <button className="button button--secondary" disabled={!datasetId} onClick={() => compatibility.refetch()}>Проверить совместимость</button>
      <button className="button button--secondary" onClick={configure}>Настроить</button>
      {!isCompetition && <button className="button" onClick={configure}>{model.requires_training ? 'Обучить' : 'Запустить'}</button>}
      <button className={`button ${compared ? '' : 'button--secondary'}`} onClick={toggle}>{compared ? 'В сравнении' : 'Добавить в сравнение'}</button>
    </div>
    {isCompetition && <div className="recipe-actions"><button className="button" onClick={() => install.mutate()}>Подключить решение</button><button className="button button--secondary" onClick={() => install.mutate()}>Получить исходный notebook</button><button className="button" onClick={configure}>Обучить на моих данных</button></div>}
    {model.environment.install_command && !model.environment.dependency_installed && <pre className="install-command">{model.environment.install_command}</pre>}
    {error && <p className="error-message">{errorMessage(error)}</p>}
  </article>
}

export default function ModelsPage() {
  const navigate = useNavigate()
  const models = useQuery({ queryKey: ['models'], queryFn: modelApi.list, refetchInterval: 5000 })
  const datasets = useQuery({ queryKey: ['datasets'], queryFn: datasetApi.list })
  const availableDatasets = datasets.data?.filter((dataset) => dataset.status === 'completed' && dataset.summary?.stage === 'normalized') ?? []
  const [datasetId, setDatasetId] = useState('')
  const [filter, setFilter] = useState('all')
  const [compared, setCompared] = useState<string[]>(() => JSON.parse(localStorage.getItem('cashgap_compare_models') || '[]'))
  const visible = useMemo(() => models.data?.filter((model) => filter === 'all' || model.type === filter) ?? [], [models.data, filter])
  const toggle = (id: string) => setCompared((current) => { const next = current.includes(id) ? current.filter((item) => item !== id) : [...current, id]; localStorage.setItem('cashgap_compare_models', JSON.stringify(next)); return next })
  return <>
    <PageHeader eyebrow="Model-first workspace" title="Модели и решения" text="Готовые модели, решения соревнований и локальные алгоритмы для прогнозирования денежных потоков и риска кассового разрыва." />
    <section className="panel model-toolbar"><label>Текущий dataset<select value={datasetId} onChange={(event) => setDatasetId(event.target.value)}><option value="">Выберите для проверки совместимости</option>{availableDatasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.id} · {dataset.summary?.source_id}</option>)}</select></label><div className="task-tabs"><button className={filter === 'all' ? 'active' : ''} onClick={() => setFilter('all')}>Все</button><button className={filter === 'competition_recipe' ? 'active' : ''} onClick={() => setFilter('competition_recipe')}>Соревнования</button><button className={filter === 'pretrained_model' ? 'active' : ''} onClick={() => setFilter('pretrained_model')}>Pretrained</button><button className={filter === 'local_trainable_model' ? 'active' : ''} onClick={() => setFilter('local_trainable_model')}>Локальные</button></div><span className="selection-counter">В сравнении: {compared.length}</span><button className="button" disabled={!compared.length} onClick={() => navigate(`/run?models=${encodeURIComponent(compared.join(','))}`)}>Настроить выбранные</button></section>
    {models.error && <p className="error-message">{errorMessage(models.error)}</p>}
    <div className="model-catalog">{visible.map((model) => <ModelCard key={model.id} model={model} datasetId={datasetId} compared={compared.includes(model.id)} toggle={() => toggle(model.id)} />)}</div>
  </>
}
