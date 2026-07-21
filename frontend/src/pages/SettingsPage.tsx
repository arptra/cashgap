import { useQuery } from '@tanstack/react-query'
import { modelApi } from '../api'
import PageHeader from '../components/PageHeader'
import StatusBadge from '../components/StatusBadge'

export default function SettingsPage() {
  const models = useQuery({ queryKey: ['models'], queryFn: modelApi.list })
  const jobs = useQuery({ queryKey: ['model-jobs'], queryFn: modelApi.jobs, refetchInterval: 3000 })
  return <>
    <PageHeader eyebrow="Local runtime" title="Настройки" text="Зависимости, model cache и фоновые операции. Credentials читаются только из environment или стандартных файлов провайдеров." />
    <section className="panel settings-grid"><div><h2>Model cache</h2><code>cache/models/&lt;model_id&gt;/</code><p className="hint">Сохраняются safetensors, config и закреплённый Hugging Face revision. Pickle/joblib извне блокируются.</p></div><div><h2>Competition sources</h2><code>external/competition_sources/&lt;recipe_id&gt;/</code><p className="hint">Notebook хранится только для attribution и никогда не исполняется.</p></div></section>
    <section className="panel table-wrap"><table><thead><tr><th>Модель</th><th>Статус</th><th>Команда зависимости</th><th>Revision</th></tr></thead><tbody>{models.data?.map((model) => <tr key={model.id}><td>{model.name}</td><td><StatusBadge status={model.environment.status} /></td><td><code>{model.environment.install_command ?? 'встроена'}</code></td><td><code>{model.environment.revision ?? '—'}</code></td></tr>)}</tbody></table></section>
    <section className="panel table-wrap"><h2>Model jobs</h2><table><thead><tr><th>Job</th><th>Тип</th><th>Статус</th><th>Сообщение</th></tr></thead><tbody>{jobs.data?.map((job) => <tr key={job.id}><td><code>{job.id}</code></td><td>{job.job_type}</td><td><StatusBadge status={job.status} /></td><td>{job.message}{job.error && <details><summary>Ошибка</summary><pre>{job.error}</pre></details>}</td></tr>)}</tbody></table></section>
  </>
}
