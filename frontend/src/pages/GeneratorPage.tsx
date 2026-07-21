import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { datasetApi, errorMessage } from '../api'
import PageHeader from '../components/PageHeader'

export default function GeneratorPage() {
  const client = useQueryClient()
  const [form, setForm] = useState({ n_clients: 3000, n_months: 24, target_gap_rate: 0.10, random_seed: 42, noise_level: 0.15, overdraft_share: 0.55 })
  const generation = useMutation({ mutationFn: () => datasetApi.generate(form), onSuccess: () => client.invalidateQueries({ queryKey: ['datasets'] }) })
  const field = (key: keyof typeof form, step = 1) => <input type="number" step={step} value={form[key]} onChange={(event) => setForm({ ...form, [key]: Number(event.target.value) })} />
  return <>
    <PageHeader eyebrow="Ground truth" title="Синтетический генератор" text="Дневная траектория баланса создаёт настоящий target следующего месяца; fraud/AML labels не используются." />
    <section className="panel form-panel"><div className="form-grid generator-grid"><label>Клиентов{field('n_clients')}</label><label>Месяцев{field('n_months')}</label><label>Event rate{field('target_gap_rate', .01)}</label><label>Random seed{field('random_seed')}</label><label>Noise{field('noise_level', .01)}</label><label>Доля overdraft{field('overdraft_share', .01)}</label></div><button className="button large-action" disabled={generation.isPending} onClick={() => generation.mutate()}>{generation.isPending ? 'Ставлю в очередь…' : 'Сгенерировать synthetic dataset'}</button>{generation.data && <p className="notice">Job {generation.data.job_id}; dataset {generation.data.dataset_id}</p>}{generation.error && <p className="error-message">{errorMessage(generation.error)}</p>}</section>
    <section className="panel explanation-grid"><div><b>1</b><h2>Дневные потоки</h2><p>Поступления, расходы, начальный и конечный баланс.</p></div><div><b>2</b><h2>Овердрафт и событие</h2><p>Cash gap фиксируется по недостаточности доступной ликвидности.</p></div><div><b>3</b><h2>Месячная витрина</h2><p>В признаки попадают только месячные агрегаты, без дневного target leakage.</p></div></section>
  </>
}

