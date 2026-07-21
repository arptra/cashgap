import { NavLink, Outlet } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'

const nav = [
  ['/models', '◈', 'Модели', 'Решения и алгоритмы'],
  ['/run', '↗', 'Запуск', 'Benchmark wizard'],
  ['/experiments', '≡', 'Эксперименты', 'Журнал запусков'],
  ['/comparison', '▥', 'Сравнение', 'Единый контракт'],
  ['/datasets', '▤', 'Данные', 'Raw и normalized'],
  ['/sources', '⌁', 'Источники данных', 'Kaggle / HF / файлы'],
  ['/settings', '⚙', 'Настройки', 'Cache и зависимости'],
]

export default function Layout() {
  const health = useQuery({ queryKey: ['health'], queryFn: async () => (await api.get('/health')).data, refetchInterval: 5000, retry: 0 })
  return <div className="app-shell">
    <aside>
      <div className="brand"><span>CG</span><div><b>CashGap</b><small>MODEL LAB</small></div></div>
      <nav>{nav.map(([path, icon, label, sub]) => <NavLink key={path} to={path} className={({ isActive }) => isActive ? 'active' : ''}><i>{icon}</i><span>{label}<small>{sub}</small></span></NavLink>)}</nav>
      <div className="connection"><i className={health.isSuccess ? 'online' : ''} /><span>{health.isSuccess ? 'API подключён' : 'API недоступен'}</span></div>
    </aside>
    <main><Outlet /></main>
  </div>
}
