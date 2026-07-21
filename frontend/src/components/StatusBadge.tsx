import type { JobStatus } from '../types'

export default function StatusBadge({ status }: { status: JobStatus | 'running' | string }) {
  const normalized = status.toLowerCase()
  const labels: Record<string, string> = {
    queued: 'в очереди', checking_access: 'проверка', downloading: 'скачивание',
    extracting: 'распаковка', profiling: 'профилирование', normalizing: 'нормализация',
    running: 'в работе', completed: 'готово', failed: 'ошибка', cancelled: 'отменено',
    accessible: 'доступен', denied: 'нет доступа', unknown: 'не проверен',
  }
  const modelLabels: Record<string, string> = { available: 'доступна', not_installed: 'не установлена', installing: 'установка', installed: 'установлена', incompatible: 'несовместима', auth_required: 'нужна авторизация' }
  const visual = normalized === 'accessible' || normalized === 'installed' ? 'completed' : ['denied', 'failed', 'not_installed', 'incompatible'].includes(normalized) ? 'failed' : ['available', 'installing', 'auth_required'].includes(normalized) ? 'running' : normalized
  return <span className={`status status--${visual}`}><i />{labels[normalized] ?? modelLabels[normalized] ?? status}</span>
}
