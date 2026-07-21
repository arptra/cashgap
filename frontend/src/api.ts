import axios from 'axios'
import type { Benchmark, BenchmarkComparison, Dataset, Experiment, Job, ModelCompatibility, ModelDefinition, ModelEnvironment, Source } from './types'

export const api = axios.create({ baseURL: '/api', timeout: 60_000 })

export const sourceApi = {
  list: async () => (await api.get<Source[]>('/sources')).data,
  check: async (id: string, options: Record<string, unknown> = {}) =>
    (await api.post<{ job_id: string }>(`/sources/${id}/check-access`, { options })).data,
  download: async (id: string, options: Record<string, unknown> = {}) =>
    (await api.post<{ job_id: string; dataset_id: string }>(`/sources/${id}/download`, { accepted_terms: true, options })).data,
}

export const jobApi = {
  list: async () => (await api.get<Job[]>('/jobs')).data,
  cancel: async (id: string) => (await api.post(`/jobs/${id}/cancel`)).data,
}

export const datasetApi = {
  list: async () => (await api.get<Dataset[]>('/datasets')).data,
  preview: async (id: string) => (await api.get<{ items: Record<string, unknown>[]; columns: string[] }>(`/datasets/${id}/preview`)).data,
  normalize: async (id: string, options: Record<string, unknown> = {}) =>
    (await api.post<{ job_id: string }>(`/datasets/${id}/normalize`, { options })).data,
  remove: async (id: string) => api.delete(`/datasets/${id}`),
  generate: async (payload: Record<string, number>) =>
    (await api.post<{ job_id: string; dataset_id: string }>('/synthetic/generate', payload)).data,
}

export const experimentApi = {
  list: async () => (await api.get<Experiment[]>('/experiments')).data,
  start: async (payload: Record<string, unknown>) =>
    (await api.post<{ run_ids: string[] }>('/experiments/start', payload)).data,
  compare: async (runIds: string[]) =>
    (await api.post<{ runs: Experiment[]; same_task: boolean }>('/experiments/compare', { run_ids: runIds })).data,
  predictions: async (id: string) =>
    (await api.get<{ items: Record<string, unknown>[]; total: number }>(`/experiments/${id}/predictions`, { params: { limit: 1000 } })).data,
  remove: async (id: string) => api.delete(`/experiments/${id}`),
}

export const modelApi = {
  list: async () => (await api.get<ModelDefinition[]>('/models')).data,
  get: async (id: string) => (await api.get<ModelDefinition>(`/models/${id}`)).data,
  check: async (id: string) => (await api.post<ModelEnvironment>(`/models/${id}/check`)).data,
  install: async (id: string) => (await api.post<{ job_id: string }>(`/models/${id}/install`)).data,
  uninstall: async (id: string) => api.delete(`/models/${id}/install`),
  compatibility: async (id: string, payload: Record<string, unknown>) =>
    (await api.post<ModelCompatibility>(`/models/${id}/compatibility`, payload)).data,
  run: async (id: string, payload: Record<string, unknown>) =>
    (await api.post<{ job_id: string; benchmark_id: string; run_ids: string[] }>(`/models/${id}/run`, payload)).data,
  jobs: async () => (await api.get<Job[]>('/model-jobs')).data,
}

export const benchmarkApi = {
  start: async (payload: Record<string, unknown>) =>
    (await api.post<{ job_id: string; benchmark_id: string; run_ids: string[] }>('/benchmarks/start', payload)).data,
  list: async () => (await api.get<Benchmark[]>('/benchmarks')).data,
  get: async (id: string) => (await api.get<Benchmark>(`/benchmarks/${id}`)).data,
  comparison: async (id: string) => (await api.get<BenchmarkComparison>(`/benchmarks/${id}/comparison`)).data,
}

export function errorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail
    if (typeof detail === 'string') return detail
    if (detail?.message) return `${detail.message}: ${(detail.reasons ?? []).join('; ')}`
    return JSON.stringify(detail ?? error.message)
  }
  return error instanceof Error ? error.message : 'Неизвестная ошибка'
}
