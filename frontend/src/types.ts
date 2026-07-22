export type JobStatus = 'queued' | 'checking_access' | 'downloading' | 'extracting' | 'profiling' | 'normalizing' | 'installing' | 'completed' | 'failed' | 'cancelled'

export interface Source {
  id: string
  title: string
  provider: string
  remote_id?: string
  adapter: string
  supported_tasks: string[]
  cash_gap_target: boolean
  requires_auth?: boolean
  license_note?: string
  usage_note?: string
  limitations?: string[]
  access_status: string
  access_message?: string
  source_revision?: string
}

export interface Job {
  id: string
  job_type: string
  status: JobStatus
  source_id?: string
  dataset_id?: string
  progress: number
  message?: string
  result?: Record<string, unknown>
  options?: Record<string, unknown>
  error?: string
  created_at: string
  updated_at: string
}

export interface Compatibility {
  classification_eligible: boolean
  forecasting_eligible: boolean
  proxy_eligible: boolean
  categorization_eligible: boolean
  reasons: string[]
  limitations?: string[]
  supported_tasks?: string[]
}

export interface Dataset {
  id: string
  created_at: string
  updated_at: string
  status: JobStatus | 'running'
  config: Record<string, unknown> & { n_clients?: number; n_months?: number; source_id?: string; stage?: string }
  summary?: {
    stage?: string
    source_id?: string
    source_provider?: string
    rows?: number
    monthly_rows?: number
    clients?: number
    n_clients?: number
    months?: number
    n_months?: number
    first_month?: string
    last_month?: string
    has_debit?: boolean
    has_credit?: boolean
    has_balance?: boolean
    has_cash_gap_target?: boolean
    multiple_currencies?: boolean
    target_rate?: number
    missing_percent?: number
    size_bytes?: number
    quality_flags?: string[]
    mapping?: Record<string, unknown>
    compatibility?: Compatibility
    file_list?: { name: string; size_bytes: number }[]
  }
  paths?: Record<string, string>
  error?: string
}

export interface Metrics {
  pr_auc?: number | null
  roc_auc?: number | null
  precision?: number
  recall?: number
  f1?: number
  brier_score?: number
  precision_at_top_10?: number
  recall_at_top_10?: number
  confusion_matrix?: number[][]
  threshold?: number
  mae?: number
  rmse?: number
  wape?: number | null
  mase?: number | null
  accuracy?: number
  f1_macro?: number
  training_seconds?: number
  processed_series?: number
  failed_series?: number
  error_rate?: number
}

export interface Experiment {
  id: string
  dataset_id: string
  model_name: string
  task: 'cash_gap_classification' | 'cash_flow_forecasting' | 'flow_forecasting' | 'transaction_categorization'
  status: 'queued' | 'running' | 'completed' | 'failed'
  created_at: string
  duration_seconds?: number
  params: Record<string, unknown>
  metrics?: Metrics
  feature_importance?: { feature: string; importance: number }[]
  split?: Record<string, unknown>
  error?: string
}

export type ModelStatus = 'AVAILABLE' | 'NOT_INSTALLED' | 'INSTALLING' | 'INSTALLED' | 'INCOMPATIBLE' | 'AUTH_REQUIRED' | 'FAILED'

export interface ModelEnvironment {
  status: ModelStatus
  message: string
  installed: boolean
  dependency_installed: boolean
  weights_cached: boolean
  size_bytes: number
  revision?: string
  install_command?: string
}

export interface ModelDefinition {
  id: string
  name: string
  type: 'competition_recipe' | 'pretrained_model' | 'local_trainable_model'
  provider: string
  task: 'cash_flow_forecasting' | 'cash_gap_classification'
  plugin: string
  requires_training: boolean
  cpu_supported: boolean
  compatible_targets: string[]
  description: string
  model_id?: string
  kernel_ref?: string
  source_url?: string
  license?: string | null
  limitations: string[]
  supports_zero_shot?: boolean
  supports_multivariate?: boolean
  supports_covariates?: boolean
  bundled?: boolean
  environment: ModelEnvironment
}

export interface ModelCompatibility {
  compatible: boolean
  reasons: string[]
  target?: string
  task?: string
  estimated_series: number
  estimated_memory_mb: number
  device: string
  requires_training: boolean
  requires_target: boolean
  details: Record<string, unknown>
}

export interface Benchmark {
  id: string
  dataset_id: string
  task: 'cash_flow_forecasting' | 'cash_gap_classification'
  target: string
  series_level: 'client' | 'client_category'
  horizon: number
  min_history: number
  model_ids: string[]
  run_ids: string[]
  status: string
  error?: string
  created_at: string
  completed_at?: string
}

export interface BenchmarkComparison {
  benchmark: Benchmark
  runs: Experiment[]
  comparable: boolean
  comparison_contract: { dataset_id: string; task: string; target: string; horizon: number; series_level: string }
}
