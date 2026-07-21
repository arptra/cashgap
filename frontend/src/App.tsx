import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import SourcesPage from './pages/SourcesPage'
import DatasetsPage from './pages/DatasetsPage'
import GeneratorPage from './pages/GeneratorPage'
import TrainingPage from './pages/TrainingPage'
import ExperimentsPage from './pages/ExperimentsPage'
import ComparisonPage from './pages/ComparisonPage'
import ResultsPage from './pages/ResultsPage'
import ModelsPage from './pages/ModelsPage'
import RunModelPage from './pages/RunModelPage'
import SettingsPage from './pages/SettingsPage'

export default function App() {
  return <Routes>
    <Route element={<Layout />}>
      <Route path="/models" element={<ModelsPage />} />
      <Route path="/run" element={<RunModelPage />} />
      <Route path="/sources" element={<SourcesPage />} />
      <Route path="/datasets" element={<DatasetsPage />} />
      <Route path="/generator" element={<GeneratorPage />} />
      <Route path="/training" element={<TrainingPage />} />
      <Route path="/experiments" element={<ExperimentsPage />} />
      <Route path="/comparison" element={<ComparisonPage />} />
      <Route path="/results" element={<ResultsPage />} />
      <Route path="/settings" element={<SettingsPage />} />
      <Route path="*" element={<Navigate to="/models" replace />} />
    </Route>
  </Routes>
}
