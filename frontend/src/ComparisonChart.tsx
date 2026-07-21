import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

export interface ComparisonChartRow {
  name: string
  'PR-AUC': number
  Precision: number
  Recall: number
}

export default function ComparisonChart({ data }: { data: ComparisonChartRow[] }) {
  return <ResponsiveContainer width="100%" height="100%">
    <BarChart data={data}>
      <CartesianGrid strokeDasharray="3 3" stroke="#dbe3ee" />
      <XAxis dataKey="name" />
      <YAxis domain={[0, 1]} />
      <Tooltip />
      <Legend />
      <Bar dataKey="PR-AUC" fill="#176b5b" radius={[5, 5, 0, 0]} />
      <Bar dataKey="Precision" fill="#d5a327" radius={[5, 5, 0, 0]} />
      <Bar dataKey="Recall" fill="#4c78a8" radius={[5, 5, 0, 0]} />
    </BarChart>
  </ResponsiveContainer>
}

