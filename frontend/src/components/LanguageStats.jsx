import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'

const LANG_COLORS = {
  lang1: '#f97316',
  lang2: '#3b82f6',
  mixed: '#a855f7',
  ne:    '#10B981',
  other: '#9ca3af',
  univ:  '#d1d5db',
}

const LANG_LABELS = {
  lang1: 'Hindi',
  lang2: 'English',
  mixed: 'Mixed',
  ne:    'Named Entity',
  other: 'Other',
  univ:  'Universal',
}

function CMIGauge({ value }) {
  const pct = Math.round(value * 100)
  const { color, barColor, label } =
    pct < 20
      ? { color: 'text-blue-500',   barColor: 'bg-blue-400',   label: 'Low mixing' }
      : pct < 50
      ? { color: 'text-purple-500', barColor: 'bg-purple-400', label: 'Moderate mixing' }
      : { color: 'text-primary',    barColor: 'bg-primary',    label: 'High mixing' }

  return (
    <div className="flex flex-col items-center gap-2 p-4 bg-[#F3F4F6] rounded-2xl">
      <div className={`text-4xl font-bold ${color}`}>{pct}</div>
      <div className="text-xs text-gray-400 font-semibold uppercase tracking-wider">
        Code-Mixing Index
      </div>
      <div className={`text-xs font-semibold ${color}`}>{label}</div>
      <div className="w-full bg-white rounded-full h-2 border border-gray-200">
        <div
          className={`h-2 rounded-full transition-all duration-500 ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="text-xs text-gray-400 text-center leading-tight">
        0 = monolingual · 100 = maximally mixed
      </p>
    </div>
  )
}

export default function LanguageStats({ languageDistribution, codeMixingIndex }) {
  const chartData = Object.entries(languageDistribution)
    .filter(([, v]) => v > 0)
    .map(([key, value]) => ({
      name:  LANG_LABELS[key] || key,
      value: Math.round(value * 100),
      key,
    }))
    .sort((a, b) => b.value - a.value)

  if (chartData.length === 0) return null

  return (
    <div className="grid grid-cols-2 gap-6 items-center">
      {/* Pie chart */}
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              innerRadius={42}
              outerRadius={70}
              paddingAngle={2}
              dataKey="value"
            >
              {chartData.map((entry) => (
                <Cell key={entry.key} fill={LANG_COLORS[entry.key] || '#9ca3af'} />
              ))}
            </Pie>
            <Tooltip
              formatter={(value, name) => [`${value}%`, name]}
              contentStyle={{
                background: '#111827',
                border: 'none',
                borderRadius: '10px',
                color: '#F9FAFB',
                fontSize: '12px',
                fontFamily: 'Inter, sans-serif',
              }}
            />
            <Legend
              iconType="circle"
              iconSize={8}
              formatter={(value) => (
                <span style={{ fontSize: '11px', color: '#6b7280', fontFamily: 'Inter, sans-serif' }}>
                  {value}
                </span>
              )}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* CMI gauge */}
      <CMIGauge value={codeMixingIndex} />
    </div>
  )
}
