import { TrendingUp, Minus, TrendingDown } from 'lucide-react'

const SENTIMENT_CONFIG = {
  positive: {
    Icon:   TrendingUp,
    label:  'Positive',
    bg:     'bg-brand-50',
    border: 'border-brand-200',
    text:   'text-primary',
    iconBg: 'bg-brand-100',
    bar:    'bg-primary',
  },
  neutral: {
    Icon:   Minus,
    label:  'Neutral',
    bg:     'bg-yellow-50',
    border: 'border-yellow-200',
    text:   'text-yellow-600',
    iconBg: 'bg-yellow-100',
    bar:    'bg-yellow-400',
  },
  negative: {
    Icon:   TrendingDown,
    label:  'Negative',
    bg:     'bg-red-50',
    border: 'border-red-200',
    text:   'text-red-600',
    iconBg: 'bg-red-100',
    bar:    'bg-red-400',
  },
}

const SCORE_BAR_COLORS = {
  positive: 'bg-primary',
  neutral:  'bg-yellow-400',
  negative: 'bg-red-400',
}

export default function SentimentGauge({ sentiment, confidence, scores }) {
  const cfg = SENTIMENT_CONFIG[sentiment] || SENTIMENT_CONFIG.neutral
  const { Icon } = cfg

  return (
    <div className={`rounded-2xl border p-6 space-y-5 ${cfg.bg} ${cfg.border}`}>
      {/* Icon + label */}
      <div className="flex flex-col items-center gap-2">
        <div className={`w-14 h-14 rounded-2xl ${cfg.iconBg} flex items-center justify-center`}>
          <Icon className={`w-7 h-7 ${cfg.text}`} strokeWidth={2} />
        </div>
        <div className={`text-xl font-bold ${cfg.text}`}>{cfg.label}</div>
        <div className="text-sm text-gray-400 font-medium">
          {(confidence * 100).toFixed(1)}% confidence
        </div>
      </div>

      {/* Score bars */}
      <div className="space-y-2.5 max-w-[220px] mx-auto">
        {Object.entries(scores)
          .sort(([, a], [, b]) => b - a)
          .map(([label, score]) => (
            <div key={label} className="flex items-center gap-2">
              <span className="text-xs w-14 text-right capitalize text-gray-500 font-medium">
                {label}
              </span>
              <div className="flex-1 bg-white/70 rounded-full h-2 overflow-hidden border border-gray-100">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${SCORE_BAR_COLORS[label] || 'bg-gray-300'}`}
                  style={{ width: `${score * 100}%` }}
                />
              </div>
              <span className="text-xs w-9 text-gray-400 font-medium">
                {(score * 100).toFixed(0)}%
              </span>
            </div>
          ))}
      </div>
    </div>
  )
}
