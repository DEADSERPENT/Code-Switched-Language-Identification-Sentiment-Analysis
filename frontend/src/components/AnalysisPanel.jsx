import { useState } from 'react'
import { Download, Layers, BarChart2, PieChart } from 'lucide-react'
import TokenDisplay from './TokenDisplay'
import SentimentGauge from './SentimentGauge'
import LanguageStats from './LanguageStats'

function exportJSON(result) {
  const blob = new Blob([JSON.stringify(result, null, 2)], {
    type: 'application/json',
  })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'coswitchnlp_result.json'
  a.click()
  URL.revokeObjectURL(url)
}

function MetaBadge({ label, value }) {
  return (
    <div className="flex flex-col items-center gap-0.5 px-4 py-2 bg-[#F3F4F6] rounded-xl">
      <span className="text-xs text-gray-400 font-medium">{label}</span>
      <span className="text-sm font-bold text-[#111827]">{value}</span>
    </div>
  )
}

export default function AnalysisPanel({ result }) {
  const [tab, setTab] = useState('tokens')

  if (!result) return null

  const {
    tokens,
    sentiment,
    sentiment_confidence,
    sentiment_scores,
    code_mixing_index,
    language_distribution,
    processing_time_ms,
    inputText,
  } = result

  const tabs = [
    { id: 'tokens',    label: 'Token Analysis', icon: Layers },
    { id: 'sentiment', label: 'Sentiment',       icon: BarChart2 },
    { id: 'stats',     label: 'Language Stats',  icon: PieChart },
  ]

  return (
    <div className="card animate-slide-up overflow-hidden">
      {/* Input echo + meta */}
      <div className="px-6 pt-5 pb-4 border-b border-gray-100 space-y-3">
        <div>
          <p className="text-xs text-gray-400 font-semibold uppercase tracking-wider mb-1">
            Input
          </p>
          <p className="text-sm text-[#111827] font-medium leading-relaxed">
            "{inputText}"
          </p>
        </div>
        <div className="flex gap-3">
          <MetaBadge label="Tokens" value={tokens.length} />
          <MetaBadge label="Processed" value={`${processing_time_ms?.toFixed(0)}ms`} />
          <MetaBadge label="CMI" value={`${(code_mixing_index * 100).toFixed(0)}%`} />
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-100 bg-[#F3F4F6]/40">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`flex-1 flex items-center justify-center gap-2 py-3 text-sm font-medium transition-colors duration-150
              ${
                tab === id
                  ? 'text-primary border-b-2 border-primary bg-white'
                  : 'text-gray-400 hover:text-gray-600 hover:bg-white/60'
              }`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="p-6">
        {tab === 'tokens' && <TokenDisplay tokens={tokens} />}

        {tab === 'sentiment' && (
          <SentimentGauge
            sentiment={sentiment}
            confidence={sentiment_confidence}
            scores={sentiment_scores}
          />
        )}

        {tab === 'stats' && (
          <LanguageStats
            languageDistribution={language_distribution}
            codeMixingIndex={code_mixing_index}
          />
        )}
      </div>

      {/* Footer */}
      <div className="px-6 pb-5 pt-0 flex justify-end border-t border-gray-100">
        <button
          onClick={() => exportJSON(result)}
          className="btn-secondary flex items-center gap-1.5"
        >
          <Download className="w-3.5 h-3.5" />
          Export JSON
        </button>
      </div>
    </div>
  )
}
