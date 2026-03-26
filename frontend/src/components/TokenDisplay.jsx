import { useState } from 'react'

const LANG_CONFIG = {
  lang1: {
    bg:     'bg-orange-50',
    text:   'text-orange-700',
    border: 'border-orange-200',
    dot:    'bg-orange-400',
    label:  'Hindi',
    shortLabel: 'HI',
  },
  lang2: {
    bg:     'bg-blue-50',
    text:   'text-blue-700',
    border: 'border-blue-200',
    dot:    'bg-blue-400',
    label:  'English',
    shortLabel: 'EN',
  },
  mixed: {
    bg:     'bg-purple-50',
    text:   'text-purple-700',
    border: 'border-purple-200',
    dot:    'bg-purple-400',
    label:  'Mixed',
    shortLabel: 'MX',
  },
  ne: {
    bg:     'bg-brand-50',
    text:   'text-primary',
    border: 'border-brand-200',
    dot:    'bg-primary',
    label:  'Named Entity',
    shortLabel: 'NE',
  },
  other: {
    bg:     'bg-gray-50',
    text:   'text-gray-500',
    border: 'border-gray-200',
    dot:    'bg-gray-300',
    label:  'Other',
    shortLabel: '—',
  },
  univ: {
    bg:     'bg-gray-50',
    text:   'text-gray-400',
    border: 'border-gray-100',
    dot:    'bg-gray-200',
    label:  'Universal',
    shortLabel: 'UV',
  },
}

function Token({ token, language, confidence, showConfidence }) {
  const [hovered, setHovered] = useState(false)
  const cfg = LANG_CONFIG[language] || LANG_CONFIG.other

  return (
    <div
      className="relative flex flex-col items-center gap-0.5 group"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <span
        className={`px-3 py-1.5 rounded-lg border font-medium text-sm cursor-default
                    transition-transform duration-100 group-hover:scale-105
                    ${cfg.bg} ${cfg.text} ${cfg.border}`}
      >
        {token}
      </span>
      <span className={`text-xs font-semibold ${cfg.text} opacity-60`}>
        {cfg.shortLabel}
      </span>
      {showConfidence && (
        <span className="text-xs text-gray-400">
          {(confidence * 100).toFixed(0)}%
        </span>
      )}

      {/* Tooltip */}
      {hovered && (
        <div className="absolute -top-10 left-1/2 -translate-x-1/2 z-20
                        bg-[#111827] text-white text-xs
                        px-2.5 py-1 rounded-lg whitespace-nowrap pointer-events-none shadow-lg">
          {cfg.label} · {(confidence * 100).toFixed(1)}%
        </div>
      )}
    </div>
  )
}

const LEGEND_KEYS = ['lang1', 'lang2', 'mixed', 'ne', 'other']

export default function TokenDisplay({ tokens }) {
  const [showConfidence, setShowConfidence] = useState(false)

  if (!tokens?.length) return null

  return (
    <div className="space-y-4">
      {/* Legend + toggle */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex flex-wrap gap-3">
          {LEGEND_KEYS.map((key) => {
            const cfg = LANG_CONFIG[key]
            return (
              <div key={key} className="flex items-center gap-1.5">
                <span className={`w-2.5 h-2.5 rounded-full ${cfg.dot}`} />
                <span className="text-xs text-gray-500 font-medium">{cfg.label}</span>
              </div>
            )
          })}
        </div>
        <label className="flex items-center gap-2 cursor-pointer select-none">
          <span className="text-xs text-gray-500 font-medium">Show confidence</span>
          <button
            type="button"
            onClick={() => setShowConfidence((v) => !v)}
            className={`relative w-9 h-5 rounded-full transition-colors duration-200 ${
              showConfidence ? 'bg-primary' : 'bg-gray-200'
            }`}
          >
            <span
              className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow-sm
                          transition-transform duration-200 ${
                            showConfidence ? 'translate-x-4' : 'translate-x-0'
                          }`}
            />
          </button>
        </label>
      </div>

      {/* Tokens */}
      <div className="flex flex-wrap gap-3 p-4 bg-[#F3F4F6] rounded-xl">
        {tokens.map((t, i) => (
          <Token
            key={i}
            token={t.token}
            language={t.language}
            confidence={t.confidence}
            showConfidence={showConfidence}
          />
        ))}
      </div>
    </div>
  )
}
