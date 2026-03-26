import { useState, useEffect } from 'react'
import axios from 'axios'
import { Loader2, ArrowRight, X, Sparkles } from 'lucide-react'

const PLACEHOLDER = 'yaar ye movie bohot amazing thi, I loved it...'
const MAX_CHARS = 1000

export default function TextInput({ onAnalyze, loading }) {
  const [text, setText] = useState('')
  const [examples, setExamples] = useState([])

  useEffect(() => {
    axios
      .get('/api/examples')
      .then(({ data }) => setExamples(data.examples || []))
      .catch(() => {})
  }, [])

  const handleSubmit = (e) => {
    e.preventDefault()
    if (text.trim() && !loading) onAnalyze(text.trim())
  }

  return (
    <div className="card p-6 space-y-4">
      <form onSubmit={handleSubmit} className="space-y-3">
        <div className="relative">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value.slice(0, MAX_CHARS))}
            placeholder={PLACEHOLDER}
            rows={4}
            className="w-full resize-none rounded-xl border border-gray-200
                       bg-[#F3F4F6] text-[#111827]
                       placeholder-gray-400
                       p-4 text-sm leading-relaxed font-[Inter]
                       focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent
                       transition-colors"
          />
          <span className="absolute bottom-3 right-3 text-xs text-gray-400 font-medium">
            {text.length}/{MAX_CHARS}
          </span>
        </div>

        <div className="flex items-center justify-between gap-3">
          <button
            type="button"
            onClick={() => setText('')}
            className="btn-secondary flex items-center gap-1.5"
            disabled={!text || loading}
          >
            <X className="w-3.5 h-3.5" />
            Clear
          </button>
          <button
            type="submit"
            className="btn-primary flex items-center gap-2"
            disabled={!text.trim() || loading}
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Analyzing…
              </>
            ) : (
              <>
                Analyze
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </div>
      </form>

      {/* Example chips */}
      {examples.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-primary" />
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Try an example
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {examples.map((ex, i) => (
              <button
                key={i}
                onClick={() => {
                  setText(ex.text)
                  onAnalyze(ex.text)
                }}
                disabled={loading}
                title={ex.label}
                className="text-xs px-3 py-1.5 rounded-full border border-brand-200
                           text-primary bg-brand-50
                           hover:bg-brand-100 hover:border-brand-300
                           transition-colors disabled:opacity-50 disabled:cursor-not-allowed
                           max-w-[220px] truncate font-medium"
              >
                {ex.text}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
