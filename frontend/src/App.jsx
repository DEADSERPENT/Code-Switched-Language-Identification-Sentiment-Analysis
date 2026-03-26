import { useState, useCallback } from 'react'
import axios from 'axios'
import { Zap } from 'lucide-react'
import TextInput from './components/TextInput'
import AnalysisPanel from './components/AnalysisPanel'

const API_BASE = '/api'

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const analyze = useCallback(async (text) => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const { data } = await axios.post(`${API_BASE}/analyze`, { text })
      setResult({ ...data, inputText: text })
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.message ||
        'Failed to connect to the backend. Make sure the FastAPI server is running on port 8000.'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }, [])

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <header className="bg-white border-b border-gray-100 sticky top-0 z-10 shadow-sm">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center shadow-sm">
              <Zap className="w-5 h-5 text-white" strokeWidth={2.5} />
            </div>
            <div>
              <h1 className="font-bold text-[#111827] text-base leading-tight tracking-tight">
                CoSwitchNLP
              </h1>
              <p className="text-xs text-gray-400 font-medium">
                Hinglish Language ID & Sentiment
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-5xl mx-auto px-6 py-10 space-y-7">
        {/* Hero */}
        <div className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 text-xs font-semibold text-primary bg-brand-50 border border-brand-200 px-3 py-1.5 rounded-full mb-1">
            <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
            XLM-RoBERTa · Fine-tuned on SentiMix
          </div>
          <h2 className="text-3xl font-bold text-[#111827] tracking-tight">
            Analyze Code-Switched Hinglish Text
          </h2>
          <p className="text-gray-500 max-w-lg mx-auto text-sm leading-relaxed">
            Token-level language identification and sentiment analysis for
            Hinglish text using a joint XLM-RoBERTa model.
          </p>
        </div>

        {/* Input */}
        <TextInput onAnalyze={analyze} loading={loading} />

        {/* Error */}
        {error && (
          <div className="card p-4 border-red-200 bg-red-50 animate-fade-in">
            <p className="text-red-600 text-sm font-medium">{error}</p>
          </div>
        )}

        {/* Loading skeleton */}
        {loading && (
          <div className="card p-6 animate-pulse-soft">
            <div className="flex flex-wrap gap-2">
              {[80, 60, 100, 70, 55, 90, 65, 75].map((w, i) => (
                <div
                  key={i}
                  className="h-10 rounded-lg bg-[#F3F4F6]"
                  style={{ width: `${w}px` }}
                />
              ))}
            </div>
          </div>
        )}

        {/* Results */}
        {result && !loading && <AnalysisPanel result={result} />}

        {/* Footer */}
        <footer className="text-center text-xs text-gray-400 pt-4 pb-10 space-y-1">
          <p>Fine-tuned on SemEval-2020 Task 9 SentiMix (Hinglish) · Base: XLM-RoBERTa</p>
          <p>Runs fully locally — no data leaves your machine</p>
        </footer>
      </main>
    </div>
  )
}
