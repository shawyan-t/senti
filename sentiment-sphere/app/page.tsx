"use client"

import React, { useState, useEffect, useRef } from "react"
import { Cloud, Loader2 } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"
import SentimizerTitle from "@/components/sentimizer-title"
import { analyzeText, analyzeFile, getAnalysis, Analysis } from "@/lib/api"
import { VisualizationDashboard } from "@/components/visualization-dashboard"
import { ProfessionalVisualizationDashboard } from "@/components/professional-visualization-dashboard"
import dynamic from 'next/dynamic';
const UMAP3DScatter = dynamic(() => import('@/components/charts/umap-3d-scatter').then(mod => mod.UMAP3DScatter), { ssr: false });

type AnalysisData = Record<string, any>

export default function Home() {
  const [activeTab, setActiveTab] = useState("ticker")
  const [text, setText] = useState("")
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isLoaded, setIsLoaded] = useState(false)
  const [titleAnimationComplete, setTitleAnimationComplete] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<Analysis | null>(null)
  const [tickerAnalysisResult, setTickerAnalysisResult] = useState<Analysis | null>(null)
  const [savedAnalysisResult, setSavedAnalysisResult] = useState<Analysis | null>(null)
  const [previousAnalyses, setPreviousAnalyses] = useState<Record<string, AnalysisData>>({})
  const [selectedAnalysisId, setSelectedAnalysisId] = useState("")
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [fileName, setFileName] = useState("")
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [saveStatus, setSaveStatus] = useState<string | null>(null)
  const [removeStatus, setRemoveStatus] = useState<string | null>(null)

  useEffect(() => {
    // Simulate loading delay for entrance animation
    const timer = setTimeout(() => {
      setIsLoaded(true)
    }, 500)

    return () => clearTimeout(timer)
  }, [])

  useEffect(() => {
    // Sync displayed analysis with active tab
    if (activeTab === "analyses") {
      setAnalysisResult(savedAnalysisResult)
      setError(null)
      loadPreviousAnalyses()
    } else if (activeTab === "about") {
      setAnalysisResult(null)
      setError(null)
    } else if (activeTab === "ticker") {
      setAnalysisResult(tickerAnalysisResult)
      setError(null)
    }
  }, [activeTab])

  const LOCAL_KEY = 'sentimizer_saved_analyses_v1'

  const getDisplayLabelFromAnalysis = (ar: any) => {
    if (ar?.query_summary?.query) return ar.query_summary.query
    if (ar?.source) return ar.source
    return 'Analysis'
  }

  const extractTickerFromLabel = (label: string) => {
    if (!label) return ''
    const idx = label.indexOf('(')
    const base = (idx > 0 ? label.slice(0, idx) : label).trim()
    const tok = base.split(' ')[0]
    return (tok || '').toUpperCase()
  }
  const loadPreviousAnalyses = () => {
    try {
      const raw = localStorage.getItem(LOCAL_KEY)
      const arr: any[] = raw ? JSON.parse(raw) : []
      const map: Record<string, AnalysisData> = {}
      arr.forEach((item: any) => {
        if (item && item.id) map[item.id] = item
      })
      setPreviousAnalyses(map)
      const keys = Object.keys(map)
      if (keys.length > 0) setSelectedAnalysisId((prev) => prev && map[prev] ? prev : keys[0])
      else setSelectedAnalysisId("")
    } catch (error) {
      console.error("Failed to load local analyses:", error)
      setPreviousAnalyses({})
    }
  }

  // Helper function to transform mathematical analysis data to UI format
  const transformMathematicalAnalysis = (result: Analysis) => {
    const compositeScore = result.mathematical_sentiment_analysis?.composite_score?.value || 0
    const sentiment = compositeScore > 0.1 ? 'positive' : compositeScore < -0.1 ? 'negative' : 'neutral'
    const confidenceInterval = result.mathematical_sentiment_analysis?.composite_score?.confidence_interval
    const dominantEmotions = result.emotion_vector_analysis?.dominant_emotions || []
    
    return {
      sentiment: {
        sentiment,
        score: compositeScore,
        rationale: `Mathematical analysis (${(result.mathematical_sentiment_analysis?.composite_score?.statistical_significance * 100 || 0).toFixed(1)}% confidence) with dominant emotions: ${dominantEmotions.join(', ')}`
      },
      analysis: result.enhanced_summary || 'No enhanced summary available.',
      metadata: {
        topics: dominantEmotions,
        confidence_interval: confidenceInterval,
        model_consensus: result.mathematical_sentiment_analysis?.multi_model_validation,
        emotion_entropy: result.emotion_vector_analysis?.emotion_entropy
      }
    }
  };

  const handleAnalyze = async () => {
    if (!text.trim()) return
    
    setIsAnalyzing(true)
    setError(null) // Clear previous errors
    try {
      const result = await analyzeText(text)
      setTickerAnalysisResult(result)
      if (activeTab === "ticker") setAnalysisResult(result)
    } catch (error) {
      console.error("Error analyzing text:", error)
      const errorMessage = error instanceof Error ? error.message : 'Analysis failed'
      setError(errorMessage)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setUploadedFile(file)
      setFileName(file.name)
    }
  }

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) {
      setUploadedFile(file)
      setFileName(file.name)
    }
  }

  const handleAnalyzeFile = async () => {
    if (!uploadedFile) return
    
    setIsAnalyzing(true)
    setError(null) // Clear previous errors
    try {
      const result = await analyzeFile(uploadedFile)
      setTickerAnalysisResult(result)
      if (activeTab === "ticker") setAnalysisResult(result)
    } catch (error) {
      console.error("Error analyzing file:", error)
      const errorMessage = error instanceof Error ? error.message : 'File analysis failed'
      setError(errorMessage)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleViewAnalysis = async () => {
    if (!selectedAnalysisId) return
    
    setIsAnalyzing(true)
    setError(null) // Clear previous errors
    try {
      const result = await getAnalysis(selectedAnalysisId)
      setSavedAnalysisResult(result)
      if (activeTab === "analyses") setAnalysisResult(result)
    } catch (error) {
      console.error("Error loading analysis:", error)
      const errorMessage = error instanceof Error ? error.message : 'Failed to load analysis'
      setError(errorMessage)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const saveCurrentAnalysisToLocal = () => {
    if (!analysisResult) return
    const id = (analysisResult as any).analysis_id || (analysisResult as any).id
    if (!id) return
    try {
      const raw = localStorage.getItem(LOCAL_KEY)
      const arr: any[] = raw ? JSON.parse(raw) : []
      const label = getDisplayLabelFromAnalysis(analysisResult as any)
      const meta = {
        id,
        source: label,
        ticker: extractTickerFromLabel(label),
        timestamp: (analysisResult as any).timestamp || new Date().toISOString(),
      }
      const exists = arr.some((x: any) => x.id === id)
      const next = exists ? arr.map((x: any) => x.id === id ? meta : x) : [...arr, meta]
      localStorage.setItem(LOCAL_KEY, JSON.stringify(next))
      if (activeTab === 'analyses') loadPreviousAnalyses()
      setSaveStatus('Saved')
      setTimeout(() => setSaveStatus(null), 1200)
    } catch (e) {
      console.error('Failed to save to local storage', e)
    }
  };

  const removeSavedAnalysisFromLocal = (id: string) => {
    try {
      const raw = localStorage.getItem(LOCAL_KEY)
      const arr: any[] = raw ? JSON.parse(raw) : []
      const next = arr.filter((x: any) => x.id !== id)
      localStorage.setItem(LOCAL_KEY, JSON.stringify(next))
      loadPreviousAnalyses()
      setRemoveStatus('Removed')
      setTimeout(() => setRemoveStatus(null), 1200)
    } catch (e) {
      console.error('Failed to remove from local storage', e)
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-b from-slate-900 via-purple-950 to-slate-900 text-gray-200">
      <AnimatePresence>
        {!isLoaded ? (
          <motion.div
            className="absolute inset-0 flex items-center justify-center bg-slate-900 z-50"
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
          >
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 1.2, opacity: 0 }}
              transition={{ duration: 0.5 }}
            >
              <Loader2 className="h-16 w-16 text-emerald-400 animate-spin" />
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>

      <main className="flex-1 container mx-auto px-4 py-8 max-w-6xl">
        {/* Title Section */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.8 }}
          className="mb-12 h-32 flex items-center justify-center"
        >
          <SentimizerTitle onAnimationComplete={() => setTitleAnimationComplete(true)} />
          <p className="absolute mt-32 text-center text-emerald-300 font-light tracking-wide">
            Stock Market Sentiment Analysis
          </p>
        </motion.div>

        {/* Main Content - Only show after title animation completes */}
        <AnimatePresence>
          {titleAnimationComplete && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
              <div className="bg-white/5 backdrop-blur-sm h-16 rounded-lg mb-8 shadow-lg shadow-purple-900/20"></div>

              <div className="w-full">
                <div className="grid grid-cols-3 mb-8 bg-slate-800/50 backdrop-blur-sm rounded-lg p-1">
                  {["ticker", "analyses", "about"].map((tab) => (
                    <button
                      key={tab}
                      onClick={() => setActiveTab(tab)}
                      className={cn(
                        "py-2 px-4 rounded-md transition-all duration-300",
                        activeTab === tab
                          ? "bg-gradient-to-r from-emerald-600 to-teal-500 text-white"
                          : "text-gray-400 hover:text-gray-200",
                      )}
                    >
                      {tab === "ticker" && "📈 Stock Ticker Analysis"}
                      {tab === "analyses" && "📊 Saved Analyses"}
                      {tab === "about" && "ℹ️ About"}
                    </button>
                  ))}
                </div>

                {activeTab === "ticker" && (
                  <div className="space-y-4">
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.5 }}
                      className="bg-slate-800/50 border border-amber-500/30 rounded-lg p-4 mb-4"
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <div className="w-3 h-3 bg-amber-400 rounded-full"></div>
                        <h3 className="text-amber-300 font-semibold">Stock Market Sentiment Analysis</h3>
                      </div>
                      <p className="text-sm text-gray-300">
                        This system analyzes <strong>NASDAQ and NYSE stock tickers only</strong>. 
                        Enter a valid ticker symbol (e.g., AAPL, NVDA, META, VOO) to get comprehensive sentiment analysis 
                        from financial news sources, analyst reports, and market discussions.
                      </p>
                    </motion.div>
                    
                    <motion.p
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.5 }}
                      className="text-emerald-300 font-medium"
                    >
                      Enter a NASDAQ or NYSE Stock Ticker:
                    </motion.p>
                    <div className="relative">
                      <input
                        type="text"
                        placeholder="Enter ticker symbol (e.g., AAPL, NVDA, META, VOO)..."
                        className="w-full h-14 bg-slate-800/70 backdrop-blur-sm border border-slate-700 focus:border-emerald-500 transition-all duration-300 rounded-lg px-4 text-lg font-mono uppercase placeholder:normal-case placeholder:font-sans"
                        value={text}
                        onChange={(e) => setText(e.target.value.toUpperCase())}
                        maxLength={5}
                      />
                      <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                        <span className="text-xs text-gray-500 bg-slate-700 px-2 py-1 rounded">
                          {text.length}/5
                        </span>
                      </div>
                    </div>
                    <div className="flex justify-between items-center">
                      <div className="space-y-1">
                        <p className="text-sm text-emerald-200/70">
                          Examples: AAPL (Apple), NVDA (NVIDIA), META (Meta), VOO (Vanguard S&P 500 ETF)
                        </p>
                        
                      </div>
                      <Button
                        onClick={handleAnalyze}
                        disabled={!text.trim() || isAnalyzing}
                        className={cn(
                          "bg-gradient-to-r from-emerald-600 to-teal-500 hover:from-emerald-500 hover:to-teal-400",
                          "text-white font-medium px-8 py-3 rounded-lg shadow-lg shadow-emerald-900/30",
                          "transition-all duration-300 transform hover:scale-105 active:scale-95",
                          "disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none",
                        )}
                      >
                        {isAnalyzing ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Analyzing {text}...
                          </>
                        ) : (
                          <>
                            📈 Analyze {text || "Ticker"}
                          </>
                        )}
                      </Button>
                    </div>
                  </div>
                )}


                {activeTab === "analyses" && (
                  <div className="space-y-4">
                    <motion.p
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.5 }}
                      className="text-emerald-300 font-medium"
                    >
                      Select a previous analysis to view:
                    </motion.p>
                    <div className="relative">
                      <select
                        className="w-full bg-slate-800/70 backdrop-blur-sm border-slate-700 rounded-lg p-3 appearance-none text-emerald-100 focus:border-emerald-500 transition-all duration-300"
                        disabled={Object.keys(previousAnalyses).length === 0}
                        value={selectedAnalysisId}
                        onChange={(e) => setSelectedAnalysisId(e.target.value)}
                      >
                        {Object.keys(previousAnalyses).length === 0 ? (
                          <option>No previous analyses</option>
                        ) : (
                          Object.entries(previousAnalyses).map(([id, data]) => (
                            <option key={id} value={id}>
                              {data.source || "Unknown"} ({new Date(data.timestamp).toLocaleString()})
                            </option>
                          ))
                        )}
                      </select>
                      <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-emerald-400">
                        <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                          <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
                        </svg>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <Button
                        className={cn(
                          "bg-gradient-to-r from-emerald-600 to-teal-500 hover:from-emerald-500 hover:to-teal-400",
                          "text-white font-medium px-6 py-2 rounded-lg shadow-lg shadow-emerald-900/30",
                          "transition-all duration-300 transform hover:scale-105 active:scale-95",
                          "disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none",
                        )}
                        disabled={Object.keys(previousAnalyses).length === 0 || !selectedAnalysisId || isAnalyzing}
                        onClick={handleViewAnalysis}
                      >
                        {isAnalyzing ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Loading...
                          </>
                        ) : (
                          "View Analysis"
                        )}
                      </Button>
                      <Button
                        variant="secondary"
                        className="text-sm bg-slate-700/60 border border-slate-600 hover:bg-slate-700"
                        disabled={!selectedAnalysisId}
                        onClick={() => selectedAnalysisId && removeSavedAnalysisFromLocal(selectedAnalysisId)}
                      >
                        Remove from Saved
                      </Button>
                      {removeStatus && (
                        <span className="text-sm text-emerald-300">{removeStatus}</span>
                      )}
                    </div>
                  </div>
                )}

                {activeTab === "about" && (
                  <div className="space-y-6">
                    {/* About Sentimizer */}
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.4 }}
                      className="bg-slate-800/50 border border-emerald-500/20 rounded-lg p-6"
                    >
                      <h3 className="text-xl font-semibold text-emerald-300 mb-3">About Sentimizer</h3>
                      <p className="text-gray-300 mb-3">
                        Sentimizer analyzes real market discussion and news to quantify sentiment and surface trends with transparent, math‑based visuals.
                      </p>
                      
                    </motion.div>

                    {/* How It's Calculated */}
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.05, duration: 0.4 }}
                      className="bg-slate-800/50 border border-emerald-500/20 rounded-lg p-6"
                    >
                      <h3 className="text-xl font-semibold text-emerald-300 mb-3">How It’s Calculated</h3>
                      <div className="space-y-3 text-sm text-gray-300">
                        <div>
                          <span className="font-medium text-emerald-200">Per‑source sentiment:</span>
                          <span className="ml-2 block text-gray-300">
                            s = mean([VADER, TextBlob, AFINN, RoBERTa]) in [-1, 1]. Combine lexicon + transformer into one per‑source score.
                          </span>
                        </div>
                        <div>
                          <span className="font-medium text-emerald-200">Corrections:</span>
                          <span className="ml-2 block text-gray-300">
                            s' = s * (1 − lambda_tox * p_tox) − gamma_sarc * p_sarc * sign(s). Short: dampen toxicity; flip/attenuate sarcasm.
                          </span>
                        </div>
                        <div>
                          <span className="font-medium text-emerald-200">Weights:</span>
                          <span className="ml-2 block text-gray-300">
                            w_final = w_fresh * w_domain * w_engage * w_retrieval * w_lang, with w_fresh = exp(−delta_t / tau). Short: fresher, credible, high‑quality, relevant, engaged sources count more.
                          </span>
                        </div>
                        <div>
                          <span className="font-medium text-emerald-200">Aggregation:</span>
                          <span className="ml-2 block text-gray-300">
                            S = Tukey biweight over s&apos; with weights w_final. Short: robust weighted mean to reduce outlier impact.
                          </span>
                        </div>
                        <div>
                          <span className="font-medium text-emerald-200">Uncertainty:</span>
                          <span className="ml-2 block text-gray-300">
                            Mean CI via bootstrap; proportions via Wilson intervals. Short: honest intervals for averages and shares.
                          </span>
                        </div>
                        <div>
                          <span className="font-medium text-emerald-200">VAD (valence/arousal/dominance):</span>
                          <span className="ml-2 block text-gray-300">
                            (v, a, d) in [-1,1]^3 from emotion probabilities and s. Short: map emotion mix and sentiment to VAD.
                          </span>
                        </div>
                        <div>
                          <span className="font-medium text-emerald-200">Timeline:</span>
                          <span className="ml-2 block text-gray-300">
                            Rolling mean(s) over dated s' plus OLS trend y = a + b * t. Short: smooth trajectories and show trend line.
                          </span>
                        </div>
                        <div>
                          <span className="font-medium text-emerald-200">3D UMAP:</span>
                          <span className="ml-2 block text-gray-300">
                            Embed(text) → UMAP 3D; color = sentiment; (optional) size/opacity = weight/recency. Short: cluster topics and sentiment in 3D.
                          </span>
                        </div>
                      </div>
                    </motion.div>

                    {/* Purpose */}
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.1, duration: 0.4 }}
                      className="bg-slate-800/50 border border-emerald-500/20 rounded-lg p-6"
                    >
                      <h3 className="text-xl font-semibold text-emerald-300 mb-3">Purpose</h3>
                      <p className="text-gray-300">
                        Provide a clear, math‑driven view of market sentiment: no synthetic data, transparent methods, and visuals that make source‑level
                        evidence, uncertainty, and trends easy to inspect.
                      </p>
                    </motion.div>
                  </div>
                )}
              </div>

              {/* Error Display */}
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="mb-4"
                >
                  <div className="bg-red-900/20 border border-red-700/50 rounded-lg p-4">
                    <div className="flex items-start gap-3">
                      <div className="text-red-400 mt-0.5">⚠️</div>
                      <div>
                        <h3 className="text-red-300 font-medium mb-1">
                          {error.includes("503") ? "Service Temporarily Unavailable" : 
                           error.includes("rate limit") ? "Rate Limit Reached" : 
                           "Analysis Failed"}
                        </h3>
                        <div className="text-red-200 text-sm leading-relaxed">
                          {error.includes("Both NewsAPI and Google Search API are currently unavailable") ? (
                            <>
                              <strong>All Data Sources Unavailable:</strong> Both our primary data sources (NewsAPI and Google Search API) 
                              are currently unavailable. This is rare and usually indicates API rate limiting on both services.
                              <br /><br />
                              <strong>What's happening:</strong> Our system tries multiple fallbacks:
                              <ol className="list-decimal list-inside mt-2 space-y-1">
                                <li>First: Use both NewsAPI + Google Search together</li>
                                <li>Fallback: Use whichever API is available</li>
                                <li>Final: Show this error only when both fail</li>
                              </ol>
                              <br />
                              <strong>Solutions:</strong>
                              <ul className="list-disc list-inside mt-2 space-y-1">
                                <li>Wait 3-5 minutes for API limits to reset</li>
                                <li>Try a different, more popular ticker (AAPL, MSFT, GOOGL)</li>
                                <li>Check back later - limits reset daily</li>
                              </ul>
                            </>
                          ) : error.includes("Google Search API is rate limited") ? (
                            <>
                              <strong>Partial Service:</strong> NewsAPI is working, but Google Search API is rate limited. 
                              You should still get some results from news sources.
                              <br /><br />
                              <strong>What's happening:</strong> We use both NewsAPI and Google Search API for comprehensive coverage. 
                              When one fails, we fall back to the other automatically.
                              <br /><br />
                              <strong>Solutions:</strong>
                              <ul className="list-disc list-inside mt-2 space-y-1">
                                <li>Try the analysis anyway - NewsAPI may provide sufficient data</li>
                                <li>Wait 2-3 minutes for Google Search limits to reset</li>
                                <li>Popular tickers get better NewsAPI coverage</li>
                              </ul>
                            </>
                          ) : error.includes("503") ? (
                            <>
                              <strong>Service Unavailable:</strong> The financial data services are currently unavailable. 
                              This may be due to API rate limiting or temporary service issues.
                              <br /><br />
                              <strong>What to do:</strong> Please wait a few minutes and try again. 
                              If the problem persists, the service may be experiencing high demand.
                            </>
                          ) : (
                            <>
                              <strong>Error:</strong> {error}
                              <br /><br />
                              <strong>What to try:</strong>
                              <ul className="list-disc list-inside mt-2 space-y-1">
                                <li>Check that you entered a valid NASDAQ/NYSE ticker (e.g., AAPL, NVDA, META)</li>
                                <li>Wait a moment and try again</li>
                                <li>Try a different ticker symbol</li>
                              </ul>
                            </>
                          )}
                        </div>
                        <button 
                          onClick={() => setError(null)}
                          className="mt-3 text-red-300 hover:text-red-200 text-sm underline"
                        >
                          Dismiss
                        </button>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Analysis Result Display */}
              {(activeTab === 'ticker' || activeTab === 'analyses') && analysisResult && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2, duration: 0.5 }}
                  className="mt-8 bg-slate-800/70 backdrop-blur-sm rounded-lg p-6 border border-emerald-500/20"
                >
                  <div className="flex items-center justify-end mb-2 gap-3">
                    <Button
                      variant="secondary"
                      className="text-sm bg-slate-700/60 border border-slate-600 hover:bg-slate-700"
                      onClick={saveCurrentAnalysisToLocal}
                    >
                      Save Analysis
                    </Button>
                    {saveStatus && (
                      <span className="text-sm text-emerald-300">{saveStatus}</span>
                    )}
                  </div>
                  <h2 className="text-2xl font-bold text-emerald-400 mb-4">Analysis Results</h2>
                  
                  {/* Comprehensive Sentiment Analysis Results */}
                  <div className="mb-6">
                    <h3 className="text-xl font-semibold text-emerald-300 mb-2">Comprehensive Sentiment Analysis</h3>
                    
                    {/* Core Sentiment */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                      <div className="bg-slate-700/50 p-3 rounded-lg">
                        <div className="text-sm text-gray-400">Sentiment Score</div>
                        <div className={`text-lg font-bold ${
                          (analysisResult.comprehensive_metrics?.sentiment?.score || 0) > 0.1 ? "text-green-500" : 
                          (analysisResult.comprehensive_metrics?.sentiment?.score || 0) < -0.1 ? "text-red-500" : 
                          "text-yellow-500"
                        }`}>
                          {(analysisResult.comprehensive_metrics?.sentiment?.score || analysisResult.mathematical_sentiment_analysis?.composite_score?.value || 0).toFixed(3)}
                        </div>
                        <div className="text-xs text-gray-400">Range: [-1, +1]</div>
                      </div>
                      
                      <div className="bg-slate-700/50 p-3 rounded-lg">
                        <div className="text-sm text-gray-400">Confidence</div>
                        <div className="text-lg font-bold text-blue-400">
                          {((analysisResult.comprehensive_metrics?.sentiment?.confidence || analysisResult.mathematical_sentiment_analysis?.composite_score?.statistical_significance || 0) * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-400">Statistical significance</div>
                      </div>
                      
                      <div className="bg-slate-700/50 p-3 rounded-lg">
                        <div className="text-sm text-gray-400">Disagreement</div>
                        <div className="text-lg font-bold text-purple-400">
                          {((analysisResult.comprehensive_metrics?.disagreement_index || analysisResult.mathematical_sentiment_analysis?.uncertainty_metrics?.polarization_index || 0) * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-400">Polarization measure</div>
                      </div>
                    </div>

                    {/* Polarity Breakdown with Wilson CIs */}
                    {analysisResult.comprehensive_metrics?.polarity_breakdown && (
                      <div className="mb-4">
                        <div className="text-sm font-medium text-emerald-200 mb-2">Polarity Distribution (Wilson 95% CI)</div>
                        <div className="grid grid-cols-3 gap-2 text-sm">
                          <div className="bg-green-900/30 p-2 rounded">
                            <div className="text-green-300">Positive: {(analysisResult.comprehensive_metrics.polarity_breakdown.positive * 100).toFixed(1)}%</div>
                            {analysisResult.comprehensive_metrics.polarity_breakdown.wilson_ci?.positive && (
                              <div className="text-xs text-gray-400">
                                CI: [{(analysisResult.comprehensive_metrics.polarity_breakdown.wilson_ci.positive[0] * 100).toFixed(1)}%, {(analysisResult.comprehensive_metrics.polarity_breakdown.wilson_ci.positive[1] * 100).toFixed(1)}%]
                              </div>
                            )}
                          </div>
                          <div className="bg-red-900/30 p-2 rounded">
                            <div className="text-red-300">Negative: {(analysisResult.comprehensive_metrics.polarity_breakdown.negative * 100).toFixed(1)}%</div>
                            {analysisResult.comprehensive_metrics.polarity_breakdown.wilson_ci?.negative && (
                              <div className="text-xs text-gray-400">
                                CI: [{(analysisResult.comprehensive_metrics.polarity_breakdown.wilson_ci.negative[0] * 100).toFixed(1)}%, {(analysisResult.comprehensive_metrics.polarity_breakdown.wilson_ci.negative[1] * 100).toFixed(1)}%]
                              </div>
                            )}
                          </div>
                          <div className="bg-yellow-900/30 p-2 rounded">
                            <div className="text-yellow-300">Neutral: {(analysisResult.comprehensive_metrics.polarity_breakdown.neutral * 100).toFixed(1)}%</div>
                            {analysisResult.comprehensive_metrics.polarity_breakdown.wilson_ci?.neutral && (
                              <div className="text-xs text-gray-400">
                                CI: [{(analysisResult.comprehensive_metrics.polarity_breakdown.wilson_ci.neutral[0] * 100).toFixed(1)}%, {(analysisResult.comprehensive_metrics.polarity_breakdown.wilson_ci.neutral[1] * 100).toFixed(1)}%]
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Tone & Style Analysis */}
                  {analysisResult.comprehensive_metrics?.tone && (
                    <div className="mb-6">
                      <h3 className="text-xl font-semibold text-emerald-300 mb-2">Tone & Style Analysis</h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div className="bg-slate-700/50 p-3 rounded-lg">
                          <div className="text-sm text-gray-400">Subjectivity</div>
                          <div className="text-lg font-bold text-indigo-400">
                            {(analysisResult.comprehensive_metrics.tone.subjectivity * 100).toFixed(0)}%
                          </div>
                        </div>
                        <div className="bg-slate-700/50 p-3 rounded-lg">
                          <div className="text-sm text-gray-400">Politeness</div>
                          <div className="text-lg font-bold text-pink-400">
                            {(analysisResult.comprehensive_metrics.tone.politeness * 100).toFixed(0)}%
                          </div>
                        </div>
                        <div className="bg-slate-700/50 p-3 rounded-lg">
                          <div className="text-sm text-gray-400">Formality</div>
                          <div className="text-lg font-bold text-cyan-400">
                            {(analysisResult.comprehensive_metrics.tone.formality * 100).toFixed(0)}%
                          </div>
                        </div>
                        <div className="bg-slate-700/50 p-3 rounded-lg">
                          <div className="text-sm text-gray-400">Assertiveness</div>
                          <div className="text-lg font-bold text-orange-400">
                            {(analysisResult.comprehensive_metrics.tone.assertiveness * 100).toFixed(0)}%
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Special Metrics */}
                  <div className="mb-6">
                    <h3 className="text-xl font-semibold text-emerald-300 mb-2">Risk & Quality Metrics</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      <div className="bg-slate-700/50 p-3 rounded-lg">
                        <div className="text-sm text-gray-400">Sarcasm Rate</div>
                        <div className="text-lg font-bold text-yellow-400">
                          {((analysisResult.comprehensive_metrics?.sarcasm_rate || 0) * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="bg-slate-700/50 p-3 rounded-lg">
                        <div className="text-sm text-gray-400">Toxicity Rate</div>
                        <div className="text-lg font-bold text-red-400">
                          {((analysisResult.comprehensive_metrics?.toxicity_rate || 0) * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="bg-slate-700/50 p-3 rounded-lg">
                        <div className="text-sm text-gray-400">Freshness Score</div>
                        <div className="text-lg font-bold text-green-400">
                          {((analysisResult.comprehensive_metrics?.freshness_score || 0) * 100).toFixed(0)}%
                        </div>
                      </div>
                      <div className="bg-slate-700/50 p-3 rounded-lg">
                        <div className="text-sm text-gray-400">Evidence Weight</div>
                        <div className="text-lg font-bold text-blue-400">
                          {(analysisResult.comprehensive_metrics?.total_evidence_weight || 0).toFixed(2)}
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Enhanced Summary */}
                  <div className="mb-6">
                    <h3 className="text-xl font-semibold text-emerald-300 mb-2">Enhanced Analysis Summary</h3>
                    <p className="text-gray-300">
                      {analysisResult.enhanced_summary || "No enhanced summary available."}
                    </p>
                  </div>

                  {/* Evidence & Source Breakdown */}
                  {analysisResult.explanatory_clusters && analysisResult.explanatory_clusters.length > 0 && (
                    <div className="mb-6">
                      <h3 className="text-xl font-semibold text-emerald-300 mb-2">Evidence Sources & Quality</h3>
                      {analysisResult.explanatory_clusters.map((cluster: any, i: number) => (
                        <div key={i} className="bg-slate-700/50 p-3 rounded-lg mb-3">
                          <div className="flex justify-between items-start mb-2">
                            <div className="text-sm font-medium text-emerald-200">
                              Cluster {i + 1} (Weight: {cluster.weight?.toFixed(3)})
                            </div>
                            <div className="text-xs text-gray-400">
                              {cluster.unit_count} units
                            </div>
                          </div>
                          <div className="text-sm text-gray-300 mb-2">
                            {cluster.representative_text}
                          </div>
                          {cluster.why_important && cluster.why_important.length > 0 && (
                            <div className="flex flex-wrap gap-1">
                              {cluster.why_important.map((reason: string, j: number) => (
                                <span key={j} className="text-xs bg-green-900/50 text-green-300 px-2 py-1 rounded">
                                  {reason}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* VAD Analysis */}
                  {analysisResult.vad_analysis && (
                    <div className="mb-6">
                      <h3 className="text-xl font-semibold text-emerald-300 mb-2">VAD Emotional Dimensions</h3>
                      <div className="grid grid-cols-3 gap-3">
                        <div className="bg-slate-700/50 p-3 rounded-lg">
                          <div className="text-sm text-gray-400">Valence</div>
                          <div className="text-lg font-bold text-purple-400">
                            {(analysisResult.vad_analysis.valence * 100).toFixed(0)}%
                          </div>
                          <div className="text-xs text-gray-400">Pleasure/Displeasure</div>
                        </div>
                        <div className="bg-slate-700/50 p-3 rounded-lg">
                          <div className="text-sm text-gray-400">Arousal</div>
                          <div className="text-lg font-bold text-red-400">
                            {(analysisResult.vad_analysis.arousal * 100).toFixed(0)}%
                          </div>
                          <div className="text-xs text-gray-400">Activation/Calm</div>
                        </div>
                        <div className="bg-slate-700/50 p-3 rounded-lg">
                          <div className="text-sm text-gray-400">Dominance</div>
                          <div className="text-lg font-bold text-yellow-400">
                            {(analysisResult.vad_analysis.dominance * 100).toFixed(0)}%
                          </div>
                          <div className="text-xs text-gray-400">Control/Submission</div>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {/* Metadata/Topics */}
                  {analysisResult.metadata && (
                    <div className="mb-8">
                      <h3 className="text-xl font-semibold text-emerald-300 mb-2">Topics & Metadata</h3>
                      
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {/* Topics */}
                        <div className="bg-slate-700/50 p-4 rounded-lg">
                          <h4 className="font-medium text-emerald-200 mb-2">Topics</h4>
                          <div className="flex flex-wrap gap-2">
                            {analysisResult.metadata.topics?.length > 0 ? (
                              analysisResult.metadata.topics.map((topic: string, i: number) => (
                                <span key={i} className="px-2 py-1 bg-emerald-900/50 text-emerald-300 rounded-full text-sm">
                                  {topic}
                                </span>
                              ))
                            ) : (
                              <span className="text-gray-400">No topics identified</span>
                            )}
                          </div>
                        </div>
                        
                        {/* Regions */}
                        <div className="bg-slate-700/50 p-4 rounded-lg">
                          <h4 className="font-medium text-emerald-200 mb-2">Regions</h4>
                          <div className="flex flex-wrap gap-2">
                            {analysisResult.metadata.regions?.length > 0 ? (
                              analysisResult.metadata.regions.map((region: string, i: number) => (
                                <span key={i} className="px-2 py-1 bg-blue-900/50 text-blue-300 rounded-full text-sm">
                                  {region}
                                </span>
                              ))
                            ) : (
                              <span className="text-gray-400">No regions identified</span>
                            )}
                          </div>
                        </div>
                        
                        {/* Entities/Keywords */}
                        <div className="bg-slate-700/50 p-4 rounded-lg">
                          <h4 className="font-medium text-emerald-200 mb-2">Entities/Keywords</h4>
                          <div className="flex flex-wrap gap-2">
                            {analysisResult.metadata.entities?.length > 0 ? (
                              analysisResult.metadata.entities.map((entity: string, i: number) => (
                                <span key={i} className="px-2 py-1 bg-purple-900/50 text-purple-300 rounded-full text-sm">
                                  {entity}
                                </span>
                              ))
                            ) : (
                              <span className="text-gray-400">No entities identified</span>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Query Summary and Recent Events */}
                  {analysisResult.query_summary && (
                    <div className="mb-8">
                      <h3 className="text-xl font-semibold text-emerald-300 mb-4">Query Analysis & Recent Context</h3>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                        <div className="bg-slate-700/50 p-4 rounded-lg">
                          <h4 className="font-medium text-emerald-200 mb-2">Query Type</h4>
                          <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                            analysisResult.query_summary.query_type === 'financial' 
                              ? 'bg-amber-900/50 text-amber-300' 
                              : 'bg-blue-900/50 text-blue-300'
                          }`}>
                            {analysisResult.query_summary.query_type === 'financial' ? '💰 Financial/Market' : '📝 General'}
                          </span>
                          <p className="text-sm text-gray-300 mt-2">"{analysisResult.query_summary.query}"</p>
                        </div>
                        
                        <div className="bg-slate-700/50 p-4 rounded-lg">
                          <h4 className="font-medium text-emerald-200 mb-2">Entities Detected</h4>
                          <div className="flex flex-wrap gap-2">
                            {analysisResult.query_summary.entities_detected?.length > 0 ? (
                              analysisResult.query_summary.entities_detected.map((entity: string, i: number) => (
                                <span key={i} className="px-2 py-1 bg-purple-900/50 text-purple-300 rounded-full text-sm">
                                  {entity}
                                </span>
                              ))
                            ) : (
                              <span className="text-gray-400">No specific entities detected</span>
                            )}
                          </div>
                        </div>
                      </div>
                      
                      <div className="bg-slate-700/50 p-4 rounded-lg">
                        <h4 className="font-medium text-emerald-200 mb-3">Recent Events Context</h4>
                        {analysisResult.query_summary.recent_events?.map((event: any, i: number) => (
                          <div key={i} className="flex items-center justify-between py-2 border-b border-slate-600 last:border-b-0">
                            <div className="flex items-center">
                              <span className={`w-2 h-2 rounded-full mr-3 ${
                                event.relevance === 'high' ? 'bg-green-400' : 
                                event.relevance === 'low' ? 'bg-yellow-400' : 'bg-gray-400'
                              }`}></span>
                              <span className="text-sm text-gray-300">{event.event}</span>
                            </div>
                            <span className="text-xs text-gray-400">
                              {event.source_count} sources
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Source Citations and Evidence */}
                  {analysisResult.source_citations && (
                    <div className="mb-8">
                      <h3 className="text-xl font-semibold text-emerald-300 mb-4">Source Citations & Evidence</h3>
                      
                      <div className="space-y-3">
                        {analysisResult.source_citations.map((citation: any, i: number) => (
                          <div key={i} className="bg-slate-700/50 p-4 rounded-lg border-l-4 border-emerald-500">
                            <div className="flex justify-between items-start mb-2">
                              <div className="flex items-center gap-2">
                                <span className="font-medium text-emerald-200">
                                  {citation.source === 'user_input' ? '👤 User Input' : `🌐 ${citation.source}`}
                                </span>
                                <span className="px-2 py-1 bg-emerald-900/50 text-emerald-300 rounded text-xs">
                                  Weight: {citation.contribution_weight}
                                </span>
                              </div>
                              <span className="text-xs text-gray-400">{citation.recency}</span>
                            </div>
                            <p className="text-sm text-gray-300 mb-2">"{citation.text_sample}"</p>
                            {citation.url && (
                              <a href={citation.url} target="_blank" rel="noopener noreferrer" 
                                 className="text-xs text-blue-400 hover:text-blue-300 underline">
                                View Source →
                              </a>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Calculation Methodology and Accuracy */}
                  {analysisResult.calculation_methodology && (
                    <div className="mb-8">
                      <h3 className="text-xl font-semibold text-emerald-300 mb-4">How Scores Are Calculated</h3>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                        <div className="bg-slate-700/50 p-4 rounded-lg">
                          <h4 className="font-medium text-emerald-200 mb-2">🧮 Sentiment Calculation</h4>
                          <p className="text-sm text-gray-300">{analysisResult.calculation_methodology.sentiment_calculation}</p>
                        </div>
                        
                        <div className="bg-slate-700/50 p-4 rounded-lg">
                          <h4 className="font-medium text-emerald-200 mb-2">🎯 Confidence Basis</h4>
                          <p className="text-sm text-gray-300">{analysisResult.calculation_methodology.confidence_basis}</p>
                        </div>
                      </div>
                      
                      <div className="bg-slate-700/50 p-4 rounded-lg mb-4">
                        <h4 className="font-medium text-emerald-200 mb-2">⚖️ Source Weighting</h4>
                        <p className="text-sm text-gray-300">{analysisResult.calculation_methodology.source_weighting}</p>
                      </div>
                      
                      <div className="bg-slate-700/50 p-4 rounded-lg">
                        <h4 className="font-medium text-emerald-200 mb-3">📊 Accuracy Indicators</h4>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div className="text-center">
                            <div className="text-lg font-bold text-emerald-300">
                              {analysisResult.calculation_methodology.accuracy_indicators.model_agreement}
                            </div>
                            <div className="text-xs text-gray-400">Model Agreement</div>
                          </div>
                          <div className="text-center">
                            <div className="text-lg font-bold text-emerald-300">
                              {analysisResult.calculation_methodology.accuracy_indicators.source_diversity}
                            </div>
                            <div className="text-xs text-gray-400">Source Diversity</div>
                          </div>
                          <div className="text-center">
                            <div className="text-lg font-bold text-emerald-300">
                              {analysisResult.calculation_methodology.accuracy_indicators.temporal_coverage}
                            </div>
                            <div className="text-xs text-gray-400">Temporal Coverage</div>
                          </div>
                          <div className="text-center">
                            <div className={`text-lg font-bold ${
                              analysisResult.calculation_methodology.accuracy_indicators.data_quality === 'high' ? 'text-green-300' :
                              analysisResult.calculation_methodology.accuracy_indicators.data_quality === 'medium' ? 'text-yellow-300' :
                              'text-red-300'
                            }`}>
                              {analysisResult.calculation_methodology.accuracy_indicators.data_quality.toUpperCase()}
                            </div>
                            <div className="text-xs text-gray-400">Data Quality</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Professional Financial Visualization Dashboard */}
                  {analysisResult.visualizations ? (
                    <ProfessionalVisualizationDashboard 
                      visualizations={analysisResult.visualizations}
                    />
                  ) : (
                    /* Fallback to old dashboard if no professional visualizations */
                    <div className="bg-slate-800/70 backdrop-blur-sm rounded-lg p-6 border border-emerald-500/20 text-center">
                      <p className="text-gray-400">No visualization data available.</p>
                    </div>
                  )}

                  {/* 3D UMAP Visualization */}
                  <div className="mt-12">
                    <h3 className="text-xl font-semibold text-emerald-300 mb-4">3D UMAP Topic/Sentiment Landscape</h3>
                    {Array.isArray((analysisResult as any).embeddings) && (analysisResult as any).embeddings.length > 0 ? (
                      <UMAP3DScatter
                        embeddings={(analysisResult as any).embeddings}
                        labels={(analysisResult as any).embedding_labels}
                        colors={(analysisResult as any).embedding_colors}
                        hovertexts={(analysisResult as any).embedding_hovertexts}
                      />
                    ) : (
                      <div className="bg-slate-800/70 backdrop-blur-sm rounded-lg p-6 border border-emerald-500/20 text-center">
                        <p className="text-gray-400">No embedding data available for UMAP visualization.</p>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3, duration: 0.8 }}
                className="bg-gradient-to-br from-slate-800/80 to-purple-900/80 backdrop-blur-md rounded-xl p-10 mt-12 text-center border border-emerald-500/20 shadow-2xl shadow-emerald-500/10"
              >
                <motion.h2
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-3xl font-light tracking-wide text-white"
                >
                  Powerful Sentiment Analysis for Everyone
                </motion.h2>
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.2 }}
                  className="mt-4 text-emerald-200/80 max-w-3xl mx-auto"
                >
                  From news articles to financial reports, Sentimizer uses AI and advanced math to interpret sentiment and visualize trends.
                </motion.p>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <footer className="mt-8 pb-8 text-center text-sm text-emerald-200/30">
        <p suppressHydrationWarning>© 2025 Sentimizer • Powered by OpenAI's GPT models</p>
      </footer>
    </div>
  )
}
