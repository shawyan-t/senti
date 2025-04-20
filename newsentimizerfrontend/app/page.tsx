"use client"

import { useState, useEffect, useRef } from "react"
import { Cloud, Loader2 } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"
import SentimizerTitle from "@/components/sentimizer-title"
import { analyzeText, analyzeFile, getAnalyses, getAnalysis, Analysis } from "@/lib/api"

type AnalysisData = Record<string, any>

export default function Home() {
  const [activeTab, setActiveTab] = useState("text")
  const [text, setText] = useState("")
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isLoaded, setIsLoaded] = useState(false)
  const [titleAnimationComplete, setTitleAnimationComplete] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<Analysis | null>(null)
  const [previousAnalyses, setPreviousAnalyses] = useState<Record<string, AnalysisData>>({})
  const [selectedAnalysisId, setSelectedAnalysisId] = useState("")
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [fileName, setFileName] = useState("")
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    // Simulate loading delay for entrance animation
    const timer = setTimeout(() => {
      setIsLoaded(true)
    }, 500)

    return () => clearTimeout(timer)
  }, [])

  useEffect(() => {
    // Load previous analyses when the tab is selected
    if (activeTab === "analyses") {
      loadPreviousAnalyses()
    }
  }, [activeTab])

  const loadPreviousAnalyses = async () => {
    try {
      const analyses = await getAnalyses()
      setPreviousAnalyses(analyses)
      if (Object.keys(analyses).length > 0) {
        setSelectedAnalysisId(Object.keys(analyses)[0])
      }
    } catch (error) {
      console.error("Failed to load analyses:", error)
    }
  }

  const handleAnalyze = async () => {
    if (!text.trim()) return
    
    setIsAnalyzing(true)
    try {
      const result = await analyzeText(text)
      setAnalysisResult(result)
    } catch (error) {
      console.error("Error analyzing text:", error)
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
    try {
      const result = await analyzeFile(uploadedFile)
      setAnalysisResult(result)
    } catch (error) {
      console.error("Error analyzing file:", error)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleViewAnalysis = async () => {
    if (!selectedAnalysisId) return
    
    setIsAnalyzing(true)
    try {
      const result = await getAnalysis(selectedAnalysisId)
      setAnalysisResult(result)
    } catch (error) {
      console.error("Error loading analysis:", error)
    } finally {
      setIsAnalyzing(false)
    }
  }

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
            Advanced AI-Powered Sentiment Analysis Platform
          </p>
        </motion.div>

        {/* Main Content - Only show after title animation completes */}
        <AnimatePresence>
          {titleAnimationComplete && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
              <div className="bg-white/5 backdrop-blur-sm h-16 rounded-lg mb-8 shadow-lg shadow-purple-900/20"></div>

              <div className="w-full">
                <div className="grid grid-cols-3 mb-8 bg-slate-800/50 backdrop-blur-sm rounded-lg p-1">
                  {["text", "upload", "analyses"].map((tab) => (
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
                      {tab === "text" && "Text/URL Input"}
                      {tab === "upload" && "Upload File"}
                      {tab === "analyses" && "My Analyses"}
                    </button>
                  ))}
                </div>

                {activeTab === "text" && (
                  <div className="space-y-4">
                    <motion.p
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.5 }}
                      className="text-emerald-300 font-medium"
                    >
                      Enter text, URL, or paste content to analyze:
                    </motion.p>
                    <Textarea
                      placeholder="Enter a URL, article, financial report, or any text you want to analyze..."
                      className="min-h-[200px] bg-slate-800/70 backdrop-blur-sm border-slate-700 focus:border-emerald-500 transition-all duration-300 rounded-lg resize-none"
                      value={text}
                      onChange={(e) => setText(e.target.value)}
                    />
                    <div className="flex justify-between items-center">
                      <p className="text-sm text-emerald-200/70">
                        Type or paste text, enter a URL, or simply ask a question about a topic.
                      </p>
                      <Button
                        onClick={handleAnalyze}
                        disabled={!text.trim() || isAnalyzing}
                        className={cn(
                          "bg-gradient-to-r from-emerald-600 to-teal-500 hover:from-emerald-500 hover:to-teal-400",
                          "text-white font-medium px-6 py-2 rounded-lg shadow-lg shadow-emerald-900/30",
                          "transition-all duration-300 transform hover:scale-105 active:scale-95",
                          "disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none",
                        )}
                      >
                        {isAnalyzing ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Analyzing...
                          </>
                        ) : (
                          "Analyze"
                        )}
                      </Button>
                    </div>
                  </div>
                )}

                {activeTab === "upload" && (
                  <div className="space-y-4">
                    <motion.p
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.5 }}
                      className="text-emerald-300 font-medium"
                    >
                      Upload a file to analyze (PDF, CSV, JSON, or text file)
                    </motion.p>
                    <motion.div
                      whileHover={{ scale: 1.01 }}
                      transition={{ type: "spring", stiffness: 300 }}
                      className="border-2 border-dashed border-slate-700 hover:border-emerald-500 rounded-lg p-8 text-center transition-colors duration-300 bg-slate-800/50 backdrop-blur-sm"
                      onDragOver={handleDragOver}
                      onDrop={handleDrop}
                    >
                      <div className="flex flex-col items-center justify-center gap-4">
                        <div className="relative">
                          <Cloud className="h-16 w-16 text-slate-600" />
                          <motion.div
                            className="absolute inset-0 flex items-center justify-center"
                            animate={{
                              opacity: [0.5, 1, 0.5],
                              scale: [0.95, 1.05, 0.95],
                            }}
                            transition={{
                              repeat: Number.POSITIVE_INFINITY,
                              duration: 3,
                              ease: "easeInOut",
                            }}
                          >
                            <Cloud className="h-16 w-16 text-emerald-500/30" />
                          </motion.div>
                        </div>
                        <div className="space-y-2">
                          {fileName ? (
                            <p className="text-xl font-light">{fileName}</p>
                          ) : (
                            <p className="text-xl font-light">Drag and drop file here</p>
                          )}
                          <p className="text-sm text-emerald-200/50">Limit 200MB per file • PDF, CSV, JSON, TXT</p>
                        </div>
                        <input
                          type="file"
                          ref={fileInputRef}
                          onChange={handleFileUpload}
                          className="hidden"
                          accept=".pdf,.csv,.json,.txt"
                        />
                        <Button
                          variant="outline"
                          className="mt-2 border-slate-600 text-emerald-300 hover:bg-emerald-900/20 hover:text-emerald-200 transition-all duration-300"
                          onClick={() => fileInputRef.current?.click()}
                        >
                          Browse files
                        </Button>
                      </div>
                    </motion.div>
                    <div>
                      <Button
                        className={cn(
                          "bg-gradient-to-r from-emerald-600 to-teal-500 hover:from-emerald-500 hover:to-teal-400",
                          "text-white font-medium px-6 py-2 rounded-lg shadow-lg shadow-emerald-900/30",
                          "transition-all duration-300 transform hover:scale-105 active:scale-95",
                          "disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none",
                        )}
                        disabled={!uploadedFile || isAnalyzing}
                        onClick={handleAnalyzeFile}
                      >
                        {isAnalyzing ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Analyzing...
                          </>
                        ) : (
                          "Analyze File"
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
                              {data.source || "Unknown"} ({new Date(data.timestamp).toLocaleDateString()})
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
                    <div>
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
                    </div>
                  </div>
                )}
              </div>

              {/* Analysis Result Display */}
              {analysisResult && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2, duration: 0.5 }}
                  className="mt-8 bg-slate-800/70 backdrop-blur-sm rounded-lg p-6 border border-emerald-500/20"
                >
                  <h2 className="text-2xl font-bold text-emerald-400 mb-4">Analysis Results</h2>
                  
                  {/* Sentiment Overview */}
                  <div className="mb-6">
                    <h3 className="text-xl font-semibold text-emerald-300 mb-2">Sentiment</h3>
                    <div className="flex items-center space-x-4">
                      <div className={`text-xl font-bold ${
                        analysisResult.sentiment?.sentiment === "positive" ? "text-green-500" : 
                        analysisResult.sentiment?.sentiment === "negative" ? "text-red-500" : 
                        "text-yellow-500"
                      }`}>
                        {analysisResult.sentiment?.sentiment?.toUpperCase() || "NEUTRAL"}
                      </div>
                      <div className="text-gray-300">
                        Score: {(analysisResult.sentiment?.score || 0).toFixed(2)}
                      </div>
                    </div>
                    <p className="mt-2 text-gray-300">
                      {analysisResult.sentiment?.rationale || "No rationale provided."}
                    </p>
                  </div>
                  
                  {/* Detailed Analysis */}
                  <div className="mb-6">
                    <h3 className="text-xl font-semibold text-emerald-300 mb-2">Detailed Analysis</h3>
                    <p className="text-gray-300">
                      {analysisResult.analysis || "No detailed analysis available."}
                    </p>
                  </div>
                  
                  {/* Metadata/Topics */}
                  {analysisResult.metadata && (
                    <div>
                      <h3 className="text-xl font-semibold text-emerald-300 mb-2">Topics & Metadata</h3>
                      
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {/* Topics */}
                        <div className="bg-slate-700/50 p-4 rounded-lg">
                          <h4 className="font-medium text-emerald-200 mb-2">Topics</h4>
                          <div className="flex flex-wrap gap-2">
                            {analysisResult.metadata.topics?.length > 0 ? (
                              analysisResult.metadata.topics.map((topic, i) => (
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
                              analysisResult.metadata.regions.map((region, i) => (
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
                              analysisResult.metadata.entities.map((entity, i) => (
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
                  From news articles to financial reports, Sentimizer uses advanced AI to analyze and interpret sentiment, extract key topics, and identify trends.
                </motion.p>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <footer className="mt-8 pb-8 text-center text-sm text-emerald-200/30">
        <p>© 2023 Sentimizer • Powered by OpenAI's GPT models</p>
      </footer>
    </div>
  )
}
