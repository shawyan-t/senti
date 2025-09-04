"use client"

import React from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog'
import { Badge } from './ui/badge'

type HistoryItem = {
  percent: number
  stage: string
  label: string
  metrics?: Record<string, any>
}

export function AnalysisProgress({ percent, history }: { percent: number; history: HistoryItem[] }) {
  const clamped = Math.max(0, Math.min(100, percent))
  const checkpoints = (history || []).sort((a, b) => a.percent - b.percent)

  return (
    <div className="mt-3">
      <div className="flex items-center justify-between text-xs sm:text-sm text-emerald-200/80 mb-1 sm:mb-2">
        <span>Analysis Progress</span>
        <span className="font-mono">{Math.floor(clamped)}%</span>
      </div>
      <div className="px-2 sm:px-0">
        <div className="relative h-3 sm:h-4 w-full rounded-full bg-emerald-900/30 overflow-visible border border-emerald-700/30">
          <div
            className="absolute left-0 top-0 h-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-700 rounded-full"
            style={{ width: `${clamped}%` }}
          />
          {/* Dynamic checkpoints from backend history */}
          {checkpoints.map((cp, idx) => {
            const left = cp.percent
            const done = cp.percent <= clamped
            const current = cp.percent <= clamped && (idx === checkpoints.length - 1 || checkpoints[idx + 1].percent > clamped) && clamped < 100
            return (
              <Dialog key={`${cp.stage}-${idx}`}>
                <DialogTrigger asChild>
                  <button
                    className={`absolute -top-1 sm:-top-1.5 h-7 w-7 sm:h-8 sm:w-8 rounded-full border-2 transition-all duration-500 transform hover:scale-110 active:scale-95 z-10 ${
                      done
                        ? 'bg-gradient-to-br from-emerald-400 to-teal-500 border-emerald-200 shadow-lg shadow-emerald-900/60'
                        : current
                          ? 'bg-gradient-to-br from-emerald-400 to-teal-500 border-emerald-200 shadow-lg shadow-emerald-900/60 animate-pulse'
                          : 'bg-gradient-to-br from-slate-600 to-slate-800 border-slate-400 hover:from-slate-500 hover:to-slate-700 hover:border-slate-300 shadow-md'
                    }`}
                    style={{ left: `calc(${left}% - 14px)` }}
                    title={`${cp.label} — click to view detailed metrics`}
                  >
                  <div className={`w-full h-full rounded-full flex items-center justify-center ${
                    done || current ? 'text-emerald-900' : 'text-slate-300'
                  }`}>
                    {done ? (
                      <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    ) : current ? (
                      <div className="w-2 h-2 bg-current rounded-full"></div>
                    ) : (
                      <div className="w-1.5 h-1.5 bg-current rounded-full opacity-60"></div>
                    )}
                  </div>
                </button>
              </DialogTrigger>
              <DialogContent className="max-w-[95vw] sm:max-w-2xl bg-slate-900 border-emerald-700/30 max-h-[90vh] overflow-hidden">
                <DialogHeader className="pb-3 sm:pb-4">
                  <div className="flex items-center justify-between">
                    <DialogTitle className="text-emerald-400 flex items-center gap-2 sm:gap-3 text-sm sm:text-base">
                      <div className="w-6 h-6 sm:w-8 sm:h-8 bg-gradient-to-br from-emerald-400 to-teal-500 rounded-full flex items-center justify-center">
                        <svg className="w-3 h-3 sm:w-4 sm:h-4 text-emerald-900" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      </div>
                      {cp.label}
                    </DialogTitle>
                    <Badge variant="secondary" className="bg-emerald-900/50 text-emerald-200 border-emerald-700 text-xs sm:text-sm">
                      {cp.percent}%
                    </Badge>
                  </div>
                </DialogHeader>
                <div className="text-xs sm:text-sm text-gray-100 leading-relaxed max-h-[60vh] sm:max-h-96 overflow-y-auto">
                  {cp.metrics && Object.keys(cp.metrics).length > 0 ? (
                    <div className="space-y-3 sm:space-y-4">
                      {/* Pipeline Stage Information */}
                      <div className="bg-slate-800/50 rounded-lg p-2 sm:p-3 border border-emerald-700/30">
                        <div className="flex items-center gap-2 mb-2">
                          <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
                          <span className="text-emerald-300 font-semibold text-xs uppercase tracking-wide">
                            Stage: {cp.stage}
                          </span>
                        </div>
                        <p className="text-gray-300 text-xs">
                          Progress: {cp.percent}% • Timestamp: {cp.metrics.timestamp || 'Real-time'}
                        </p>
                      </div>

                      {/* Detailed Metrics Display */}
                      <div className="grid gap-3">
                        {Object.entries(cp.metrics).map(([key, value]) => {
                          // Skip basic fields already shown above
                          if (['timestamp', 'stage', 'percent'].includes(key)) return null;
                          
                          const isNumeric = typeof value === 'number';
                          const isObject = typeof value === 'object' && value !== null;
                          
                          
                          return (
                            <div key={key} className="bg-slate-900/40 rounded-md p-2 sm:p-3 border border-slate-700/50">
                              <div className="flex items-start justify-between mb-1">
                                <span className="text-emerald-300 font-medium text-xs capitalize">
                                  {key === 'unique_results' ? 'Sources Found' :
                                   key === 'search_results_count' ? 'Results Processed' :
                                   key === 'domains_found' ? 'Domains Discovered' :
                                   key === 'total_content_chars' ? 'Content Characters' :
                                   key === 'total_units_processed' ? 'Units Processed' :
                                   key === 'clusters_formed' ? 'Clusters Formed' :
                                   key === 'domains_analyzed' ? 'Domains Analyzed' :
                                   key === 'total_weight' ? 'Total Weight' :
                                   key === 'weight_concentration' ? 'Weight Concentration' :
                                   key === 'tukey_biweight_aggregation' ? 'Tukey Biweight' :
                                   key === 'freshness_score' ? 'Freshness Score' :
                                   key === 'composite_sentiment' ? 'Composite Sentiment' :
                                   key === 'sources_processed_for_plots' ? 'Sources Processed' :
                                   key === 'sentiment_data_points' ? 'Sentiment Data Points' :
                                   key === 'timeline_data_points' ? 'Timeline Data Points' :
                                   key === 'polarity_categories_computed' ? 'Polarity Categories' :
                                   key === 'confidence_intervals_calculated' ? 'Confidence Intervals' :
                                   key === 'vad_dimensions_computed' ? 'VAD Dimensions' :
                                   key === 'sentiment_distribution_calculated' ? 'Distribution Calculated' :
                                   key === 'vad_analysis_complete' ? 'VAD Analysis' :
                                   key === 'total_visualizations' ? 'Total Visualizations' :
                                   key.replace(/_/g, ' ')}
                                </span>
                                {isNumeric && (
                                  <div className="bg-emerald-900/50 px-2 py-0.5 rounded text-emerald-200 text-xs font-bold">
                                    {typeof value === 'number' ? value.toLocaleString() : value}
                                  </div>
                                )}
                              </div>
                              
                              <div className="text-gray-200 text-xs">
                                {isObject ? (
                                  <div className="space-y-1">
                                    {Object.entries(value).map(([subKey, subValue]) => (
                                      <div key={subKey} className="flex justify-between items-center py-0.5">
                                        <span className="text-gray-400">{subKey}:</span>
                                        <span className="text-emerald-200 font-mono">
                                          {typeof subValue === 'number' ? subValue.toFixed(4) : String(subValue)}
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                ) : Array.isArray(value) ? (
                                  <div className="space-y-1">
                                    {value.slice(0, 5).map((item, idx) => (
                                      <div key={idx} className="bg-slate-800/50 px-2 py-1 rounded text-xs">
                                        {String(item)}
                                      </div>
                                    ))}
                                    {value.length > 5 && (
                                      <div className="text-gray-400 text-xs">...and {value.length - 5} more</div>
                                    )}
                                  </div>
                                ) : typeof value === 'boolean' ? (
                                  <div className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs ${
                                    value ? 'bg-emerald-900/50 text-emerald-200' : 'bg-gray-600/50 text-gray-300'
                                  }`}>
                                    {value ? '✓ Active' : '✗ Inactive'}
                                  </div>
                                ) : (
                                  <div className="font-mono break-all">
                                    {isNumeric ? 
                                      (value < 1 && value > -1 ? value.toFixed(4) : value.toLocaleString()) 
                                      : String(value)
                                    }
                                  </div>
                                )}
                              </div>
                              
                              {/* Mathematical indicators for calculations */}
                              {(key.includes('score') || key.includes('weight') || key.includes('coefficient')) && isNumeric && (
                                <div className="mt-2 flex items-center gap-1">
                                  <div className="w-full bg-slate-700 rounded-full h-1">
                                    <div 
                                      className="bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 h-1 rounded-full transition-all"
                                      style={{ width: `${Math.min(100, Math.abs(value) * 100)}%` }}
                                    />
                                  </div>
                                  <span className="text-xs text-gray-400 ml-1">
                                    {value > 0 ? '↗' : value < 0 ? '↘' : '→'}
                                  </span>
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                      
                      {/* Chronological Pipeline Timeline */}
                      <div className="mt-4 pt-3 border-t border-slate-700/50">
                        <h4 className="text-emerald-300 font-medium text-xs mb-3 flex items-center gap-2">
                          <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                          </svg>
                          Pipeline Timeline
                        </h4>
                        
                        <div className="relative">
                          {/* Timeline line */}
                          <div className="absolute left-3 top-0 bottom-0 w-0.5 bg-gradient-to-b from-emerald-500 to-slate-600"></div>
                          
                          <div className="space-y-3">
                            {[
                              { stage: 'start', label: 'Initialization', desc: 'Pipeline setup and validation', percent: 0 },
                              { stage: 'validate', label: 'Input Validation', desc: 'Query parsing and sanitization', percent: 10 },
                              { stage: 'fetch_sources', label: 'Source Discovery', desc: 'Web crawling and content extraction', percent: 35 },
                              { stage: 'preprocessing', label: 'Data Processing', desc: 'Content cleaning and normalization', percent: 50 },
                              { stage: 'feature_extraction', label: 'Feature Engineering', desc: 'Mathematical feature extraction', percent: 65 },
                              { stage: 'sentiment_analysis', label: 'Sentiment Calculation', desc: 'ML model inference and scoring', percent: 80 },
                              { stage: 'postprocessing', label: 'Result Aggregation', desc: 'Statistical analysis and weighting', percent: 95 },
                              { stage: 'complete', label: 'Finalization', desc: 'Results compilation and delivery', percent: 100 }
                            ].map((timelineStage, idx) => {
                              // Use OVERALL progress percentage, not individual bubble percentage
                              const isCurrentStage = Math.abs(clamped - timelineStage.percent) <= 5 && clamped < 100;
                              const isPastStage = clamped >= timelineStage.percent;
                              const isFutureStage = clamped < timelineStage.percent;
                              
                              return (
                                <div key={timelineStage.stage} className="relative flex items-start gap-3">
                                  <div className={`relative z-10 w-6 h-6 rounded-full flex items-center justify-center border-2 ${
                                    isCurrentStage 
                                      ? 'bg-emerald-400 border-emerald-200 animate-pulse'
                                      : isPastStage
                                        ? 'bg-emerald-500 border-emerald-300'
                                        : 'bg-slate-700 border-slate-500'
                                  }`}>
                                    {isPastStage ? (
                                      <svg className="w-3 h-3 text-emerald-900" fill="currentColor" viewBox="0 0 20 20">
                                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                      </svg>
                                    ) : (
                                      <div className="w-2 h-2 bg-current rounded-full opacity-60"></div>
                                    )}
                                  </div>
                                  
                                  <div className="flex-1 min-w-0">
                                    <div className={`font-medium text-xs ${
                                      isCurrentStage ? 'text-emerald-300' : isPastStage ? 'text-emerald-400' : 'text-gray-400'
                                    }`}>
                                      {timelineStage.label}
                                    </div>
                                    <div className="text-xs text-gray-500 mt-0.5">
                                      {timelineStage.desc}
                                    </div>
                                    {isCurrentStage && (
                                      <Badge className="mt-1 bg-emerald-900/50 text-emerald-200 border-emerald-700 text-xs">
                                        Current
                                      </Badge>
                                    )}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                        
                        {/* Processing Status */}
                        <div className="mt-4 pt-3 border-t border-slate-700/50">
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-gray-400">Overall Progress</span>
                            <div className="flex items-center gap-1">
                              <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse"></div>
                              <span className="text-emerald-300">
                                {cp.percent < 100 ? 'Processing...' : 'Complete'}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <div className="text-emerald-300 text-sm mb-3">Processing Stage</div>
                      <div className="text-gray-400 text-xs mb-2">
                        Stage: <span className="text-emerald-200 font-mono">{cp.stage}</span>
                      </div>
                      <p className="text-gray-500 text-xs">
                        {cp.metrics ? 
                          'Backend sent empty metrics object' : 
                          'Waiting for metrics from backend'
                        }
                      </p>
                    </div>
                  )}
                </div>
              </DialogContent>
            </Dialog>
          )
          })}
        </div>
      </div>
    </div>
  )
}
