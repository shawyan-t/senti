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

const formatMetricValue = (key: string, value: any): string => {
  if (typeof value === 'number') {
    if (key.includes('score') || key.includes('composite')) {
      return value.toFixed(3)
    }
    if (key.includes('count') || key.includes('results') || key.includes('queries')) {
      return value.toLocaleString()
    }
  }
  return String(value)
}

const getStageIcon = (stage: string) => {
  switch (stage) {
    case 'start': return '🚀'
    case 'validate': return '✅'
    case 'fetched': return '📡'
    case 'units': return '📋'
    case 'engine': return '⚙️'
    case 'model': return '🧠'
    case 'visuals': return '📊'
    case 'final': return '🎉'
    default: return '📈'
  }
}

const getMetricIcon = (key: string) => {
  if (key.includes('score') || key.includes('composite')) return '🎯'
  if (key.includes('count') || key.includes('results')) return '📊'
  if (key.includes('queries') || key.includes('planned')) return '🔍'
  if (key.includes('ticker')) return '💰'
  if (key.includes('visual')) return '📈'
  if (key.includes('analysis')) return '🔬'
  if (key.includes('search') || key.includes('use_search')) return '🌐'
  return '📋'
}

const getStageDescription = (stage: string, metrics?: Record<string, any>) => {
  switch (stage) {
    case 'start':
      return `Initializing comprehensive sentiment analysis${metrics?.use_search ? ' with web search enabled' : ''}`
    case 'validate':
      return `Ticker validated and ${metrics?.queries_planned || 0} specialized financial queries generated`
    case 'fetched':
      return `Retrieved ${metrics?.unique_results || 0} unique sources from financial news, Reddit, and search engines`
    case 'units':
      return 'Processing and structuring content units for mathematical analysis'
    case 'engine':
      return 'Running multi-model sentiment aggregation with temporal weighting and source quality scoring'
    case 'model':
      return `Mathematical sentiment computed: ${metrics?.composite_score ? formatMetricValue('composite_score', metrics.composite_score) : 'processing...'}`
    case 'visuals':
      return `Generated ${metrics?.visual_count || 0} professional visualization charts`
    case 'final':
      return `Analysis complete! Saved as ${metrics?.analysis_id ? metrics.analysis_id.slice(0, 8) + '...' : 'new analysis'}`
    default:
      return 'Processing analysis pipeline...'
  }
}

export function AnalysisProgress({ percent, history }: { percent: number; history: HistoryItem[] }) {
  const clamped = Math.max(0, Math.min(100, percent))
  const checkpoints = (history || []).sort((a, b) => a.percent - b.percent)

  return (
    <div className="mt-4">
      <div className="flex items-center justify-between text-sm font-medium text-emerald-200 mb-2">
        <span>Financial Analysis Pipeline</span>
        <span className="text-emerald-300">{Math.floor(clamped)}%</span>
      </div>
      
      <div className="relative h-4 w-full rounded-full bg-purple-900/30 overflow-hidden border border-purple-600/30 shadow-inner">
        <div
          className="absolute left-0 top-0 h-full bg-gradient-to-r from-emerald-500 via-teal-400 to-emerald-400 transition-all duration-1000 ease-out shadow-lg"
          style={{ width: `${clamped}%` }}
        />
        
        {/* Enhanced checkpoint bubbles */}
        {checkpoints.map((cp, idx) => {
          const left = cp.percent
          const done = cp.percent <= clamped
          const isActive = idx === checkpoints.findIndex(c => c.percent > clamped) - 1
          
          return (
            <Dialog key={`${cp.stage}-${idx}`}>
              <DialogTrigger asChild>
                <button
                  className={`absolute -top-2 h-8 w-8 rounded-full border-2 transition-all duration-300 flex items-center justify-center text-xs font-bold shadow-lg ${
                    done
                      ? 'bg-gradient-to-br from-emerald-400 to-teal-500 border-emerald-200 text-white shadow-emerald-900/50 scale-110'
                      : isActive
                      ? 'bg-gradient-to-br from-amber-400 to-orange-500 border-amber-200 text-white animate-pulse shadow-amber-900/50'
                      : 'bg-purple-800/60 border-purple-600/50 text-purple-300 hover:bg-purple-700/70 hover:scale-105'
                  }`}
                  style={{ left: `calc(${left}% - 16px)` }}
                  title={`${cp.label} — ${done ? 'Complete' : isActive ? 'In Progress' : 'Pending'}`}
                >
                  {getStageIcon(cp.stage)}
                </button>
              </DialogTrigger>
              
              <DialogContent className="sm:max-w-2xl bg-gradient-to-br from-slate-900 to-purple-950 border border-purple-600/30">
                <DialogHeader>
                  <DialogTitle className="text-emerald-400 flex items-center gap-2 text-xl">
                    <span className="text-2xl">{getStageIcon(cp.stage)}</span>
                    {cp.label}
                    <Badge variant="outline" className="ml-2 border-emerald-500/50 text-emerald-300">
                      {cp.percent}%
                    </Badge>
                  </DialogTitle>
                </DialogHeader>
                
                <div className="space-y-4">
                  {/* Stage Description */}
                  <div className="p-4 bg-purple-900/20 rounded-lg border border-purple-600/20">
                    <h4 className="text-sm font-semibold text-emerald-300 mb-2">Pipeline Stage Details</h4>
                    <p className="text-sm text-white leading-relaxed">
                      {getStageDescription(cp.stage, cp.metrics)}
                    </p>
                  </div>

                  {/* Detailed Metrics */}
                  {cp.metrics && Object.keys(cp.metrics).length > 0 && (
                    <div className="p-4 bg-purple-900/20 rounded-lg border border-purple-600/20">
                      <h4 className="text-sm font-semibold text-emerald-300 mb-3">Technical Metrics</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {Object.entries(cp.metrics).map(([key, value]) => (
                          <div key={key} className="flex items-center justify-between p-2 bg-slate-800/40 rounded border border-slate-600/20">
                            <div className="flex items-center gap-2">
                              <span className="text-lg">{getMetricIcon(key)}</span>
                              <span className="text-xs font-medium text-purple-300 capitalize">
                                {key.replace(/_/g, ' ')}
                              </span>
                            </div>
                            <span className="text-sm font-bold text-white bg-emerald-400/20 px-2 py-1 rounded">
                              {formatMetricValue(key, value)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Progress Status */}
                  <div className="flex items-center justify-between text-xs text-purple-300">
                    <span>Stage: {cp.stage}</span>
                    <span>
                      Status: {done ? '✅ Complete' : isActive ? '⏳ In Progress' : '⏸️ Pending'}
                    </span>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          )
        })}
      </div>
      
      {/* Current active stage label directly below progress bar */}
      {checkpoints.length > 0 && (
        <div className="mt-3 relative h-6">
          {checkpoints.map((cp, idx) => {
            const done = cp.percent <= clamped
            const isActive = idx === checkpoints.findIndex(c => c.percent > clamped) - 1
            
            // Only show labels for completed stages and the current active stage
            if (!done && !isActive) return null
            
            return (
              <div 
                key={`${cp.stage}-label-${idx}`} 
                className="absolute flex flex-col items-center transform -translate-x-1/2"
                style={{ left: `${cp.percent}%` }}
              >
                <span className={`text-[10px] font-medium transition-colors text-center leading-tight max-w-[60px] ${
                  done 
                    ? 'text-emerald-300' 
                    : isActive 
                    ? 'text-amber-300 animate-pulse' 
                    : 'text-purple-400'
                }`}>
                  {cp.label}
                </span>
                {cp.metrics && Object.keys(cp.metrics).length > 0 && (
                  <span className="text-[8px] text-white/60 mt-0.5">
                    {Object.keys(cp.metrics).length} metrics
                  </span>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
