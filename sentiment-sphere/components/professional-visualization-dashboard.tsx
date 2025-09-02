"use client"

import React, { useEffect, useRef } from 'react'
import dynamic from 'next/dynamic'
import { Loader2 } from 'lucide-react'

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[300px] flex items-center justify-center bg-slate-800/50 rounded-lg">
      <Loader2 className="h-8 w-8 text-emerald-500 animate-spin" />
    </div>
  )
})

interface VisualizationData {
  chart_data: string
  description: string
}

interface ProfessionalVisualizationDashboardProps {
  visualizations: {
    sentiment_index?: VisualizationData
    polarity_distribution?: VisualizationData
    vad_compass?: VisualizationData
    source_quality?: VisualizationData
    sentiment_timeline?: VisualizationData
  }
  className?: string
}

function PlotlyChart({ 
  chartData, 
  height = 300, 
  className = "" 
}: { 
  chartData: string
  height?: number
  className?: string 
}) {
  try {
    const plotlyData = JSON.parse(chartData)
    
    return (
      <div className={`w-full ${className}`}>
        <Plot
          data={plotlyData.data}
          layout={{
            ...plotlyData.layout,
            height,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {
              color: '#e2e8f0'
            }
          }}
          config={{
            displayModeBar: false,
            responsive: true
          }}
          style={{ width: '100%', height: `${height}px` }}
        />
      </div>
    )
  } catch (error) {
    console.error('Error parsing chart data:', error)
    return (
      <div className="w-full h-[300px] flex items-center justify-center bg-red-900/20 rounded-lg border border-red-700/50">
        <div className="text-center">
          <div className="text-red-400 text-sm">⚠️ Chart Error</div>
          <div className="text-red-300 text-xs mt-1">Failed to render visualization</div>
        </div>
      </div>
    )
  }
}

export function ProfessionalVisualizationDashboard({ 
  visualizations, 
  className = "" 
}: ProfessionalVisualizationDashboardProps) {
  return (
    <div className={`mt-12 ${className}`}>
      <h3 className="text-2xl font-bold text-emerald-400 mb-6">Professional Financial Sentiment Dashboard</h3>
      
      {/* Missing data banner */}
      {(!visualizations || Object.keys(visualizations).length === 0) && (
        <div className="mb-4 p-3 rounded bg-amber-900/30 text-amber-300 border border-amber-700/40">
          No visualization data available. Data sources may be incomplete.
        </div>
      )}

      <div className="space-y-8">
        {/* Executive Overview Section */}
        <div>
          <h4 className="text-xl font-semibold text-emerald-300 mb-4">📊 Executive Overview</h4>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Sentiment Index */}
            {visualizations.sentiment_index ? (
              <div className="bg-slate-800/70 backdrop-blur-sm rounded-lg p-4 border border-emerald-500/20">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-3 h-3 bg-emerald-400 rounded-full"></div>
                  <h5 className="text-lg font-semibold text-emerald-300">Sentiment Index</h5>
                </div>
                <p className="text-sm text-gray-400 mb-4">{visualizations.sentiment_index.description}</p>
                <PlotlyChart 
                  chartData={visualizations.sentiment_index.chart_data} 
                  height={350}
                />
              </div>
            ) : (
              <div className="bg-slate-800/70 backdrop-blur-sm rounded-lg p-4 border border-slate-600/30 text-gray-400">
                No Sentiment Index data available
              </div>
            )}

            {/* Polarity Distribution */}
            {visualizations.polarity_distribution ? (
              <div className="bg-slate-800/70 backdrop-blur-sm rounded-lg p-4 border border-emerald-500/20">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-3 h-3 bg-blue-400 rounded-full"></div>
                  <h5 className="text-lg font-semibold text-emerald-300">Polarity Distribution</h5>
                </div>
                <p className="text-sm text-gray-400 mb-4">{visualizations.polarity_distribution.description}</p>
                <PlotlyChart 
                  chartData={visualizations.polarity_distribution.chart_data} 
                  height={350}
                />
              </div>
            ) : (
              <div className="bg-slate-800/70 backdrop-blur-sm rounded-lg p-4 border border-slate-600/30 text-gray-400">
                No polarity distribution data available
              </div>
            )}

            {/* VAD Compass */}
            {visualizations.vad_compass ? (
              <div className="bg-slate-800/70 backdrop-blur-sm rounded-lg p-4 border border-emerald-500/20">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-3 h-3 bg-purple-400 rounded-full"></div>
                  <h5 className="text-lg font-semibold text-emerald-300">VAD Compass</h5>
                </div>
                <p className="text-sm text-gray-400 mb-4">{visualizations.vad_compass.description}</p>
                <PlotlyChart 
                  chartData={visualizations.vad_compass.chart_data} 
                  height={400}
                />
              </div>
            ) : (
              <div className="bg-slate-800/70 backdrop-blur-sm rounded-lg p-4 border border-slate-600/30 text-gray-400">
                No VAD data available
              </div>
            )}
          </div>
        </div>

        {/* Source Analysis Section */}
        <div>
          <h4 className="text-xl font-semibold text-emerald-300 mb-4">🔍 Source Analysis</h4>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Source Quality Matrix */}
            {visualizations.source_quality ? (
              <div className="bg-slate-800/70 backdrop-blur-sm rounded-lg p-4 border border-emerald-500/20">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-3 h-3 bg-amber-400 rounded-full"></div>
                  <h5 className="text-lg font-semibold text-emerald-300">Source Quality Matrix</h5>
                </div>
                <p className="text-sm text-gray-400 mb-4">{visualizations.source_quality.description}</p>
                <PlotlyChart 
                  chartData={visualizations.source_quality.chart_data} 
                  height={300}
                />
              </div>
            ) : (
              <div className="bg-slate-800/70 backdrop-blur-sm rounded-lg p-4 border border-slate-600/30 text-gray-400">
                No source quality data available
              </div>
            )}

            {/* Sentiment Timeline */}
            {visualizations.sentiment_timeline ? (
              <div className="bg-slate-800/70 backdrop-blur-sm rounded-lg p-4 border border-emerald-500/20">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-3 h-3 bg-cyan-400 rounded-full"></div>
                  <h5 className="text-lg font-semibold text-emerald-300">Sentiment Timeline</h5>
                </div>
                <p className="text-sm text-gray-400 mb-4">{visualizations.sentiment_timeline.description}</p>
                <PlotlyChart 
                  chartData={visualizations.sentiment_timeline.chart_data} 
                  height={350}
                />
              </div>
            ) : (
              <div className="bg-slate-800/70 backdrop-blur-sm rounded-lg p-4 border border-slate-600/30 text-gray-400">
                No timeline data available
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Footer */}
      <div className="mt-6 text-center">
        <p className="text-sm text-gray-400">
          🎯 Professional visualizations generated using advanced statistical analysis with bootstrap confidence intervals and Wilson bounds
        </p>
      </div>
    </div>
  )
}
