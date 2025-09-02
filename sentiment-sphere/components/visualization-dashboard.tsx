"use client"

import React, { useEffect, useState } from 'react'
import { DashboardLayout, DashboardPanel } from './dashboard-layout'
import dynamic from 'next/dynamic'
import { Loader2 } from 'lucide-react'

// Dynamically import our chart components to avoid SSR issues with Plotly
const EmotionRadarChart = dynamic(
  () => import('./charts/emotion-radar-chart').then(mod => mod.EmotionRadarChart),
  { 
    ssr: false,
    loading: () => <ChartLoader />
  }
)

const SentimentTimeline = dynamic(
  () => import('./charts/sentiment-timeline').then(mod => mod.SentimentTimeline),
  { 
    ssr: false,
    loading: () => <ChartLoader />
  }
)

const KeywordTreemap = dynamic(
  () => import('./charts/keyword-treemap').then(mod => mod.KeywordTreemap),
  { 
    ssr: false,
    loading: () => <ChartLoader />
  }
)

function ChartLoader() {
  return (
    <div className="w-full h-[250px] flex items-center justify-center">
      <Loader2 className="h-8 w-8 text-emerald-500 animate-spin" />
    </div>
  )
}

export interface AnalysisData {
  sentiment?: {
    sentiment: string
    score: number
    rationale?: string
  }
  analysis?: string
  metadata?: {
    topics?: string[]
    regions?: string[]
    entities?: string[]
  }
  timeSeriesData?: {
    date: string
    sentiment: number
    volume?: number
  }[]
  emotions?: {
    emotion: string
    score: number
  }[]
  keywords?: {
    keyword: string
    frequency: number
    sentiment?: 'positive' | 'neutral' | 'negative'
  }[]
}

interface VisualizationDashboardProps {
  analysisData: AnalysisData
  className?: string
}

export function VisualizationDashboard({ 
  analysisData,
  className 
}: VisualizationDashboardProps) {
  const [mounted, setMounted] = useState(false)
  
  // Ensure component is mounted before rendering
  useEffect(() => {
    setMounted(true)
  }, [])

  // Use only provided emotions; no synthetic fallbacks
  const emotions = analysisData.emotions || []
  
  // Use only provided keywords; no synthetic fallbacks
  const keywords = analysisData.keywords || []
  
  // Use only provided time series data; no synthetic fallbacks
  const timeSeriesData = analysisData.timeSeriesData || []

  if (!mounted) return null

  return (
    <div className={className}>
      <h2 className="text-2xl font-bold text-emerald-400 mb-6">Visualization Dashboard</h2>
      
      <DashboardLayout>
        {/* Sentiment Timeline Chart */}
        <DashboardPanel 
          title="Sentiment Timeline" 
          subtitle="Sentiment trend analysis over time"
          className="col-span-2"
        >
          <SentimentTimeline data={timeSeriesData} height={250} />
        </DashboardPanel>
        
        {/* Emotion Radar Chart */}
        <DashboardPanel 
          title="Emotion Analysis" 
          subtitle="Emotional dimensions expressed in content"
        >
          {emotions.length > 0 ? (
            <EmotionRadarChart emotions={emotions} height={300} />
          ) : (
            <div className="w-full h-[300px] flex items-center justify-center text-gray-400">
              No emotion data available
            </div>
          )}
        </DashboardPanel>
        
        {/* Keyword Treemap */}
        <DashboardPanel 
          title="Key Topics & Themes" 
          subtitle="Important keywords sized by frequency"
        >
          {keywords.length > 0 ? (
            <KeywordTreemap keywords={keywords} height={300} />
          ) : (
            <div className="w-full h-[300px] flex items-center justify-center text-gray-400">
              No keyword data available
            </div>
          )}
        </DashboardPanel>
      </DashboardLayout>
    </div>
  )
}
// Note: All mock/synthetic generators removed for production correctness
