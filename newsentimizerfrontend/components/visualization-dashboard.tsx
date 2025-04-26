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

  // Extract emotion data or use defaults
  const emotions = analysisData.emotions || generateEmotionsFromSentiment(analysisData.sentiment?.score || 0)
  
  // Prepare keywords data
  const keywords = analysisData.keywords || generateKeywordsFromTopics(analysisData.metadata?.topics)
  
  // Prepare time series data
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
          <EmotionRadarChart emotions={emotions} height={300} />
        </DashboardPanel>
        
        {/* Keyword Treemap */}
        <DashboardPanel 
          title="Key Topics & Themes" 
          subtitle="Important keywords sized by frequency"
        >
          <KeywordTreemap keywords={keywords} height={300} />
        </DashboardPanel>
      </DashboardLayout>
    </div>
  )
}

// Helper function to generate mock emotion data based on sentiment score
function generateEmotionsFromSentiment(sentimentScore: number) {
  // Default base values
  const emotions = [
    { emotion: 'Joy', score: 0.2 },
    { emotion: 'Trust', score: 0.3 },
    { emotion: 'Fear', score: 0.1 },
    { emotion: 'Surprise', score: 0.15 },
    { emotion: 'Sadness', score: 0.1 },
    { emotion: 'Disgust', score: 0.05 },
    { emotion: 'Anger', score: 0.05 },
    { emotion: 'Anticipation', score: 0.25 },
  ]
  
  // Adjust based on sentiment score (-1 to 1)
  if (sentimentScore > 0) {
    // More positive sentiment increases joy, trust, anticipation
    emotions[0].score += sentimentScore * 0.4 // Joy
    emotions[1].score += sentimentScore * 0.3 // Trust
    emotions[7].score += sentimentScore * 0.2 // Anticipation
    
    // Decrease negative emotions
    emotions[2].score = Math.max(0.05, emotions[2].score - sentimentScore * 0.2) // Fear
    emotions[4].score = Math.max(0.05, emotions[4].score - sentimentScore * 0.2) // Sadness
    emotions[5].score = Math.max(0.05, emotions[5].score - sentimentScore * 0.2) // Disgust
    emotions[6].score = Math.max(0.05, emotions[6].score - sentimentScore * 0.2) // Anger
  } else if (sentimentScore < 0) {
    // More negative sentiment increases sadness, fear, anger, disgust
    emotions[2].score -= sentimentScore * 0.3 // Fear
    emotions[4].score -= sentimentScore * 0.3 // Sadness
    emotions[5].score -= sentimentScore * 0.2 // Disgust
    emotions[6].score -= sentimentScore * 0.3 // Anger
    
    // Decrease positive emotions
    emotions[0].score = Math.max(0.05, emotions[0].score + sentimentScore * 0.3) // Joy
    emotions[1].score = Math.max(0.05, emotions[1].score + sentimentScore * 0.2) // Trust
    emotions[7].score = Math.max(0.05, emotions[7].score + sentimentScore * 0.2) // Anticipation
  }
  
  // Normalize so the sum equals 1
  const total = emotions.reduce((sum, e) => sum + e.score, 0)
  return emotions.map(e => ({ ...e, score: e.score / total }))
}

// Helper function to generate keyword data from topics
function generateKeywordsFromTopics(topics?: string[]) {
  if (!topics || topics.length === 0) {
    return []
  }
  
  // Convert topics to keyword data
  return topics.map((topic, index) => {
    // Frequency decreases with index (first topics are more important)
    const frequency = 100 - (index * 10)
    
    // Randomly assign sentiment
    const sentiments: Array<'positive' | 'neutral' | 'negative'> = ['positive', 'neutral', 'negative']
    const sentiment = sentiments[Math.floor(Math.random() * sentiments.length)]
    
    return {
      keyword: topic,
      frequency: Math.max(10, frequency), // Ensure minimum frequency of 10
      sentiment
    }
  })
} 