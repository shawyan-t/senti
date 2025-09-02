"use client"

import React, { useEffect, useState } from 'react'
import dynamic from 'next/dynamic'
import { Loader2 } from 'lucide-react'

// Dynamically import Plotly wrapper (pre-minified Plotly build)
const Plot = dynamic(() => import('./plotly-wrapper'), {
  ssr: false,
  loading: () => <div className="w-full h-[250px] flex items-center justify-center">
    <Loader2 className="h-8 w-8 text-emerald-500 animate-spin" />
  </div>
})

interface EmotionData {
  emotion: string
  score: number
}

interface EmotionRadarChartProps {
  emotions: EmotionData[]
  height?: number
  className?: string
}

const DEFAULT_EMOTIONS = [
  { emotion: 'Joy', score: 0 },
  { emotion: 'Trust', score: 0 },
  { emotion: 'Fear', score: 0 },
  { emotion: 'Surprise', score: 0 },
  { emotion: 'Sadness', score: 0 },
  { emotion: 'Disgust', score: 0 },
  { emotion: 'Anger', score: 0 },
  { emotion: 'Anticipation', score: 0 },
]

const EMOTION_COLORS = {
  'Joy': '#F59E0B', // Amber
  'Trust': '#10B981', // Emerald
  'Fear': '#6366F1', // Indigo
  'Surprise': '#8B5CF6', // Violet
  'Sadness': '#3B82F6', // Blue
  'Disgust': '#EC4899', // Pink
  'Anger': '#EF4444', // Red
  'Anticipation': '#F97316', // Orange
}

export function EmotionRadarChart({ 
  emotions = DEFAULT_EMOTIONS, 
  height = 250,
  className 
}: EmotionRadarChartProps) {
  const [mounted, setMounted] = useState(false)
  
  // Ensure component is mounted before rendering Plotly
  useEffect(() => {
    setMounted(true)
  }, [])

  // Prepare the data for the radar chart
  const chartData = [{
    type: 'scatterpolar',
    r: emotions.map(e => e.score),
    theta: emotions.map(e => e.emotion),
    fill: 'toself',
    fillcolor: 'rgba(16, 185, 129, 0.2)', // Emerald with opacity
    line: {
      color: 'rgb(16, 185, 129)'
    },
    hoverinfo: 'text',
    text: emotions.map(e => `${e.emotion}: ${(e.score * 100).toFixed(0)}%`),
  }]

  const layout = {
    polar: {
      radialaxis: {
        visible: true,
        range: [0, 1],
        tickfont: {
          size: 8,
          color: 'rgba(255, 255, 255, 0.6)'
        }
      },
      angularaxis: {
        tickfont: {
          size: 10,
          color: 'rgba(255, 255, 255, 0.8)'
        }
      },
      bgcolor: 'rgba(0,0,0,0)'
    },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: {
      l: 40,
      r: 40,
      b: 30,
      t: 30,
    },
    height: height,
    autosize: true,
    showlegend: false,
  }

  const config = {
    displayModeBar: false,
    responsive: true
  }

  if (!mounted) return null

  return (
    <div className={className}>
      <Plot
        data={chartData}
        layout={layout}
        config={config}
        className="w-full"
      />
    </div>
  )
} 
