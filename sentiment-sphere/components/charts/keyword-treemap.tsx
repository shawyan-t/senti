"use client"

import React, { useEffect, useState } from 'react'
import dynamic from 'next/dynamic'
import { Loader2 } from 'lucide-react'

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { 
  ssr: false,
  loading: () => <div className="w-full h-[250px] flex items-center justify-center">
    <Loader2 className="h-8 w-8 text-emerald-500 animate-spin" />
  </div>
})

interface KeywordData {
  keyword: string
  frequency: number
  sentiment?: 'positive' | 'neutral' | 'negative'
}

interface KeywordTreemapProps {
  keywords: KeywordData[]
  height?: number
  className?: string
}

export function KeywordTreemap({ 
  keywords = [], 
  height = 250,
  className 
}: KeywordTreemapProps) {
  const [mounted, setMounted] = useState(false)
  
  // Ensure component is mounted before rendering Plotly
  useEffect(() => {
    setMounted(true)
  }, [])

  // Don't show fake data - return a message instead if no real keyword data
  if (keywords.length === 0) {
    return (
      <div className={`w-full flex items-center justify-center text-gray-400 ${className}`} style={{height}}>
        <div className="text-center">
          <div className="text-lg mb-2">No Keyword Data Available</div>
          <div className="text-sm">Keyword analysis requires extractable topics or entities</div>
        </div>
      </div>
    )
  }

  // Sort keywords by frequency descending
  const sortedKeywords = [...keywords].sort((a, b) => b.frequency - a.frequency)

  // For treemap, we need labels, values, and parents
  const labels = sortedKeywords.map(k => k.keyword)
  const values = sortedKeywords.map(k => k.frequency)
  
  // Color grading by frequency percentile
  const n = sortedKeywords.length;
  const colors = sortedKeywords.map((_, i) => {
    const percentile = i / n;
    if (percentile < 1/3) return 'rgba(16, 185, 129, 0.8)'; // Green
    if (percentile < 2/3) return 'rgba(245, 158, 11, 0.8)'; // Yellow
    return 'rgba(239, 68, 68, 0.8)'; // Red
  })

  const data = [{
    type: 'treemap',
    labels: labels,
    parents: Array(labels.length).fill(''),
    values: values,
    textinfo: 'label+value',
    hoverinfo: 'label+value+percent root',
    marker: {
      colors: colors,
      line: {
        width: 1,
        color: 'rgba(0, 0, 0, 0.3)'
      }
    },
  }]

  const layout = {
    margin: {
      l: 0,
      r: 0,
      b: 0,
      t: 0,
    },
    height: height,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {
      color: 'rgba(255, 255, 255, 0.9)'
    }
  }

  const config = {
    displayModeBar: false,
    responsive: true
  }

  if (!mounted) return null

  return (
    <div className={className}>
      <Plot
        data={data}
        layout={layout}
        config={config}
        className="w-full"
      />
    </div>
  )
}

// Mock keyword data function removed - no longer generating fake data