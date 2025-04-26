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

  if (keywords.length === 0) {
    keywords = generateMockKeywordData()
  }

  // Sort keywords by frequency descending
  const sortedKeywords = [...keywords].sort((a, b) => b.frequency - a.frequency)

  // For treemap, we need labels, values, and parents
  const labels = sortedKeywords.map(k => k.keyword)
  const values = sortedKeywords.map(k => k.frequency)
  
  // Generate colors based on sentiment
  const colors = sortedKeywords.map(k => {
    if (k.sentiment === 'positive') return 'rgba(16, 185, 129, 0.8)' // Emerald
    if (k.sentiment === 'negative') return 'rgba(239, 68, 68, 0.8)' // Red
    return 'rgba(107, 114, 128, 0.8)' // Gray (neutral)
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

// Generate mock keyword data if none is provided
function generateMockKeywordData(): KeywordData[] {
  const mockKeywords = [
    { keyword: 'Technology', frequency: 85, sentiment: 'positive' as const },
    { keyword: 'Economy', frequency: 65, sentiment: 'neutral' as const },
    { keyword: 'Politics', frequency: 60, sentiment: 'negative' as const },
    { keyword: 'Environment', frequency: 45, sentiment: 'positive' as const },
    { keyword: 'Health', frequency: 40, sentiment: 'positive' as const },
    { keyword: 'Education', frequency: 35, sentiment: 'neutral' as const },
    { keyword: 'Sports', frequency: 30, sentiment: 'positive' as const },
    { keyword: 'Entertainment', frequency: 25, sentiment: 'neutral' as const },
    { keyword: 'Business', frequency: 20, sentiment: 'positive' as const },
    { keyword: 'Science', frequency: 15, sentiment: 'positive' as const },
  ]
  
  return mockKeywords
} 