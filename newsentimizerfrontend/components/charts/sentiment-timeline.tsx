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

interface TimelineDataPoint {
  date: string
  sentiment: number
  volume?: number
}

interface SentimentTimelineProps {
  data: TimelineDataPoint[]
  height?: number
  showVolume?: boolean
  className?: string
}

export function SentimentTimeline({ 
  data = [], 
  height = 250,
  showVolume = true,
  className 
}: SentimentTimelineProps) {
  const [mounted, setMounted] = useState(false)
  
  // Ensure component is mounted before rendering Plotly
  useEffect(() => {
    setMounted(true)
  }, [])

  if (data.length === 0) {
    data = generateMockTimelineData()
  }

  // Sort data by date
  const sortedData = [...data].sort((a, b) => 
    new Date(a.date).getTime() - new Date(b.date).getTime()
  )

  const dates = sortedData.map(d => d.date)
  const sentiments = sortedData.map(d => d.sentiment)
  const volumes = sortedData.map(d => d.volume || 0)

  // Define traces for the chart
  const traces: any[] = [
    {
      type: 'scatter',
      mode: 'lines',
      name: 'Sentiment',
      x: dates,
      y: sentiments,
      line: {
        color: '#10B981', // Emerald
        width: 3
      },
      hoverinfo: 'text',
      text: sortedData.map(d => 
        `Date: ${new Date(d.date).toLocaleDateString()}<br>Sentiment: ${d.sentiment.toFixed(2)}`
      ),
    }
  ]

  // Add volume bars if showVolume is true
  if (showVolume) {
    traces.push({
      type: 'bar',
      name: 'Volume',
      x: dates,
      y: volumes,
      marker: {
        color: 'rgba(59, 130, 246, 0.3)', // Blue with opacity
      },
      opacity: 0.7,
      yaxis: 'y2',
      hoverinfo: 'text',
      text: sortedData.map(d => 
        `Date: ${new Date(d.date).toLocaleDateString()}<br>Volume: ${d.volume || 0}`
      ),
    })
  }

  const layout = {
    xaxis: {
      title: '',
      showgrid: false,
      zeroline: false,
      tickfont: {
        color: 'rgba(255, 255, 255, 0.6)'
      }
    },
    yaxis: {
      title: 'Sentiment',
      range: [-1, 1],
      zeroline: true,
      zerolinecolor: 'rgba(255, 255, 255, 0.2)',
      gridcolor: 'rgba(255, 255, 255, 0.1)',
      tickfont: {
        color: 'rgba(255, 255, 255, 0.6)'
      }
    },
    yaxis2: showVolume ? {
      title: 'Volume',
      titlefont: {
        color: 'rgba(59, 130, 246, 0.8)'
      },
      tickfont: {
        color: 'rgba(59, 130, 246, 0.6)'
      },
      overlaying: 'y',
      side: 'right',
      showgrid: false,
    } : {},
    margin: {
      l: 40,
      r: 40,
      b: 40,
      t: 10,
      pad: 0
    },
    autosize: true,
    height: height,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    showlegend: false,
    hovermode: 'closest'
  }

  const config = {
    displayModeBar: false,
    responsive: true
  }

  if (!mounted) return null

  return (
    <div className={className}>
      <Plot
        data={traces}
        layout={layout}
        config={config}
        className="w-full"
      />
    </div>
  )
}

// Generate mock timeline data if none is provided
function generateMockTimelineData(): TimelineDataPoint[] {
  const data: TimelineDataPoint[] = []
  const today = new Date()
  
  for (let i = 30; i >= 0; i--) {
    const date = new Date(today)
    date.setDate(today.getDate() - i)
    
    // Generate a random walk sentiment between -1 and 1
    const randomChange = (Math.random() - 0.5) * 0.2
    const prevSentiment = data.length > 0 ? data[data.length - 1].sentiment : 0
    let sentiment = prevSentiment + randomChange
    sentiment = Math.max(-1, Math.min(1, sentiment)) // Clamp between -1 and 1
    
    // Random volume between 10 and 100
    const volume = Math.floor(Math.random() * 90) + 10
    
    data.push({
      date: date.toISOString().split('T')[0],
      sentiment,
      volume
    })
  }
  
  return data
} 