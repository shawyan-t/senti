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

  // Don't show fake data - return a message instead if no real timeline data
  if (data.length === 0) {
    return (
      <div className={`w-full flex items-center justify-center text-gray-400 ${className}`} style={{height}}>
        <div className="text-center">
          <div className="text-lg mb-2">No Timeline Data Available</div>
          <div className="text-sm">Timeline analysis requires multiple data points over time</div>
        </div>
      </div>
    )
  }

  // Sort data by date
  const sortedData = [...data].sort((a, b) => 
    new Date(a.date).getTime() - new Date(b.date).getTime()
  )

  const dates = sortedData.map(d => d.date)
  const sentiments = sortedData.map(d => d.sentiment)

  // Fixed y-axis ticks for consistent spacing
  const yMin = -1
  const yMax = 1
  const tickVals = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
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
        width: 3,
        shape: 'spline',
        smoothing: 1.25
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
      range: [yMin, yMax],
      tickmode: 'array' as const,
      tickvals: tickVals,
      ticktext: tickVals.map(v => v.toFixed(2).replace('-0.00', '0.00')),
      ticks: 'outside' as const,
      ticklen: 8,
      tickcolor: 'rgba(255, 255, 255, 0.3)',
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
    height: Math.max(height, 360),
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

// Mock timeline data function removed - no longer generating fake data
