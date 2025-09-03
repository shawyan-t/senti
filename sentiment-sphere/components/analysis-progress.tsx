"use client"

import React from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog'

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
      <div className="flex items-center justify-between text-xs text-emerald-200/80 mb-1">
        <span>Analysis Progress</span>
        <span>{Math.floor(clamped)}%</span>
      </div>
      <div className="relative h-3 w-full rounded-full bg-emerald-900/30 overflow-hidden border border-emerald-700/30">
        <div
          className="absolute left-0 top-0 h-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-700"
          style={{ width: `${clamped}%` }}
        />
        {/* Dynamic checkpoints from backend history */}
        {checkpoints.map((cp, idx) => {
          const left = cp.percent
          const done = cp.percent <= clamped
          return (
            <Dialog key={`${cp.stage}-${idx}`}>
              <DialogTrigger asChild>
                <button
                  className={`absolute -top-1 h-5 w-5 rounded-full border transition-colors ${
                    done
                      ? 'bg-emerald-400 border-emerald-100 shadow shadow-emerald-900/40'
                      : 'bg-slate-700 border-slate-500 hover:bg-slate-600'
                  }`}
                  style={{ left: `calc(${left}% - 10px)` }}
                  title={`${cp.label} — click to view details`}
                />
              </DialogTrigger>
              <DialogContent className="sm:max-w-lg">
                <DialogHeader>
                  <DialogTitle className="text-emerald-400">{cp.label}</DialogTitle>
                </DialogHeader>
                <div className="text-sm text-gray-200 leading-relaxed">
                  {cp.metrics && Object.keys(cp.metrics).length > 0 ? (
                    <ul className="list-disc pl-5 space-y-1 text-gray-300">
                      {Object.entries(cp.metrics).map(([k, v]) => (
                        <li key={k}><span className="text-emerald-300">{k}:</span> {String(v)}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-gray-300">No metrics available for this step yet.</p>
                  )}
                </div>
              </DialogContent>
            </Dialog>
          )
        })}
      </div>
      {checkpoints.length > 0 && (
        <div className="mt-1 flex justify-between text-[10px] text-emerald-200/70">
          {checkpoints.map((cp, idx) => (
            <span key={`${cp.stage}-label-${idx}`} className={cp.percent <= clamped ? 'text-emerald-300' : ''}>
              {cp.label}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
