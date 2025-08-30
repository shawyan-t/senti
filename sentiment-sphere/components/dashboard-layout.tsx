"use client"

import React from "react"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"

interface DashboardLayoutProps {
  children: React.ReactNode
  className?: string
}

export function DashboardLayout({ children, className }: DashboardLayoutProps) {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={cn(
        "grid grid-cols-1 md:grid-cols-2 gap-4 mt-6 w-full", 
        className
      )}
    >
      {children}
    </motion.div>
  )
}

export function DashboardPanel({ 
  children, 
  className,
  title,
  subtitle,
}: { 
  children: React.ReactNode
  className?: string
  title: string
  subtitle?: string
}) {
  return (
    <div className={cn(
      "bg-slate-800/70 backdrop-blur-sm rounded-lg p-5 border border-emerald-500/20",
      "overflow-hidden transition-all duration-300 hover:border-emerald-500/40 hover:shadow-md hover:shadow-emerald-500/10",
      className
    )}>
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-emerald-300">{title}</h3>
        {subtitle && (
          <p className="text-xs text-emerald-200/60 mt-0.5">{subtitle}</p>
        )}
      </div>
      <div className="w-full h-full">
        {children}
      </div>
    </div>
  )
} 