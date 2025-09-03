import type React from "react"
import "./globals.css"
import type { Metadata } from "next"
import { Inter } from "next/font/google"

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
})

export const metadata: Metadata = {
  title: "Sentimizer - AI-Powered Sentiment Analysis",
  description: "Advanced AI-Powered Sentiment Analysis Platform",
    generator: 'v0.dev'
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="min-h-screen antialiased bg-gradient-to-b from-slate-900 via-purple-950 to-slate-900 text-gray-200">
        {children}
      </body>
    </html>
  )
}
