import type React from "react"
import "./globals.css"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import { Analytics } from "@vercel/analytics/next"

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
})

export const metadata: Metadata = {
  title: "Sentimizer - AI-Powered Sentiment Analysis",
  description: "Advanced AI-Powered Sentiment Analysis Platform",
  generator: 'v0.dev',
  themeColor: '#0f172a', // Dark slate to match page background
  viewport: 'width=device-width, initial-scale=1, viewport-fit=cover',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={inter.variable}>
      <head>
        <meta name="theme-color" content="#0f172a" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-title" content="Sentimizer" />
        <style dangerouslySetInnerHTML={{
          __html: `
            html, body {
              background: linear-gradient(to bottom, #0f172a, #581c87, #0f172a) !important;
              min-height: 100vh !important;
              min-height: -webkit-fill-available !important;
            }
            html {
              height: -webkit-fill-available;
            }
          `
        }} />
      </head>
      <body className="min-h-screen antialiased bg-gradient-to-b from-slate-900 via-purple-950 to-slate-900 text-gray-200" style={{minHeight: '100vh', background: 'linear-gradient(to bottom, #0f172a, #581c87, #0f172a)'}}>
        {children}
        <Analytics />
      </body>
    </html>
  )
}
