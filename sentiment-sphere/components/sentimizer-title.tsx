"use client"

import { useEffect, useRef, useState } from "react"
import { motion, AnimatePresence } from "framer-motion"

interface SentimizerTitleProps {
  onAnimationComplete: () => void
}

export default function SentimizerTitle({ onAnimationComplete }: SentimizerTitleProps) {
  const [animationComplete, setAnimationComplete] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Animation for the intro sequence
  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimationComplete(true)
      onAnimationComplete()
    }, 3500) // Animation duration before transitioning to static title

    return () => clearTimeout(timer)
  }, [onAnimationComplete])

  // Canvas animation effect
  useEffect(() => {
    if (!canvasRef.current || animationComplete) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Set canvas dimensions
    const setCanvasDimensions = () => {
      const { width, height } = canvas.getBoundingClientRect()
      const dpr = window.devicePixelRatio || 1
      canvas.width = width * dpr
      canvas.height = height * dpr
      ctx.scale(dpr, dpr)
      return { width, height }
    }

    const { width, height } = setCanvasDimensions()

    // Animation variables
    let animationFrameId: number
    let progress = 0
    const text = "SENTIMIZER"
    const fontSize = Math.min(width / 8, width < 640 ? 48 : 80) // Smaller on mobile
    const particles: Particle[] = []
    const particleCount = 200

    // Create initial particles
    for (let i = 0; i < particleCount; i++) {
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        size: Math.random() * 3 + 1,
        speedX: (Math.random() - 0.5) * 3,
        speedY: (Math.random() - 0.5) * 3,
        color: `rgba(${16 + Math.random() * 20}, ${185 + Math.random() * 40}, ${129 + Math.random() * 40}, ${0.5 + Math.random() * 0.5})`,
      })
    }

    // Animation loop
    const animate = () => {
      ctx.clearRect(0, 0, width, height)

      // Update progress
      progress += 0.01
      if (progress > 1) progress = 1

      // Draw particles
      particles.forEach((particle) => {
        particle.x += particle.speedX
        particle.y += particle.speedY

        // Wrap around edges
        if (particle.x < 0) particle.x = width
        if (particle.x > width) particle.x = 0
        if (particle.y < 0) particle.y = height
        if (particle.y > height) particle.y = 0

        // Draw particle
        ctx.fillStyle = particle.color
        ctx.beginPath()
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2)
        ctx.fill()
      })

      // Draw text with increasing clarity
      ctx.save()
      ctx.font = `bold ${fontSize}px "Aerospace", sans-serif`
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"

      // Text shadow/glow effect
      const glowSize = 10 * progress
      ctx.shadowColor = "rgba(16, 185, 129, 0.8)"
      ctx.shadowBlur = glowSize

      // Draw text with gradient
      const gradient = ctx.createLinearGradient(width / 2 - 150, height / 2, width / 2 + 150, height / 2)
      gradient.addColorStop(0, "#10b981")
      gradient.addColorStop(0.5, "#5eead4")
      gradient.addColorStop(1, "#10b981")

      ctx.fillStyle = gradient
      ctx.globalAlpha = progress
      ctx.fillText(text, width / 2, height / 2)

      // Draw tech lines around text
      if (progress > 0.3) {
        const lineProgress = (progress - 0.3) / 0.7
        ctx.strokeStyle = "rgba(16, 185, 129, " + lineProgress * 0.7 + ")"
        ctx.lineWidth = 1

        // Horizontal lines
        const lineCount = 5
        const lineSpacing = fontSize / (lineCount + 1)

        for (let i = 0; i < lineCount; i++) {
          const y = height / 2 - fontSize / 2 + lineSpacing * (i + 1)
          const lineWidth = 200 * lineProgress

          ctx.beginPath()
          ctx.moveTo(width / 2 - lineWidth, y)
          ctx.lineTo(width / 2 + lineWidth, y)
          ctx.stroke()
        }

        // Vertical accent lines
        ctx.beginPath()
        ctx.moveTo(width / 2 - 120, height / 2 - fontSize / 2 - 10 * lineProgress)
        ctx.lineTo(width / 2 - 120, height / 2 + fontSize / 2 + 10 * lineProgress)
        ctx.stroke()

        ctx.beginPath()
        ctx.moveTo(width / 2 + 120, height / 2 - fontSize / 2 - 10 * lineProgress)
        ctx.lineTo(width / 2 + 120, height / 2 + fontSize / 2 + 10 * lineProgress)
        ctx.stroke()
      }

      ctx.restore()

      animationFrameId = requestAnimationFrame(animate)
    }

    // Handle window resize
    const handleResize = () => {
      setCanvasDimensions()
    }

    window.addEventListener("resize", handleResize)
    animate()

    // Cleanup
    return () => {
      window.removeEventListener("resize", handleResize)
      cancelAnimationFrame(animationFrameId)
    }
  }, [animationComplete])

  return (
    <div className="w-full h-full flex items-center justify-center">
      <AnimatePresence mode="wait">
        {!animationComplete ? (
          <motion.div
            key="animation"
            className="absolute inset-0"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
          >
            <canvas ref={canvasRef} className="w-full h-full" />
          </motion.div>
        ) : (
          <motion.div
            key="static"
            className="flex items-center justify-center w-full"
            initial={{ opacity: 0, scale: 1.1 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
          >
            <h1 className="text-3xl sm:text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 via-teal-300 to-emerald-500 tracking-wider uppercase font-tech">
              SENTIMIZER
            </h1>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// TypeScript interface for particles
interface Particle {
  x: number
  y: number
  size: number
  speedX: number
  speedY: number
  color: string
}
