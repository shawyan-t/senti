"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Cloud, Upload } from "lucide-react"
import { Button } from "@/components/ui/button"

export function FileUpload() {
  const [isDragging, setIsDragging] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0]
      setFile(droppedFile)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0])
    }
  }

  const handleBrowseClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click()
    }
  }

  return (
    <div className="space-y-4">
      <p>Upload a file to analyze (PDF, CSV, JSON, or text file)</p>
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          isDragging ? "border-[#5EABD7] bg-[#5EABD7]/5" : "border-gray-700"
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="flex flex-col items-center justify-center gap-4">
          <Cloud className="h-12 w-12 text-gray-500" />
          <div className="space-y-2">
            <p className="text-xl">{file ? file.name : "Drag and drop file here"}</p>
            <p className="text-sm text-gray-500">Limit 200MB per file • PDF, CSV, JSON, TXT</p>
          </div>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".pdf,.csv,.json,.txt"
            className="hidden"
          />
          <Button variant="outline" className="mt-2" onClick={handleBrowseClick}>
            Browse files
          </Button>
        </div>
      </div>
      <div>
        <Button variant="outline" className="bg-gray-800 text-white" disabled={!file}>
          <Upload className="mr-2 h-4 w-4" />
          Analyze File
        </Button>
      </div>
    </div>
  )
}
