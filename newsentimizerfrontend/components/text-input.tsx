"use client"

import type React from "react"

import { useState } from "react"
import { Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"

export function TextInput() {
  const [text, setText] = useState("")

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value)
  }

  const handleAnalyze = () => {
    // Handle analysis logic here
    console.log("Analyzing:", text)
  }

  return (
    <div className="space-y-4">
      <p>Enter text, URL, or paste content to analyze:</p>
      <Textarea
        placeholder="Enter a URL, article, financial report, or any text you want to analyze..."
        className="min-h-[200px] bg-gray-800 border-gray-700"
        value={text}
        onChange={handleTextChange}
      />
      <div className="flex justify-between items-center">
        <p className="text-sm text-gray-400">
          Type or paste text, enter a URL, or simply ask a question about a topic.
        </p>
        <Button className="bg-[#F05D5E] hover:bg-[#e04a4a] text-white" onClick={handleAnalyze} disabled={!text.trim()}>
          <Search className="mr-2 h-4 w-4" />
          Analyze
        </Button>
      </div>
    </div>
  )
}
