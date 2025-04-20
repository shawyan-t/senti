"use client"

import { useState } from "react"
import { Eye } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

type Analysis = {
  id: number
  name: string
  date: string
}

export function AnalysesList() {
  const [selectedAnalysis, setSelectedAnalysis] = useState<string | null>(null)

  const previousAnalyses: Analysis[] = [
    { id: 1, name: "Swamp Izzo 2025", date: "2025-04-11" },
    { id: 2, name: "Cardi", date: "2025-04-05" },
    { id: 3, name: "swamp izzo", date: "2025-04-19" },
    { id: 4, name: "I AM MUSIC", date: "2025-04-11" },
    { id: 5, name: "I am MUSIC", date: "2025-04-11" },
    { id: 6, name: "Cryptocurrency", date: "2025-04-19" },
    { id: 7, name: "Swamp Izzo", date: "2025-04-09" },
    { id: 8, name: "Luka Doncic", date: "2025-04-19" },
    { id: 9, name: "Swamp Izzo", date: "2025-04-09" },
  ]

  const handleSelectChange = (value: string) => {
    setSelectedAnalysis(value)
  }

  const handleViewAnalysis = () => {
    // Handle view analysis logic here
    console.log("Viewing analysis:", selectedAnalysis)
  }

  return (
    <div className="space-y-4">
      <p>Select a previous analysis to view:</p>
      <Select onValueChange={handleSelectChange}>
        <SelectTrigger className="w-full bg-gray-800 border-gray-700">
          <SelectValue placeholder="Select an analysis" />
        </SelectTrigger>
        <SelectContent className="bg-gray-800 border-gray-700">
          {previousAnalyses.map((analysis) => (
            <SelectItem key={analysis.id} value={analysis.id.toString()}>
              {analysis.name} ({analysis.date})
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <div>
        <Button
          className="bg-[#F05D5E] hover:bg-[#e04a4a] text-white"
          onClick={handleViewAnalysis}
          disabled={!selectedAnalysis}
        >
          <Eye className="mr-2 h-4 w-4" />
          View Analysis
        </Button>
      </div>
    </div>
  )
}
