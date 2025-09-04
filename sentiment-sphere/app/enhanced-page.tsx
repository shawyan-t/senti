"use client"

import { useState } from "react"
import { MoreVertical } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { TextInput } from "@/components/text-input"
import { FileUpload } from "@/components/file-upload"
import { AnalysesList } from "@/components/analyses-list"

export default function EnhancedPage() {
  const [activeTab, setActiveTab] = useState("text")

  const handleTabChange = (value: string) => {
    setActiveTab(value)
  }

  return (
    <div className="flex flex-col min-h-screen">
      <header className="flex justify-between items-center p-4">
        <div className="flex-1"></div>
        <div className="flex items-center gap-2">
          <Button variant="ghost">Deploy</Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon">
                <MoreVertical className="h-5 w-5" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem>Settings</DropdownMenuItem>
              <DropdownMenuItem>Help</DropdownMenuItem>
              <DropdownMenuItem>Logout</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      <main className="flex-1 container mx-auto px-4 py-8 max-w-6xl">
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-[#5EABD7] mb-2">SENTIMIZER</h1>
          <p className="text-gray-400">Advanced AI-Powered Sentiment Analysis Platform</p>
        </div>

        <div className="bg-white/5 h-8 sm:h-16 rounded-lg mb-6"></div>

        <Tabs defaultValue="text" onValueChange={handleTabChange} className="w-full">
          <TabsList className="border-b border-gray-700 bg-transparent w-full justify-start rounded-none h-auto p-0 mb-6">
            <TabsTrigger
              value="text"
              className={`px-4 py-2 rounded-none data-[state=active]:border-b-2 data-[state=active]:border-[#F05D5E] data-[state=active]:bg-transparent data-[state=active]:shadow-none ${activeTab === "text" ? "text-white" : "text-gray-400"}`}
            >
              Text/URL Input
            </TabsTrigger>
            <TabsTrigger
              value="upload"
              className={`px-4 py-2 rounded-none data-[state=active]:border-b-2 data-[state=active]:border-[#F05D5E] data-[state=active]:bg-transparent data-[state=active]:shadow-none ${activeTab === "upload" ? "text-white" : "text-gray-400"}`}
            >
              Upload File
            </TabsTrigger>
            <TabsTrigger
              value="analyses"
              className={`px-4 py-2 rounded-none data-[state=active]:border-b-2 data-[state=active]:border-[#F05D5E] data-[state=active]:bg-transparent data-[state=active]:shadow-none ${activeTab === "analyses" ? "text-white" : "text-gray-400"}`}
            >
              My Analyses
            </TabsTrigger>
          </TabsList>

          <TabsContent value="text" className="mt-0">
            <TextInput />
          </TabsContent>

          <TabsContent value="upload" className="mt-0">
            <FileUpload />
          </TabsContent>

          <TabsContent value="analyses" className="mt-0">
            <AnalysesList />
          </TabsContent>
        </Tabs>

        <div className="bg-white rounded-lg p-6 sm:p-8 mt-6 sm:mt-8 text-center text-gray-700">
          <h2 className="text-2xl font-semibold text-gray-600 mb-4">Welcome to Sentimizer!</h2>
          <p>Enter text, a URL, or upload a file above to begin analyzing sentiment.</p>
        </div>
      </main>

      <footer className="mt-auto border-t border-gray-800 py-4 text-center text-sm text-gray-500">
        <span suppressHydrationWarning>Powered by OpenAI&apos;s GPT models • © 2025 Sentimizer</span>
      </footer>
    </div>
  )
}
