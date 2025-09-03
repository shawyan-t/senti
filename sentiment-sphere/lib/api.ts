const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface TextAnalysisRequest {
  text: string;
}

export interface FileAnalysisRequest {
  file: File;
}

export interface Analysis {
  analysis_id: string;
  status: string;
  analysis_type: string;
  source: string;
  text: string;
  enhanced_summary: string;
  // Optional, present for comprehensive pipeline
  comprehensive_metrics?: {
    // Some UI code expects an optional sentiment object
    sentiment?: {
      score?: number;
      confidence?: number;
    };
    disagreement_index?: number;
    polarity_breakdown?: {
      positive: number;
      negative: number;
      neutral: number;
      wilson_ci?: {
        positive?: [number, number];
        negative?: [number, number];
        neutral?: [number, number];
      };
    };
    tone?: {
      subjectivity: number;
      politeness: number;
      formality: number;
      assertiveness: number;
    };
    sarcasm_rate?: number;
    toxicity_rate?: number;
    freshness_score?: number;
    novelty_score?: number;
    total_evidence_weight?: number;
  };
  calculation_methodology?: {
    sentiment_calculation?: string;
    confidence_basis?: string;
    source_weighting?: string;
    accuracy_indicators?: {
      model_agreement?: string;
      source_diversity?: string;
      temporal_coverage?: string;
      data_quality?: string; // 'high' | 'medium' | 'low' etc.
    };
  };
  query_summary?: {
    query: string;
    query_type?: string;
    entities_detected?: string[];
    recent_events?: Array<Record<string, any>>;
  };
  visualizations?: Record<string, any>;
  mathematical_sentiment_analysis: {
    composite_score: {
      value: number;
      confidence_interval: [number, number];
      statistical_significance: number;
    };
    multi_model_validation: any;
    uncertainty_metrics: any;
  };
  emotion_vector_analysis: {
    plutchik_coordinates: Record<string, number>;
    dominant_emotions: string[];
    emotion_entropy: number;
    emotional_coherence: number;
  };
  content_metrics: any;
  // Legacy compatibility
  analysis?: any;
  sentiment?: any;
  metadata?: any;
}

// Helper function to handle API responses
const handleResponse = async (response: Response) => {
  if (!response.ok) {
    // Try to get the error message from the response
    let errorMessage = `Error: ${response.status} ${response.statusText}`;
    try {
      const errorData = await response.json();
      if (errorData.detail) {
        errorMessage = errorData.detail;
      }
    } catch (e) {
      // If we can't parse the JSON, just use the status
      console.error("Error parsing error response:", e);
    }
    console.error("API Error:", errorMessage);
    throw new Error(errorMessage);
  }
  return response.json();
};

export const analyzeText = async (text: string): Promise<Analysis> => {
  console.log(`Sending comprehensive analysis request to ${API_BASE_URL}/api/analyze/comprehensive`);
  const response = await fetch(`${API_BASE_URL}/api/analyze/comprehensive`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ 
      text, 
      use_search_apis: true  // Enable search APIs for enhanced context
    }),
  });

  return handleResponse(response);
};

export const analyzeFile = async (file: File): Promise<Analysis> => {
  console.log(`Sending file analysis request to ${API_BASE_URL}/api/analyze/file`);
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/analyze/file`, {
    method: 'POST',
    body: formData,
  });

  return handleResponse(response);
};

export const getAnalyses = async (): Promise<Record<string, Analysis>> => {
  console.log(`Fetching analyses from ${API_BASE_URL}/api/analyses`);
  const response = await fetch(`${API_BASE_URL}/api/analyses`);
  console.log("Response status:", response.status);
  
  if (response.status === 204) {
    console.log("No analyses found");
    return {}; // Return empty object if no content
  }
  
  return handleResponse(response);
};

export const getAnalysis = async (analysisId: string): Promise<Analysis> => {
  console.log(`Fetching analysis ${analysisId} from ${API_BASE_URL}/api/analysis/${analysisId}`);
  const response = await fetch(`${API_BASE_URL}/api/analysis/${analysisId}`);

  return handleResponse(response);
};

export const getOnlineSentiment = async (query: string): Promise<any> => {
  console.log(`Fetching online sentiment for ${query}`);
  const formData = new FormData();
  formData.append('query', query);

  const response = await fetch(`${API_BASE_URL}/api/online-sentiment`, {
    method: 'POST',
    body: formData,
  });

  return handleResponse(response);
};

// Option B: background job + polling helpers
export const submitAnalysis = async (text: string, use_search_apis = true): Promise<{ task_id: string }> => {
  const response = await fetch(`${API_BASE_URL}/api/analyze/submit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, use_search_apis }),
  });
  return handleResponse(response);
};

export const getAnalysisStatus = async (taskId: string): Promise<{ task_id: string; status: string; analysis_id?: string; error?: string }> => {
  const response = await fetch(`${API_BASE_URL}/api/analyze/status/${taskId}`);
  return handleResponse(response);
};

export const getAnalysisResult = async (taskId: string): Promise<Analysis> => {
  const response = await fetch(`${API_BASE_URL}/api/analyze/result/${taskId}`);
  // If task not completed, server returns 202
  if (response.status === 202) {
    throw new Error('Task not completed');
  }
  return handleResponse(response);
};
