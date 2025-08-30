# SentimentSphere 🌍

**Advanced AI-Powered Sentiment Analysis Platform**

A sophisticated sentiment analysis platform that combines multiple OpenAI models with real-time data enrichment and advanced visualizations to provide comprehensive emotional intelligence insights.

## ✨ Features

### 🧠 Multi-Model AI Pipeline
- **GPT-4o**: Real-time sentiment analysis with context awareness
- **GPT-o1**: Deep analytical reasoning for complex content
- **GPT-o3-mini**: Efficient metadata extraction
- **HuggingFace DistilRoBERTa**: Scientific emotion classification

### 🌐 Real-Time Context Enrichment
- Live search engine integration with RAG (Retrieval-Augmented Generation)
- Dynamic context gathering from multiple web sources
- Cache-optimized search results for performance

### 📊 Advanced Visualizations
- **3D UMAP Embeddings**: Sentence-level emotional landscape visualization
- **Plutchik Emotion Mapping**: Scientific 8-emotion framework integration
- **Interactive Charts**: Regional sentiment maps, temporal trends, keyword analysis
- **Real-Time Dashboards**: Dynamic data visualization with Plotly

### 🔄 Multi-Format Input Support
- **Text Analysis**: Direct text input and URL processing
- **File Upload**: PDF, CSV, JSON, TXT file analysis
- **Batch Processing**: Multiple document analysis capabilities

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ 
- Python 3.8+
- OpenAI API Key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/SentimentSphere.git
   cd SentimentSphere
   ```

2. **Set up the frontend**
   ```bash
   cd sentiment-sphere
   npm install
   ```

3. **Set up the backend**
   ```bash
   cd api
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Add your API keys
   echo "OPENAI_API_KEY=your_openai_api_key_here" >> sentiment-sphere/.env
   ```

5. **Run the application**
   ```bash
   # Terminal 1: Start the API backend
   cd sentiment-sphere/api
   python main.py
   
   # Terminal 2: Start the frontend
   cd sentiment-sphere
   npm run dev
   ```

6. **Open your browser**
   Navigate to `http://localhost:3000`

## 🏗️ Architecture

```
SentimentSphere/
├── sentiment-sphere/           # Main application
│   ├── app/                   # Next.js 15 frontend
│   ├── components/            # React components
│   ├── api/                   # FastAPI backend
│   │   ├── utils/            # Python utilities
│   │   └── main.py           # API server
│   ├── lib/                   # TypeScript utilities
│   └── data/                  # Analysis storage
├── docs/                      # Documentation
├── scripts/                   # Utility scripts
└── README.md                 # This file
```

## 🔬 Technical Differentiators

### Emotional Vector Mathematics
- **Plutchik's Emotion Wheel**: Mathematical relationships between emotions
- **Valence-Arousal Modeling**: 2D circumplex model with mathematical precision
- **Emotion Entropy**: Measure emotional complexity and uncertainty

### Scientific Integration
- **Psychological Affect Models**: Integration with established psychology frameworks
- **Dimensional Emotion Space**: Multi-dimensional emotional state representation
- **Temporal Emotion Dynamics**: Analysis of emotional patterns over time

## 📖 API Documentation

### Core Endpoints

#### Analyze Text
```bash
POST /api/analyze/text
Content-Type: application/json

{
  "text": "Your text or URL here",
  "use_search_apis": true
}
```

#### Analyze File
```bash
POST /api/analyze/file
Content-Type: multipart/form-data

file: <uploaded_file>
use_search_apis: true
```

#### Get Analyses
```bash
GET /api/analyses
```

## 🧪 Example Analysis Output

```json
{
  "analysis_id": "uuid-here",
  "sentiment": {
    "sentiment": "positive",
    "score": 0.85,
    "confidence": 0.92,
    "rationale": "Analysis rationale...",
    "sentiment_trend": "improving"
  },
  "emotions": [
    {"emotion": "Joy", "score": 0.45},
    {"emotion": "Trust", "score": 0.32},
    {"emotion": "Anticipation", "score": 0.23}
  ],
  "embeddings": [[...], [...], [...]],
  "metadata": {
    "topics": ["technology", "innovation"],
    "regions": ["North America"],
    "entities": ["AI", "OpenAI"]
  }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- HuggingFace for transformer models
- Plotly for visualization capabilities
- The open-source community for various libraries

## 📧 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/SentimentSphere](https://github.com/yourusername/SentimentSphere)