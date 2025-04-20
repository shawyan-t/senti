from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
from datetime import datetime
import os

# Import utility functions
from utils.text_processor import (
    process_text_input, 
    extract_text_from_pdf, 
    extract_text_from_html,
    detect_file_type,
    process_csv_data
)
from utils.openai_client import (
    determine_input_type, 
    perform_detailed_analysis, 
    analyze_sentiment, 
    extract_metadata
)
from utils.data_manager import save_analysis, load_analysis, load_all_analyses
from utils.external_data import get_online_sentiment, get_online_sentiment_with_search
from utils.visualizations import (
    create_3d_globe_visualization,
    create_interest_over_time_chart,
    create_topic_popularity_chart,
    create_keyword_chart
)

app = FastAPI(title="Sentimizer API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Allow frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

class AnalysisRequest(BaseModel):
    content: str
    type: str = "text"
    options: Optional[Dict[str, Any]] = None

@app.post("/api/analyze/text")
async def analyze_text(input_data: TextInput):
    try:
        print(f"Received text analysis request: {input_data.text[:50]}...")
        
        # Process text input
        processed_text = process_text_input(input_data.text)
        print("Text processed successfully")
        
        # Determine input type
        input_type_info = determine_input_type(processed_text)
        print(f"Input type determined: {input_type_info}")
        
        # Perform analysis
        print("Performing detailed analysis...")
        analysis_result = perform_detailed_analysis(processed_text, input_type_info)
        print("Analysis completed")
        
        print("Analyzing sentiment...")
        sentiment = analyze_sentiment(processed_text)
        print("Sentiment analysis completed")
        
        print("Extracting metadata...")
        metadata = extract_metadata(processed_text)
        print("Metadata extraction completed")
        
        # Save analysis
        print("Saving analysis...")
        analysis_data = {
            "source": input_data.text[:50] + "..." if len(input_data.text) > 50 else input_data.text,
            "content": processed_text,
            "analysis": analysis_result,
            "sentiment": sentiment,
            "metadata": metadata,
            "source_type": "text_input",
            "timestamp": datetime.now().isoformat()
        }
        analysis_id = save_analysis(analysis_data)
        print(f"Analysis saved with ID: {analysis_id}")
        
        return {
            "analysis_id": analysis_id,
            "analysis": analysis_result,
            "sentiment": sentiment,
            "metadata": metadata
        }
    except Exception as e:
        print(f"Error in analyze_text: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/file")
async def analyze_file(file: UploadFile = File(...)):
    try:
        print(f"Received file analysis request: {file.filename}")
        
        file_content = await file.read()
        file_type = detect_file_type(file.filename)
        print(f"File type detected: {file_type}")
        
        if file_type == "pdf":
            text = extract_text_from_pdf(file_content)
        elif file_type == "csv":
            text = process_csv_data(file_content)
        else:
            text = file_content.decode()
        
        print("File content extracted")
        
        # Determine input type
        input_type_info = determine_input_type(text)
        print(f"Input type determined: {input_type_info}")
        
        # Perform analysis
        print("Performing detailed analysis...")
        analysis_result = perform_detailed_analysis(text, input_type_info)
        print("Analysis completed")
        
        print("Analyzing sentiment...")
        sentiment = analyze_sentiment(text)
        print("Sentiment analysis completed")
        
        print("Extracting metadata...")
        metadata = extract_metadata(text)
        print("Metadata extraction completed")
    
        # Save analysis
        print("Saving analysis...")
        analysis_data = {
            "source": file.filename,
            "content": text[:1000] + "..." if len(text) > 1000 else text,  # Truncate long content
            "analysis": analysis_result,
            "sentiment": sentiment,
            "metadata": metadata,
            "source_type": file_type,
            "timestamp": datetime.now().isoformat()
        }
        analysis_id = save_analysis(analysis_data)
        print(f"Analysis saved with ID: {analysis_id}")
        
        return {
            "analysis_id": analysis_id,
            "analysis": analysis_result,
            "sentiment": sentiment,
            "metadata": metadata
        }
    except Exception as e:
        print(f"Error in analyze_file: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analyses")
async def get_analyses():
    try:
        print("Loading all analyses...")
        # Print the analyses directory path for debugging
        from utils.data_manager import ANALYSES_DIR, DATA_DIR
        print(f"Analyses directory: {ANALYSES_DIR}")
        print(f"Data directory: {DATA_DIR}")
        print(f"Analyses directory exists: {os.path.exists(ANALYSES_DIR)}")
        print(f"Data directory exists: {os.path.exists(DATA_DIR)}")
        
        # List files in both directories
        if os.path.exists(ANALYSES_DIR):
            analyses_files = os.listdir(ANALYSES_DIR)
            print(f"Files in analyses directory: {len(analyses_files)}")
            print(f"First few files: {analyses_files[:5] if len(analyses_files) > 0 else 'No files'}")
        
        if os.path.exists(DATA_DIR):
            data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
            print(f"JSON files in data directory: {len(data_files)}")
            print(f"First few files: {data_files[:5] if len(data_files) > 0 else 'No files'}")
        
        analyses = load_all_analyses()
        print(f"Loaded {len(analyses)} analyses")
        
        # Print keys of first few analyses for debugging
        if analyses:
            print(f"First few analyses keys: {list(analyses.keys())[:5]}")
            first_key = next(iter(analyses))
            print(f"Sample analysis structure: {list(analyses[first_key].keys())}")
        
        return analyses
            
    except Exception as e:
        print(f"Error in get_analyses: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    try:
        analysis = load_analysis(analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return analysis
    except Exception as e:
        print(f"Error in get_analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/online-sentiment")
async def get_online_sentiment_analysis(query: str = Form(...)):
    try:
        sentiment = get_online_sentiment_with_search(query)
        return sentiment
    except Exception as e:
        print(f"Error in get_online_sentiment_analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)