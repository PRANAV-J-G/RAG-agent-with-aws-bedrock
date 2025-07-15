from fastapi import FastAPI, HTTPException, UploadFile, File, Form 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import boto3
from fastapi import FastAPI,Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates 
import json
import os
from typing import List, Optional
import uuid
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Agent with AWS Bedrock", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    sources: List[dict] = []
    timestamp: str

class KnowledgeBaseStatus(BaseModel):
    knowledge_base_id: str
    status: str
    data_source_id: str
    last_sync: Optional[str] = None

# AWS Configuration
class AWSConfig:
    def __init__(self):
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.knowledge_base_id = os.getenv('KNOWLEDGE_BASE_ID', '')
        self.agent_id = os.getenv('AGENT_ID', '')
        self.agent_alias_id = os.getenv('AGENT_ALIAS_ID', 'TSTALIASID')
        self.s3_bucket = os.getenv('S3_BUCKET', '')
        
        # Initialize AWS clients
        self.bedrock_agent_runtime = boto3.client(
            'bedrock-agent-runtime',
            region_name=self.region
        )
        
        self.bedrock_agent = boto3.client(
            'bedrock-agent',
            region_name=self.region
        )
        
        self.s3_client = boto3.client(
            's3',
            region_name=self.region
        )

config = AWSConfig()

# Session management
sessions = {}


@app.on_event("startup")
async def startup_event():
    logger.info("Starting RAG Agent API")
    logger.info(f"Knowledge Base ID: {config.knowledge_base_id}")
    logger.info(f"Agent ID: {config.agent_id}")


templates = Jinja2Templates(directory="templates")
@app.get("/",response_class=HTMLResponse)
async def root(request:Request):
    return templates.TemplateResponse('index.html',{'request':request})

@app.get("/health")
async def health_check():
    try:
        # Check if Bedrock is accessible
        response = config.bedrock_agent.get_agent(agentId=config.agent_id)
        return {"status": "healthy", "agent_status": response['agent']['agentStatus']}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.post("/upload", response_model=dict)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF document to S3 and trigger knowledge base sync"""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Generate unique filename
        file_key = f"documents/{uuid.uuid4()}_{file.filename}"
        
        # Upload to S3
        config.s3_client.upload_fileobj(
            file.file,
            config.s3_bucket,
            file_key,
            ExtraArgs={'ContentType': 'application/pdf'}
        )
        
        # Trigger knowledge base sync
        sync_response = config.bedrock_agent.start_ingestion_job(
            knowledgeBaseId=config.knowledge_base_id,
            dataSourceId=os.getenv('DATA_SOURCE_ID', ''),
            description=f"Sync after uploading {file.filename}"
        )
        
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "s3_key": file_key,
            "sync_job_id": sync_response['ingestionJob']['ingestionJobId'],
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """Query the RAG agent with a question"""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Query the Bedrock agent
        response = config.bedrock_agent_runtime.invoke_agent(
            agentId=config.agent_id,
            agentAliasId=config.agent_alias_id,
            sessionId=session_id,
            inputText=request.question,
            enableTrace=True
        )
        
        # Process the response
        answer = ""
        sources = []
        
        for event in response['completion']:
            if 'chunk' in event:
                chunk = event['chunk']
                if 'bytes' in chunk:
                    answer += chunk['bytes'].decode('utf-8')
            elif 'trace' in event:
                trace = event['trace']
                if 'trace' in trace:
                    trace_data = trace['trace']
                    if 'knowledgeBaseLookupOutput' in trace_data:
                        kb_output = trace_data['knowledgeBaseLookupOutput']
                        if 'retrievedReferences' in kb_output:
                            for ref in kb_output['retrievedReferences']:
                                sources.append({
                                    'content': ref.get('content', {}).get('text', ''),
                                    'location': ref.get('location', {}),
                                    'score': ref.get('score', 0)
                                })
        
        # Store session
        sessions[session_id] = {
            'last_query': request.question,
            'last_response': answer,
            'timestamp': datetime.now().isoformat()
        }
        
        return QueryResponse(
            answer=answer,
            session_id=session_id,
            sources=sources,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-kb", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base directly (without agent)"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Query knowledge base directly
        response = config.bedrock_agent_runtime.retrieve_and_generate(
            input={
                'text': request.question
            },
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': config.knowledge_base_id,
                    'modelArn': f'arn:aws:bedrock:{config.region}::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0'
                }
            }
        )
        
        answer = response['output']['text']
        sources = []
        
        if 'citations' in response:
            for citation in response['citations']:
                if 'retrievedReferences' in citation:
                    for ref in citation['retrievedReferences']:
                        sources.append({
                            'content': ref.get('content', {}).get('text', ''),
                            'location': ref.get('location', {}),
                            'score': ref.get('score', 0)
                        })
        
        return QueryResponse(
            answer=answer,
            session_id=session_id,
            sources=sources,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Knowledge base query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-base/status", response_model=KnowledgeBaseStatus)
async def get_knowledge_base_status():
    """Get knowledge base status and sync information"""
    try:
        # Get knowledge base info
        kb_response = config.bedrock_agent.get_knowledge_base(
            knowledgeBaseId=config.knowledge_base_id
        )
        
        # Get data source info
        data_sources = config.bedrock_agent.list_data_sources(
            knowledgeBaseId=config.knowledge_base_id
        )
        
        data_source_id = data_sources['dataSourceSummaries'][0]['dataSourceId']
        
        # Get latest ingestion jobs
        ingestion_jobs = config.bedrock_agent.list_ingestion_jobs(
            knowledgeBaseId=config.knowledge_base_id,
            dataSourceId=data_source_id,
            maxResults=1
        )
        
        last_sync = None
        if ingestion_jobs['ingestionJobSummaries']:
            last_sync = ingestion_jobs['ingestionJobSummaries'][0]['updatedAt'].isoformat()
        
        return KnowledgeBaseStatus(
            knowledge_base_id=config.knowledge_base_id,
            status=kb_response['knowledgeBase']['status'],
            data_source_id=data_source_id,
            last_sync=last_sync
        )
        
    except Exception as e:
        logger.error(f"Failed to get knowledge base status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge-base/sync")
async def sync_knowledge_base():
    """Trigger knowledge base synchronization"""
    try:
        # Get data source ID
        data_sources = config.bedrock_agent.list_data_sources(
            knowledgeBaseId=config.knowledge_base_id
        )
        
        data_source_id = data_sources['dataSourceSummaries'][0]['dataSourceId']
        
        # Start sync job
        sync_response = config.bedrock_agent.start_ingestion_job(
            knowledgeBaseId=config.knowledge_base_id,
            dataSourceId=data_source_id,
            description="Manual sync triggered"
        )
        
        return {
            "message": "Sync started successfully",
            "job_id": sync_response['ingestionJob']['ingestionJobId'],
            "status": sync_response['ingestionJob']['status']
        }
        
    except Exception as e:
        logger.error(f"Sync failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return sessions[session_id]

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {"sessions": list(sessions.keys()), "count": len(sessions)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)