from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import logging
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uvicorn
from typing import List, Optional
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Personal RAG Chatbot API",
    description="API for querying Theja Ashwin's personal knowledge base using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    category_filter: Optional[List[str]] = None
    max_results: Optional[int] = 5

class QueryResponse(BaseModel):
    response: str
    context_sources: List[str]
    categories_found: List[str]
    status: str

class IndexStatus(BaseModel):
    status: str
    message: str
    total_vectors: Optional[int] = None
    index_name: str

class HealthResponse(BaseModel):
    status: str
    message: str
    embedding_model: str
    index_name: str

# ------------------------------
# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check for required API keys
if not PINECONE_API_KEY:
    raise Exception("PINECONE_API_KEY not found in environment variables!")

if not GEMINI_API_KEY:
    raise Exception("GEMINI_API_KEY not found in environment variables!")

# ------------------------------
# Initialize Pinecone instance
pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index_name = "personal-data-index-v2"  # New index name to avoid conflicts

# Check if index exists and handle dimension mismatch
def setup_pinecone_index():
    """Setup Pinecone index with proper error handling"""
    try:
        # Try to get existing index info
        index_info = pc.describe_index(index_name)
        current_dimension = index_info.dimension
        required_dimension = 768
        
        if current_dimension != required_dimension:
            logger.warning(f"Index dimension mismatch: found {current_dimension}, need {required_dimension}")
            logger.info("Deleting old index and creating new one...")
            pc.delete_index(index_name)
            import time
            time.sleep(5)  # Wait for deletion
            
            # Create new index with correct dimensions
            create_new_index()
        else:
            logger.info(f"✅ Using existing index with correct dimensions ({current_dimension})")
            
    except Exception as e:
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            logger.info("Index not found, creating new one...")
            create_new_index()
        else:
            logger.error(f"Error accessing index: {str(e)}")
            raise Exception(f"Error accessing index: {str(e)}")

def create_new_index():
    """Create new index in a supported region"""
    # Free tier regions (try in order of preference)
    regions = ["us-east-1"]
    
    for region in regions:
        try:
            logger.info(f"Attempting to create index in {region}...")
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=region)
            )
            logger.info(f"✅ Successfully created index in {region}!")
            return
        except Exception as region_error:
            error_msg = str(region_error)
            if "free plan" in error_msg.lower() or "upgrade" in error_msg.lower():
                logger.warning(f"Region {region} not supported on free plan")
                continue
            else:
                logger.warning(f"Failed in {region}: {error_msg}")
                continue
    
    # If we get here, all regions failed
    raise Exception("❌ Failed to create index in any supported region.")

# Setup the index
setup_pinecone_index()

index = pc.Index(index_name)

# ------------------------------
# Initialize embedding model
embedding_model = SentenceTransformer('all-mpnet-base-v2')

# Initialize Gemini client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize text splitter for intelligent chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
)

# Global flag to track if data is loaded
data_loaded = False

# ------------------------------
# Load personal data from JSON
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def upsert_data(data):
    """Upsert data with intelligent chunking for better retrieval"""
    total_items = len(data)
    
    for idx, (key, item) in enumerate(data.items()):
        try:
            logger.info(f"Processing {key}... ({idx+1}/{total_items})")
            
            # Handle different data structures
            if isinstance(item, list):
                # Handle list items (like achievements_awards, certifications_training, etc.)
                if key == "projects":
                    # Special handling for projects array
                    for i, project in enumerate(item):
                        if isinstance(project, dict):
                            try:
                                project_text = f"Project: {project.get('name', 'Unknown')}\n"
                                project_text += f"Description: {project.get('description', '')}\n"
                                project_text += f"Tech Stack: {', '.join(project.get('tech_stack', []))}\n"
                                if project.get('source'):
                                    project_text += f"Source: {project.get('source')}\n"
                                if project.get('live_demo'):
                                    project_text += f"Live Demo: {project.get('live_demo')}\n"
                                
                                process_text_chunk(f"{key}project{i}", project_text, "projects", "project_details")
                            except Exception as project_error:
                                logger.error(f"Error processing project {i}: {project_error}")
                                logger.error(f"Project data: {project}")
                                continue
                else:
                    # Handle other list items
                    text = "; ".join(str(item_text) for item_text in item if item_text)
                    if text:
                        category = get_category_from_key(key)
                        process_text_chunk(key, text, category, key)
            
            elif isinstance(item, dict):
                # Handle dictionary items
                if key == "contact_links":
                    # Special handling for contact links
                    contact_text = ""
                    for contact_key, contact_value in item.items():
                        if contact_value:
                            contact_text += f"{contact_key}: {contact_value}\n"
                    if contact_text:
                        process_text_chunk(key, contact_text, "contact", "contact_information")
                
                elif "content" in item:
                    # Handle items with content field
                    content = item["content"]
                    if isinstance(content, list):
                        # Content is a list (like education_background, work_experience)
                        if key == "work_experience":
                            # Special handling for work experience
                            text = ""
                            for exp in content:
                                if isinstance(exp, dict):
                                    text += f"Role: {exp.get('role', 'Unknown')}\n"
                                    text += f"Company: {exp.get('company', 'Unknown')}\n"
                                    if exp.get('duration'):
                                        text += f"Duration: {exp.get('duration')}\n"
                                    if exp.get('responsibilities'):
                                        text += f"Responsibilities: {'; '.join(exp.get('responsibilities', []))}\n"
                                    text += "\n"
                        else:
                            text = "; ".join(str(c) for c in content if c)
                    else:
                        # Content is a string
                        text = str(content)
                    
                    category = item.get("metadata", {}).get("category", get_category_from_key(key))
                    item_type = item.get("type", key)
                    
                    if text and len(text.strip()) > 0:
                        process_text_chunk(key, text, category, item_type)
                
                else:
                    # Handle other dictionary structures
                    text = str(item)
                    if text and len(text.strip()) > 0:
                        category = get_category_from_key(key)
                        process_text_chunk(key, text, category, key)
            
            else:
                # Handle simple string/other types
                text = str(item)
                if text and len(text.strip()) > 0:
                    category = get_category_from_key(key)
                    process_text_chunk(key, text, category, key)
            
        except Exception as e:
            logger.error(f"Error processing {key}: {str(e)}")
            continue
    
    logger.info("Data indexed successfully with intelligent chunking.")

def get_category_from_key(key):
    """Map keys to categories"""
    category_mapping = {
        "profile": "personal_info",
        "education_background": "education", 
        "work_experience": "experience",
        "technical_skills": "skills",
        "projects": "projects",
        "achievements_awards": "achievements",
        "certifications_training": "certifications",
        "testimonials_feedback": "testimonials",
        "open_source_contributions": "contributions",
        "personal_interests": "personal",
        "languages": "personal",
        "contact_links": "contact"
    }
    return category_mapping.get(key, "general")

def process_text_chunk(key, text, category, item_type):
    """Process and upsert a text chunk"""
    if not text or len(text.strip()) < 10:  # Skip very short texts
        return
        
    # Split long texts into chunks
    if len(text) > 200:
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{key}chunk{i}"
            vector = embedding_model.encode(chunk, convert_to_tensor=False)
            vector = vector.tolist()
            
            # Verify vector dimension
            if len(vector) != 768:
                logger.warning(f"Skipping {chunk_id}: Vector dimension {len(vector)} != 768")
                return
            
            metadata = {
                "text": chunk,
                "category": category,
                "type": item_type,
                "source_key": key,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            
            index.upsert([(chunk_id, vector, metadata)])
    else:
        # For short texts, index as single chunk
        vector = embedding_model.encode(text, convert_to_tensor=False)
        vector = vector.tolist()
        
        # Verify vector dimension
        if len(vector) != 768:
            logger.warning(f"Skipping {key}: Vector dimension {len(vector)} != 768")
            return
        
        metadata = {
            "text": text,
            "category": category,
            "type": item_type,
            "source_key": key,
            "chunk_index": 0,
            "total_chunks": 1
        }
        
        index.upsert([(key, vector, metadata)])

def query_index(query_text, category_filter=None, max_results=5):
    """Query index with improved filtering and ranking"""
    query_vector = embedding_model.encode(query_text, convert_to_tensor=False)
    query_vector = query_vector.tolist()
    
    # Determine relevant categories based on query
    query_lower = query_text.lower()
    relevant_categories = category_filter or []
    
    if not category_filter:
        if any(word in query_lower for word in ['project', 'projects', 'work', 'built', 'developed']):
            relevant_categories.append("projects")
        if any(word in query_lower for word in ['skill', 'skills', 'technology', 'technologies', 'programming']):
            relevant_categories.append("skills")
        if any(word in query_lower for word in ['experience', 'job', 'intern', 'work']):
            relevant_categories.append("experience")
        if any(word in query_lower for word in ['education', 'study', 'college', 'university']):
            relevant_categories.append("education")
        if any(word in query_lower for word in ['contact', 'email', 'phone', 'reach']):
            relevant_categories.append("contact")
        if any(word in query_lower for word in ['achievement', 'award', 'recognition', 'hackathon']):
            relevant_categories.append("achievements")
        if any(word in query_lower for word in ['personal', 'about', 'who', 'name']):
            relevant_categories.append("personal_info")
        if any(word in query_lower for word in ['certification', 'training', 'course']):
            relevant_categories.append("certifications")
    
    # Build filter for categories if specific categories are identified
    filter_dict = None
    if relevant_categories:
        filter_dict = {"category": {"$in": relevant_categories}}
    
    # Query with increased top_k and filtering
    result = index.query(
        vector=query_vector,
        top_k=20,  # Increased for better context
        include_metadata=True,
        filter=filter_dict
    )
    
    # Sort by similarity score and return top matches
    matches_sorted = sorted(result['matches'], key=lambda x: x['score'], reverse=True)
    
    # Return top most relevant texts and categories found
    context_texts = [match['metadata']['text'] for match in matches_sorted[:max_results]]
    categories_found = list(set([match['metadata']['category'] for match in matches_sorted[:max_results]]))
    
    return context_texts, categories_found

def generate_gemini_response(query, context_texts):
    """Generate response using Google Gemini API with improved prompting"""
    context = "\n\n".join(context_texts)
    
    prompt = f"""
You are an AI assistant that answers questions ONLY using the provided context about Theja Ashwin.

Context:
{context}

Question: {query}

Rules:
1. Answer ONLY based on the context provided above.
2. If the context does not contain the answer, say "I don't have that information in my knowledge base".
3. Provide concise and structured responses.
4. List projects, skills, or experiences in bullet points when asked.
5. Do not invent, assume, or hallucinate any information.
6. Be specific and include relevant details from the context.
7. If asked about multiple items (like projects), organize them clearly with descriptions.

Answer:
"""

    try:
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=0.1,  # Low temperature for deterministic responses
            max_output_tokens=600,  # Increased for detailed responses
        )

        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=contents,
            config=generate_content_config,
        )
        
        return response.text
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ------------------------------
# API Endpoints

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Personal RAG Chatbot API is running",
        embedding_model="all-mpnet-base-v2",
        index_name=index_name
    )

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base with a question"""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Ensure data is loaded
        await ensure_data_loaded()
        
        # Query the index
        context_texts, categories_found = query_index(
            request.query, 
            request.category_filter, 
            request.max_results
        )
        
        if not context_texts:
            return QueryResponse(
                response="I don't have that information in my knowledge base.",
                context_sources=[],
                categories_found=[],
                status="no_results"
            )
        
        # Generate response using Gemini
        response = generate_gemini_response(request.query, context_texts)
        
        return QueryResponse(
            response=response,
            context_sources=context_texts,
            categories_found=categories_found,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/reindex", response_model=IndexStatus)
async def reindex_data():
    """Reindex the knowledge base data"""
    try:
        global data_loaded
        data_loaded = False
        
        # Clear existing data
        index.delete(delete_all=True)
        logger.info("Cleared all data from index")
        
        # Reload data
        await ensure_data_loaded()
        
        # Get index stats
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        
        return IndexStatus(
            status="success",
            message="Data reindexed successfully",
            total_vectors=total_vectors,
            index_name=index_name
        )
        
    except Exception as e:
        logger.error(f"Error reindexing data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reindexing data: {str(e)}")

@app.get("/index/status", response_model=IndexStatus)
async def get_index_status():
    """Get current index status and statistics"""
    try:
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        
        return IndexStatus(
            status="active",
            message=f"Index is active with {total_vectors} vectors",
            total_vectors=total_vectors,
            index_name=index_name
        )
        
    except Exception as e:
        logger.error(f"Error getting index status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting index status: {str(e)}")

@app.get("/categories")
async def get_available_categories():
    """Get list of available categories for filtering"""
    categories = [
        "personal_info", "education", "experience", "skills", 
        "projects", "achievements", "certifications", "testimonials",
        "contributions", "personal", "contact"
    ]
    return {"categories": categories}

# Helper function to ensure data is loaded
async def ensure_data_loaded():
    """Ensure the knowledge base data is loaded"""
    global data_loaded
    
    if not data_loaded:
        try:
            logger.info("Loading and indexing data...")
            data = load_data("./data/myDetails.json")
            upsert_data(data)
            data_loaded = True
            logger.info("✅ Data loaded and indexed successfully!")
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

# Initialize data on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting Personal RAG Chatbot API...")
    await ensure_data_loaded()
    logger.info("API startup complete!")

if __name__ == "__main__":
    uvicorn.run(
        "fast:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

