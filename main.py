"""
Local LLM Service - Main Application
Provides HTTP API for local LLM inference with chat functionality.
Private access without authorization - localhost only.
"""
import os
import asyncio
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
import json

from chat_handler import ChatHandler


# Configuration
class Settings:
    def __init__(self):
        self.model_path: str = os.getenv("MODEL_PATH", "models/llama-2-7b-chat.gguf")
        self.n_ctx: int = int(os.getenv("N_CTX", "2048"))
        self.n_gpu_layers: int = int(os.getenv("N_GPU_LAYERS", "0"))
        self.n_threads: int = int(os.getenv("N_THREADS", "4"))
        self.max_tokens: int = int(os.getenv("MAX_TOKENS", "512"))
        self.temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
        self.host: str = os.getenv("HOST", "127.0.0.1")
        self.port: int = int(os.getenv("PORT", "8000"))


settings = Settings()
chat_handler: Optional[ChatHandler] = None
templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global chat_handler
    # Startup: Load model
    print(f"Loading model from {settings.model_path}...")
    chat_handler = ChatHandler(model_path=settings.model_path)
    chat_handler.load_model(
        n_ctx=settings.n_ctx,
        n_gpu_layers=settings.n_gpu_layers,
        n_threads=settings.n_threads
    )
    print("Model loaded successfully!")
    yield
    # Shutdown: Cleanup
    print("Shutting down...")


app = FastAPI(title="Local LLM Service", version="1.0.0", lifespan=lifespan)

# CORS middleware - allow all for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    message: str
    model: str
    usage: Dict[str, int]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the web chat interface"""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        model_loaded=chat_handler is not None and chat_handler.llm is not None
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - generates a response for the given message.
    Private access without authorization.
    """
    if chat_handler is None or chat_handler.llm is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    max_tokens = request.max_tokens or settings.max_tokens
    temperature = request.temperature if request.temperature is not None else settings.temperature
    
    try:
        response = chat_handler.chat(
            message=request.message,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return ChatResponse(
            message=response,
            model=settings.model_path,
            usage={"prompt_tokens": 0, "completion_tokens": 0}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint - returns response as Server-Sent Events.
    Private access without authorization.
    """
    if chat_handler is None or chat_handler.llm is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    max_tokens = request.max_tokens or settings.max_tokens
    temperature = request.temperature if request.temperature is not None else settings.temperature
    
    async def generate():
        try:
            for token in chat_handler.stream_chat(
                message=request.message,
                max_tokens=max_tokens,
                temperature=temperature
            ):
                yield {
                    "event": "token",
                    "data": json.dumps({"token": token})
                }
            yield {
                "event": "done",
                "data": json.dumps({"done": True})
            }
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
    
    return EventSourceResponse(generate())


@app.get("/api/models")
async def list_models():
    """List available models"""
    models_dir = "models"
    available_models = []
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith(".gguf") or file.endswith(".bin"):
                available_models.append({
                    "name": file,
                    "path": os.path.join(models_dir, file)
                })
    
    return {
        "current_model": settings.model_path,
        "available_models": available_models
    }


if __name__ == "__main__":
    import uvicorn
    print(f"Starting Local LLM Service on {settings.host}:{settings.port}")
    print(f"Model: {settings.model_path}")
    print("Private access - no authorization required")
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port
    )
