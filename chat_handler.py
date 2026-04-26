"""
Chat Handler Module
Handles chat message processing and streaming responses.
"""
from typing import Optional, Dict, Any, Generator, AsyncGenerator
from llama_cpp import Llama


class ChatHandler:
    """Handles chat interactions with the LLM"""
    
    def __init__(self, model_path: str, max_tokens: int = 512):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.llm: Optional[Llama] = None
    
    def load_model(self, n_ctx: int = 2048, n_gpu_layers: int = 0, n_threads: int = 4):
        """Load the model from disk"""
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads
        )
        return self.llm
    
    def chat(self, message: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate a response for the given message"""
        if self.llm is None:
            raise RuntimeError("Model not loaded")
        
        output = self.llm(
            message,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\n", "###", "User:", "user:"]
        )
        return output["choices"][0]["text"].strip()
    
    def stream_chat(self, message: str, max_tokens: int = 512, temperature: float = 0.7) -> Generator[str, None, None]:
        """Stream chat responses token by token"""
        if self.llm is None:
            raise RuntimeError("Model not loaded")
        
        output = self.llm(
            message,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\n", "###", "User:", "user:"],
            stream=True
        )
        
        for chunk in output:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("text", "")
                if delta:
                    yield delta
