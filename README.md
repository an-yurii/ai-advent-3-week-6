# Local LLM Service

A private local LLM service with HTTP API and web chat interface. No authorization required - designed for local/private use.

## Features

- 🔒 **Private Access** - Runs locally, no authorization required
- 🌐 **HTTP API** - RESTful endpoints for chat integration
- 💬 **Web Chat Interface** - Beautiful browser-based chat UI
- ⚡ **Streaming Support** - Real-time token streaming via SSE
- 🎛️ **Configurable** - Easy configuration via environment variables

## Requirements

- Python 3.10+
- 8GB+ RAM (depending on model size)
- A GGUF format model file (e.g., Llama-2-7B-Chat)

## Installation

### 1. Clone or create the project structure

```bash
cd /path/to/ai-advent-3-week-6
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download a model

Download a GGUF format model and place it in the `models/` directory:

```bash
# Example: Download Llama-2-7B-Chat GGUF
mkdir -p models
cd models
# Download from Hugging Face or other sources
# Example URL (check for latest):
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
cd ..
```

Popular model sources:
- [TheBloke on Hugging Face](https://huggingface.co/TheBloke) - Quantized GGUF models
- [Llama.cpp models](https://github.com/ggerganov/llama.cpp#model-repository)

### 5. Configure (optional)

Copy the example environment file and adjust settings:

```bash
cp .env.example .env
```

Edit `.env` to configure:
- `MODEL_PATH` - Path to your model file
- `N_CTX` - Context window size
- `N_GPU_LAYERS` - Set > 0 for GPU acceleration
- `HOST` - Bind address (default: 127.0.0.1 for local only)
- `PORT` - Server port (default: 8000)

## Usage

### Start the server

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Access the chat interface

Open your browser and navigate to:

```
http://127.0.0.1:8000
```

### API Endpoints

#### Health Check
```bash
curl http://127.0.0.1:8000/health
```

#### Chat (JSON response)
```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "max_tokens": 256, "temperature": 0.7}'
```

#### Chat (Streaming)
```bash
curl -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a story"}'
```

#### List Models
```bash
curl http://127.0.0.1:8000/api/models
```

## Configuration Options

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `MODEL_PATH` | Path to GGUF model file | `models/llama-2-7b-chat.gguf` |
| `N_CTX` | Context window size | `2048` |
| `N_GPU_LAYERS` | GPU layers (0 = CPU only) | `0` |
| `N_THREADS` | CPU threads | `4` |
| `MAX_TOKENS` | Max generation tokens | `512` |
| `TEMPERATURE` | Sampling temperature | `0.7` |
| `HOST` | Server bind address | `127.0.0.1` |
| `PORT` | Server port | `8000` |

## Security Notes

⚠️ **This service is designed for local/private use without authorization.**

- By default, binds to `127.0.0.1` (localhost only)
- No authentication or rate limiting
- **Do not expose to public networks** without adding security measures
- For production use, consider adding:
  - API key authentication
  - Rate limiting
  - HTTPS/TLS
  - Network firewall rules

## Troubleshooting

### Model not loading
- Ensure the model file exists at the specified path
- Check that you have enough RAM for the model
- Verify the model is in GGUF format

### Slow inference
- Increase `N_GPU_LAYERS` if you have a compatible GPU
- Reduce `N_CTX` for smaller context window
- Use a smaller/quantized model (Q4_K_M recommended)

### Out of memory
- Use a smaller model (7B instead of 13B/70B)
- Use more quantized model (Q4_K_S or Q3_K_M)
- Reduce `N_CTX` value

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Efficient LLM inference
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - Python bindings
