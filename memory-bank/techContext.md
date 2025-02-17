# Technical Context

## Technology Stack

### Core Dependencies
1. **Python 3.12+**
   - Type hints
   - Async support
   - Modern language features

2. **FFmpeg**
   - Audio/video processing
   - Format conversion
   - Waveform generation
   - Subtitle rendering

3. **Whisper-timestamped**
   - Audio transcription
   - Word-level timing
   - Multiple model sizes
   - GPU acceleration

4. **litellm**
   - LLM integration
   - Highlight selection
   - Content summarization
   - Provider abstraction

### Development Tools
- **Git**: Version control
- **Poetry**: Dependency management
- **pytest**: Testing framework
- **black**: Code formatting
- **mypy**: Type checking
- **ruff**: Linting

## System Requirements

### Hardware Requirements
- CPU: Multi-core processor
- RAM: 8GB minimum (16GB recommended)
- GPU: Optional, supports CUDA/ROCm
- Storage: SSD recommended for cache

### Software Requirements
- Python 3.12 or higher
- FFmpeg latest stable version
- CUDA toolkit (for GPU support)
- System-level audio codecs

## Project Structure

### Directory Layout
```
karaokify/
├── karaokify.py      # Main video generation
├── transcribe.py     # Audio transcription
├── requirements.txt  # Dependencies
├── fonts/           # Font resources
└── memory-bank/     # Documentation
```

### Module Organization
1. **Transcription Module**
   ```python
   # transcribe.py
   class Transcriber:
       def __init__(self, model_size: str)
       def transcribe(self, audio_path: str) -> dict
       def cache_result(self, key: str, data: dict)
   ```

2. **Video Generation Module**
   ```python
   # karaokify.py
   class VideoGenerator:
       def __init__(self, config: dict)
       def generate(self, transcript: dict) -> str
       def create_effects(self, segment: dict)
   ```

## Configuration

### Environment Variables
```bash
OPENAI_API_KEY=sk-...      # For LLM integration
CACHE_DIR=/path/to/cache   # Cache location
GPU_ENABLED=true          # Enable GPU
MODEL_SIZE=medium         # Whisper model size
```

### Command Line Arguments
```bash
# Transcription
--audio_path    # Input audio file
--model_size    # Whisper model size
--cache         # Enable caching

# Video Generation
--video_input   # Input video/image
--transcript    # Transcript JSON
--output        # Output path
```

## Performance Considerations

### Optimization Strategies
1. **Caching**
   - SHA-256 validation
   - Intermediate file storage
   - Cache cleanup policies

2. **GPU Acceleration**
   - Transcription speedup
   - Parallel processing
   - Memory management

3. **Resource Management**
   - Chunk processing
   - Memory efficiency
   - Temp file cleanup

### Bottlenecks
1. **Transcription**
   - CPU/GPU intensive
   - Model loading time
   - Memory usage

2. **Video Processing**
   - I/O operations
   - Effect rendering
   - Format conversion

## Integration Points

### External APIs
1. **OpenAI/LiteLLM**
   - API key management
   - Rate limiting
   - Error handling

2. **FFmpeg**
   - Command construction
   - Output parsing
   - Error handling

### File Formats
1. **Input**
   - Audio: WAV, MP3
   - Video: MP4, MOV
   - Images: PNG, JPG

2. **Output**
   - Video: MP4
   - Transcript: JSON
   - Cache: Binary

## Security

### Data Protection
- Secure API key storage
- Cache file permissions
- Input validation
- Temp file cleanup

### Error Handling
- Graceful degradation
- Clear error messages
- Logging system
- Recovery procedures

## Deployment

### Installation
```bash
# Basic setup
pip install -r requirements.txt

# GPU support
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```

### Testing
```bash
# Run tests
pytest tests/

# Check types
mypy .

# Lint code
ruff check .
```

## Maintenance

### Monitoring
- Process resource usage
- Cache size/usage
- Error rates
- Processing times

### Updates
- Dependency updates
- Security patches
- Model updates
- Feature additions

### Documentation
- API documentation
- Usage examples
- Configuration guide
- Troubleshooting
