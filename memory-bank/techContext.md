# Technical Context

## Development Environment

### Core Requirements
- Python 3.12 or higher
- FFmpeg installed and available on PATH
- GPU support (optional but recommended)

### Dependencies
#### Transcription (transcribe.py)
```
torch
torchaudio
whisper-timestamped
onnxruntime  # or onnxruntime-gpu if available
```

#### Video Generation (karaokify.py)
```
fonttools
Pillow
litellm
boto3
```

## Technology Stack

### 1. Audio Processing
- **Whisper-timestamped**
  - Modified version of OpenAI's Whisper
  - Enhanced timestamp accuracy
  - Word-level confidence scoring
  - Voice activity detection
  - Disfluency detection

- **FFmpeg**
  - Audio extraction
  - Waveform generation
  - Audio crossfading
  - Video assembly
  - Hardware acceleration support

### 2. Machine Learning
- **LLM Integration**
  - Uses litellm for model access
  - Default model: anthropic.claude-3-5-sonnet
  - Configurable through environment variables
  - Fallback mechanisms for failures

- **GPU Acceleration**
  - CUDA support for Whisper
  - Hardware acceleration for FFmpeg
  - Automatic device selection
  - Fallback to CPU when needed

### 3. Video Processing
- **Subtitle Generation**
  - ASS (Advanced SubStation Alpha) format
  - Word-level timing control
  - Multi-layer support
  - Rich styling capabilities

- **Waveform Visualization**
  - Dynamic generation
  - Customizable appearance
  - Shadow effects
  - Transparent background

### 4. File Management
- **Caching System**
  - SHA-256 hash validation
  - Intermediate file caching
  - Automatic cleanup
  - Cache invalidation

## Technical Constraints

### 1. System Requirements
- FFmpeg must be installed and on PATH
- Sufficient disk space for intermediate files
- Adequate RAM for video processing
- Python 3.12+ environment

### 2. Performance Considerations
- GPU memory for transcription
- Disk I/O for caching
- CPU usage for video processing
- Memory usage for large files

### 3. API Dependencies
- LLM API access (e.g., Anthropic, OpenAI)
- API rate limits
- Authentication requirements
- Cost considerations

## Development Setup

### 1. Installation
```bash
# Clone repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt

# Verify FFmpeg installation
ffmpeg -version
```

### 2. Environment Configuration
```bash
# LLM API Configuration
export ANTHROPIC_API_KEY="your-api-key"
# or
export OPENAI_API_KEY="your-api-key"

# AWS Configuration (if using Bedrock)
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="your-region"
```

### 3. Testing Setup
```bash
# Verify transcription
python transcribe.py --audio_path="test.wav" --model_size="small"

# Verify video generation
python karaokify.py --video_input="test.mp4" --transcript="transcript.json"
```

## Optimization Guidelines

### 1. Performance
- Use GPU acceleration when available
- Implement efficient caching
- Optimize file I/O operations
- Manage memory usage

### 2. Quality
- Balance audio quality vs file size
- Optimize video resolution
- Fine-tune subtitle timing
- Adjust waveform parameters

### 3. Resource Usage
- Clean up temporary files
- Monitor memory consumption
- Manage cache size
- Handle large files efficiently

## Debugging Tools

### 1. FFmpeg
- Debug level logging
- Progress monitoring
- Error tracking
- Format validation

### 2. Python
- Logging configuration
- Error tracebacks
- Memory profiling
- Performance monitoring

### 3. LLM Integration
- Response validation
- Error handling
- Fallback mechanisms
- Rate limit monitoring

## Future Considerations

### 1. Scalability
- Batch processing support
- Parallel processing
- Cloud integration
- API service potential

### 2. Maintenance
- Dependency updates
- Security patches
- Performance improvements
- Feature additions

### 3. Integration
- Additional LLM providers
- New video effects
- Format support
- Platform compatibility
