# Project Brief: Karaokify

## Overview
Karaokify is a Python-based tool that automatically generates karaoke-style highlight videos from audio/video content. It transforms long-form content into engaging, visually appealing videos with synchronized subtitles, waveform visualizations, and smooth transitions.

## Core Requirements

### Input Flexibility
- Accept either video files (containing both audio and background) or separate audio and background files
- Support various audio formats (WAV, MP3) and video formats (MP4)
- Handle long-form content like podcasts, interviews, lectures, etc.

### Transcription (transcribe.py)
- Utilize Whisper-timestamped for accurate audio transcription
- Generate word-level timestamps and confidence scores
- Clean and remove duplicate/redundant segments
- Output JSON format with precise timing information

### Video Generation (karaokify.py)
- Extract key highlights using LLM-based selection
- Create smooth audio transitions with crossfading
- Generate dynamic waveform visualizations
- Implement karaoke-style word-by-word subtitle highlighting
- Support customizable video dimensions, fonts, and styling
- Cache intermediate files for efficiency

## Project Goals
1. Automate the creation of engaging highlight videos
2. Maintain high audio-visual quality with professional transitions
3. Provide flexible customization options
4. Ensure efficient processing with caching mechanisms
5. Support both full-length and highlight reel outputs

## Technical Scope
- Python 3.12+ environment
- FFmpeg dependency for audio/video processing
- LLM integration via litellm for highlight selection
- GPU acceleration support where available
- Modular design for easy maintenance and extension

## Success Criteria
- Accurate transcription with word-level timing
- Smooth audio transitions between segments
- Visually appealing waveform overlay
- Synchronized karaoke-style subtitles
- Efficient caching of intermediate files
- Support for both full content and highlight reels
