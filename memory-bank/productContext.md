# Product Context

## Problem Statement
Content creators and educators need an efficient way to transform long-form audio/video content into engaging, visually appealing highlight videos. Manual creation of such videos is time-consuming and requires significant video editing expertise.

## Solution
Karaokify automates the process of creating karaoke-style highlight videos by:
1. Transcribing audio with precise word-level timing
2. Intelligently selecting key segments using LLM technology
3. Generating visually appealing videos with synchronized subtitles and waveform visualization
4. Providing smooth transitions between segments

## User Experience

### Content Creator Flow
1. Prepare input:
   - Either a video file containing both audio and visuals
   - Or separate audio file and background image/video
2. Run transcription:
   ```bash
   python transcribe.py --audio_path="content.wav" --model_size="medium"
   ```
3. Generate video:
   ```bash
   python karaokify.py --video_input="content.mp4" --transcript="transcript.json"
   ```
4. Optionally customize:
   - Video dimensions and quality
   - Font styles and sizes
   - Highlight duration and transitions
   - Waveform appearance

### End User Experience
- Clear, synchronized subtitles that highlight words as they're spoken
- Professional-looking waveform visualization
- Smooth transitions between segments
- High-quality audio with crossfaded transitions
- Engaging highlight reels that maintain context and flow

## Key Benefits

### For Content Creators
- Save hours of manual video editing time
- Create consistent, professional-looking outputs
- Easily generate highlight reels from long content
- Maintain control over visual styling
- Efficient caching of intermediate files

### For Viewers
- Better engagement through visual elements
- Easier content comprehension with synchronized text
- Professional-quality viewing experience
- Quick access to key content through highlights

## Use Cases
1. **Podcast Highlights**
   - Transform audio podcasts into shareable video clips
   - Create episode teasers and promotional content

2. **Educational Content**
   - Convert lectures into engaging video segments
   - Create study materials with synchronized text

3. **Interview Highlights**
   - Extract and present key moments from interviews
   - Generate promotional clips for social media

4. **Conference Talks**
   - Create highlight reels from presentations
   - Generate shareable clips of key insights

## Success Metrics
1. Processing Efficiency
   - Fast transcription with high accuracy
   - Efficient video generation with caching
   - Quick highlight selection through LLM

2. Output Quality
   - Clear, readable subtitles
   - Smooth audio transitions
   - Professional visual appearance
   - Accurate word timing

3. User Satisfaction
   - Minimal manual intervention needed
   - Consistent, reliable results
   - Flexible customization options
   - Intuitive command-line interface
