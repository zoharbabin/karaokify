# karaokify – Turn Long Audio/Video into Karaoke-Style Highlights Videos

This project allows you to automatically generate a refined, **karaoke-style** video from either a video file or an audio file. It highlights textual segments, crossfades audio clips, overlays waveforms, and burns subtitles into a final MP4 video. The process is ideal for creating highlight reels of podcasts, interviews, lectures, or any long-form audio/video content.

---

## Features

- **Flexible Input**: Accept either a video file (which provides both audio and background) or separate audio and background files.
- **Highlight Extractor**: Uses an LLM (via [litellm](https://pypi.org/project/litellm/)) to intelligently select key segments from a transcript.
- **Crossfade**: Smoothly transitions between highlight segments in the audio.
- **Waveform Overlay**: Generates a colorized waveform with an optional shadow or offset effect.
- **Dynamic Karaoke Subtitles**: Burns time-aligned, word-level subtitles into the final output.
- **Customizable**: Easily adjust font, size, video dimensions, subtitle formatting, and more.

> NOTE: This project was built and tested on latest Mac, but should work or easily adapted to any OS.

---

## Installation

1. **Clone** this repository (or download the script files).

2. **Install dependencies** (Python 3.12 or higher is recommended):
   ```bash
   pip install -r requirements.txt
   ```
   This will install libraries necessary for both:
   - `karaokify.py` (karaoke video generation).
   - `transcribe.py` (Whisper-based audio transcription).
   
3. **Set up LLM credentials**  
   - The script uses `litellm` to call LLM endpoints (e.g., Claude, GPT). Check [litellm's docs](https://pypi.org/project/litellm/) for details on configuring your model keys/tokens if needed for highlight extraction.
   - Configure the model you wish to use in `LITELLM_MODEL_STRING` from the [supported litellm models](https://docs.litellm.ai/docs/providers).
   - If you're using an AWS Bedrock model, make sure to configure the AWS boto3 env vars too.

4. **Ensure you have [FFmpeg](https://ffmpeg.org/download.html) installed and available** on your `PATH`. This script calls `ffmpeg` via `subprocess`.

---

## Generating the Transcript

### `transcribe.py`
We've included a script `transcribe.py` that uses the [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped) library (a modified version of OpenAI's Whisper) to create a JSON transcript from an input audio file. The generated transcript is directly compatible with `karaokify.py`.

Example usage:
```bash
python transcribe.py \
    --audio_path="path/to/audio.wav" \
    --model_size="medium" \
    --output_path="transcript_cleaned.json"
```

**Key arguments**:
- `--audio_path`: Path to the audio file (e.g., `.wav`, `.mp3`).
- `--model_size`: Whisper model variant (e.g., `small`, `medium`, `large`). Defaults to `medium`.
- `--device`: Device to run inference on (e.g., `cuda`, `cpu`). If omitted, the script picks the best device automatically.
- `--output_path`: JSON file path for the cleaned transcript. Defaults to `transcript_cleaned.json`.

The script removes duplicate or redundant segments and saves a final JSON with structure:
```json
{
  "segments": [
    {
      "id": 0,
      "start": 3.24,
      "end": 7.56,
      "text": "Hello world...",
      "words": [
        { "start": 3.24, "end": 3.57, "text": "Hello" },
        { "start": 3.60, "end": 3.80, "text": "world" },
        ...
      ]
    },
    ...
  ]
}
```
If you have your **own** transcription pipeline, ensure it outputs a similar JSON structure (with `"segments"` containing `start`, `end`, `text`, and optionally `words`). Then you can skip `transcribe.py` and jump directly to `karaokify.py`.

---

## Generating the Karaoke-Style Video

### `karaokify.py`
After you have a `transcript.json` file (either from `transcribe.py` or another process), you can generate the karaoke-style video. You can use either a video file as input or separate audio and background files:

Using separate audio and background files:
```bash
python karaokify.py \
    --audio=my_podcast.mp3 \
    --transcript=transcript_cleaned.json \
    --background=background.mp4 \
    --output=final_video.mp4 \
    --title="My Podcast"
```

Or using a video file as input:
```bash
python karaokify.py \
    --video_input=my_lecture.mp4 \
    --transcript=transcript_cleaned.json \
    --output=final_video.mp4 \
    --title="My Lecture"
```

**Common Arguments**:
- `--video_input`: Path to a video file that contains both audio and video (overrides --audio and --background).
- `--audio`: Path to the audio file (e.g., `.mp3`, `.wav`).  
- `--transcript`: Path to the JSON transcript file.  
- `--background`: Path to a background image or video (`.png`, `.jpg`, `.mp4`).  
- `--output`: Filename for the final output video (`default="final_karaoke.mp4"`).  
- `--title`: Text to display at the top of the video.  
- `--temp_dir`: Temporary directory for intermediate outputs (`default="temp_ffmpeg"`).  
- `--font_file`: Path to a TrueType font file (e.g., `OpenSans-Bold.ttf`).  
- `--duration`: If set, only create a highlight reel of this many seconds.  
- `--crossfade_duration`: Overlap (in seconds) between consecutive highlights (`default=1.0`).  
- `--video_width/--video_height`: Dimensions of the output video.  

For more details, run:
```bash
python karaokify.py --help
```

---

## Example: Creating a 90-Second Highlight Reel

If you only want a short highlight reel (e.g., 90 seconds total), you can specify:

Using separate audio and background:
```bash
python karaokify.py \
    --audio=my_podcast.mp3 \
    --transcript=transcript_cleaned.json \
    --background=background.mp4 \
    --duration=90 \
    --crossfade_duration=1.5 \
    --output=highlight_reel.mp4 \
    --title="My Podcast – 90s Reel"
```

Or with a video input:
```bash
python karaokify.py \
    --video_input=my_lecture.mp4 \
    --transcript=transcript_cleaned.json \
    --duration=90 \
    --crossfade_duration=1.5 \
    --output=highlight_reel.mp4 \
    --title="My Lecture – 90s Reel"
```

**The script will do the following**:
1. **LLM-based Highlights**: Uses `litellm` to select ~90s of "best" segments from the transcript.  
2. **Trim & Crossfade**: Extracts those segments from the audio and crossfades them.  
3. **Waveform & Subtitles**: Generates a waveform and word-level subtitles for each segment.  
4. **Overlay**: Places the waveform on the background video, then burns subtitles onto the result.  
5. **Outputs** the single MP4 named `highlight_reel.mp4`.

---

## Contributing

Contributions are encouraged! You can:
- Submit pull requests for bug fixes or improvements.
- Suggest new features, enhancements, or better default styles.
- Add new functionalities (e.g., more advanced transitions, new visualization modes, etc.).
- Report issues or request help via GitHub Issues.

---

## License

This project is open source under the [MIT License](LICENSE). You are free to use, modify, and distribute this software as you see fit. We welcome any contributions back to the community.

---

## Additional Notes

- **FFmpeg**: Required for running `karaokify.py`. Ensure it's installed and on your PATH.  
- **Transcript Format**: The script expects JSON transcripts with `segments` containing `start`, `end`, `text`, etc. For best results, use `transcribe.py`.  
- **Video Input**: When using --video_input, the script automatically extracts the audio and uses the video as background.
- **litellm** usage:  
  - If you are using local or open LLMs, you may need to specify the endpoint in your environment.  
  - For production usage with Claude or GPT, you typically need API keys.  

Enjoy your **karaoke-style** video creation!  
Happy hacking!