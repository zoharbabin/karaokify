#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transcribe an audio file using the Whisper model from the whisper-timestamped library,
then clean up duplicate or redundant segments, and finally save the results as JSON.

Usage:
    1. Install dependencies:
        pip install -r requirements.txt
    2. Run the script:
         python transcribe.py \
            --audio_path="path/to/audio.wav" \
            --model_size="medium" \
            --output_path="transcript_cleaned.json"

Available command-line arguments:
    --audio_path      Path to the audio file to transcribe. (Required)
    --model_size      Which Whisper model variant to load (default: "medium").
    --device          Device to run inference on (e.g., "cpu", "cuda", or "mps").
                      If not specified, the script will attempt to pick the best device
                      automatically (CUDA, then MPS, then CPU).
    --output_path     JSON file path to save the cleaned transcript (default: "transcript_cleaned.json").
"""

import argparse
import json
import torch
import whisper_timestamped as whisper


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Object containing parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file using Whisper and clean duplicates."
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to the audio file to transcribe.",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="medium",
        help='Which Whisper model variant to load (e.g. "small", "medium", "large").',
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            'Device to run inference on (e.g., "cpu", "cuda", or "mps"). '
            "If not provided, the script will pick the best available device automatically."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="transcript_cleaned.json",
        help="JSON file path to save the cleaned transcript.",
    )
    return parser.parse_args()


def pick_best_device():
    """
    Determine the best available device for PyTorch inference:
    1. CUDA if available
    2. MPS (Apple Silicon) is not yet supported by whisper, so avoid that
    3. Otherwise CPU

    Returns:
        str: The name of the best device ("cuda", or "cpu").
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def clean_transcription(transcription_result):
    """
    Remove duplicate or redundant segments from a transcription result.

    Args:
        transcription_result (dict): Dictionary containing transcription output
            with a "segments" key.

    Returns:
        dict: A copy of `transcription_result` with cleaned segments.
    """
    segments = transcription_result.get("segments", [])
    cleaned_segments = []
    last_text = None

    for segment in segments:
        if segment["text"] != last_text:  # Skip duplicates
            cleaned_segments.append(segment)
        last_text = segment["text"]

    transcription_result["segments"] = cleaned_segments
    return transcription_result


def main():
    """
    Main function to load the model, transcribe the audio, clean the result,
    and save the cleaned transcript to JSON.
    """
    # Parse command-line arguments
    args = parse_args()

    # Determine which device to use
    final_device = args.device if args.device else pick_best_device()

    print(f"Using device: {final_device}")

    # Load audio file
    print(f"Loading audio file: {args.audio_path}")
    audio = whisper.load_audio(args.audio_path)

    # Load the Whisper model
    print(f"Loading Whisper model: {args.model_size} on device '{final_device}'")
    model = whisper.load_model(args.model_size, device=final_device)

    # Configuration for transcription
    transcription_config = {
        "language": "en",  # Force English transcription
        "task": "transcribe",  # Speech recognition (not translation)
        "vad": "silero",  # Use Voice Activity Detection to remove non-speech segments
        "detect_disfluencies": True,  # Detect and include disfluencies
        "trust_whisper_timestamps": True,  # Trust Whisper's native timestamps
        "compute_word_confidence": True,  # Compute word-level confidence scores
        "include_punctuation_in_confidence": True,  # Punctuation confidence
        "min_word_duration": 0.1,  # Minimum duration for word timestamps
        "plot_word_alignment": False,  # Disable plotting alignment
        "compression_ratio_threshold": 2.4,  # Avoid highly repetitive outputs
        "logprob_threshold": -1.0,  # Filter low-confidence outputs
        "no_speech_threshold": 0.6,  # Adjust silence sensitivity
        "beam_size": 5,  # Beam search size
        "best_of": 5,  # Consider top candidates
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8),  # Fallback decoding temps
        "condition_on_previous_text": True,  # Context-aware transcription
        "refine_whisper_precision": 0.02,  # Precision for refining timestamps
        "remove_empty_words": True,  # Remove empty words
    }

    # Transcribe the audio
    print("Transcribing audio...")
    result = whisper.transcribe(model, audio, **transcription_config)

    # Clean the transcription by removing duplicates
    print("Cleaning up transcript...")
    cleaned_result = clean_transcription(result)

    # Save the cleaned transcript to JSON
    print(f"Saving cleaned transcript to '{args.output_path}'")
    with open(args.output_path, "w", encoding="utf-8") as output_file:
        json.dump(cleaned_result, output_file, indent=2, ensure_ascii=False)

    print(f"Cleaned transcript saved as '{args.output_path}'")


if __name__ == "__main__":
    main()
