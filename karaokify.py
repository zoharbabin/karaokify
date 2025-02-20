#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import re
import subprocess
import hashlib
import time
from pprint import pprint

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

import litellm

LITELLM_MODEL_STRING = "anthropic.claude-3-5-sonnet-20241022-v2:0"


def parse_arguments():
    """
    Parse command-line arguments for creating a refined, karaoke-style video.
    """
    parser = argparse.ArgumentParser(
        prog="karaokify.py",
        description=(
            "Create a refined, karaoke-style video with two-color waveform "
            "and no black background. This script allows you to generate "
            "karaoke-style videos from either a video file or separate audio "
            "and background files. Features include highlight reels, crossfaded "
            "audio segments, waveform overlay, and burned-in subtitles."
        ),
        epilog=(
            "Example:\n"
            "  python karaokify.py --audio=my_podcast.mp3 --transcript=transcript.json \\\n"
            "       --background=background.mp4 --output=final_video.mp4 --title=\"My Podcast\"\n\n"
            "For a short highlight reel (90 seconds) with crossfade:\n"
            "  python karaokify.py --audio=my_podcast.mp3 --transcript=transcript.json \\\n"
            "       --background=background.mp4 --duration=90 --crossfade_duration=1.5\n\n"
            "Using a video file as input:\n"
            "  python karaokify.py --video_input=my_video.mp4 --transcript=transcript.json \\\n"
            "       --output=final_video.mp4 --title=\"My Video\"\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--video_input",
        help="Path to the input video file that contains audio and video for karaoke. If provided, it overrides --audio and --background."
    )
    parser.add_argument(
        "--audio",
        required=False,
        help="Path to the audio file (e.g., .mp3 or .wav)."
    )
    parser.add_argument(
        "--transcript",
        required=True,
        help="Path to the JSON transcript file."
    )
    parser.add_argument(
        "--background",
        required=False,
        help="Path to the background image or video (e.g., .png, .jpg, .mp4)."
    )
    parser.add_argument(
        "--output",
        default="final_karaoke.mp4",
        help="Output video file name."
    )
    parser.add_argument(
        "--title",
        default="My Podcast Title",
        help="Title text to display at the top."
    )
    parser.add_argument(
        "--temp_dir",
        default="temp_ffmpeg",
        help="Temp directory for intermediate files."
    )
    parser.add_argument(
        "--font_file",
        default="./fonts/OpenSansBold.ttf",
        help=(
            "Path to a modern TTF font file (e.g., OpenSans-Bold.ttf). "
            "Should be a full path or relative path to this script."
        )
    )
    parser.add_argument(
        "--video_width",
        type=int,
        default=1024,
        help="Output video width."
    )
    parser.add_argument(
        "--video_height",
        type=int,
        default=1024,
        help="Output video height."
    )
    parser.add_argument(
        "--waveform_height",
        type=int,
        default=200,
        help="Height of the waveform overlay."
    )
    parser.add_argument(
        "--waveform_fps",
        type=int,
        default=30,
        help="Frame rate for the generated waveform video."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help=(
            "Max duration of highlight reel (in seconds). "
            "If omitted, uses full length."
        )
    )
    parser.add_argument(
        "--crossfade_duration",
        type=float,
        default=1.0,
        help="Seconds of overlap between consecutive highlights."
    )

    return parser.parse_args()


class CacheManager:
    """
    Manages file caching with SHA-256 validation and cleanup.
    Implements check-compute-store pattern from system patterns.
    """
    def __init__(self, temp_dir):
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        
    def get_file_hash(self, file_path: str, block_size: int = 65536) -> str:
        """
        Calculate SHA-256 hash of a file for cache validation.
        
        Args:
            file_path: Path to the file to hash
            block_size: Size of blocks to read for memory efficiency
            
        Returns:
            str: Hexadecimal SHA-256 hash of the file
        """
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for block in iter(lambda: f.read(block_size), b""):
                    sha256_hash.update(block)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"Error calculating file hash for {file_path}: {e}")
            return ""

    def get_cached_file(self, cache_key: str, source_file: str = None) -> str:
        """
        Check if a cached file exists and is valid using SHA-256 validation.
        
        Args:
            cache_key: Base name of the cached file
            source_file: If provided, validate cache against this source file's hash
            
        Returns:
            str: Path to valid cached file or empty string if invalid/missing
        """
        cached_file = os.path.join(self.temp_dir, cache_key)
        hash_file = os.path.join(self.temp_dir, f"{cache_key}.hash")
        
        # Check if both cache file and hash file exist
        if not os.path.exists(cached_file) or not os.path.exists(hash_file):
            return ""
        
        try:
            # If source file provided, validate against its current hash
            if source_file:
                current_hash = self.get_file_hash(source_file)
                if not current_hash:  # Error calculating hash
                    return ""
                    
                with open(hash_file, 'r') as f:
                    cached_hash = f.read().strip()
                    
                if current_hash != cached_hash:
                    # Hash mismatch - source file has changed
                    self.cleanup_invalid_cache(cache_key)
                    return ""
                    
            # Validate the cached file's hash matches its hash file
            cached_file_hash = self.get_file_hash(cached_file)
            if not cached_file_hash:  # Error calculating hash
                return ""
                
            # Additional validation of cached file integrity
            if os.path.getsize(cached_file) == 0:
                self.cleanup_invalid_cache(cache_key)
                return ""
                
            return cached_file
            
        except Exception as e:
            print(f"Error validating cache for {cache_key}: {e}")
            self.cleanup_invalid_cache(cache_key)
            return ""

    def save_file_hash(self, file_path: str, cache_key: str) -> bool:
        """
        Save hash of source file for future cache validation.
        
        Args:
            file_path: Path to the file to hash
            cache_key: Base name for the cache files
            
        Returns:
            bool: True if hash was saved successfully, False otherwise
        """
        try:
            hash_file = os.path.join(self.temp_dir, f"{cache_key}.hash")
            file_hash = self.get_file_hash(file_path)
            if not file_hash:
                return False
                
            with open(hash_file, 'w') as f:
                f.write(file_hash)
            return True
        except Exception as e:
            print(f"Error saving file hash for {cache_key}: {e}")
            return False
            
    def cleanup_invalid_cache(self, cache_key: str) -> None:
        """
        Clean up invalid cache files for a given cache key.
        
        Args:
            cache_key: Base name of the cache files to clean up
        """
        try:
            cached_file = os.path.join(self.temp_dir, cache_key)
            hash_file = os.path.join(self.temp_dir, f"{cache_key}.hash")
            
            if os.path.exists(cached_file):
                os.remove(cached_file)
            if os.path.exists(hash_file):
                os.remove(hash_file)
        except Exception as e:
            print(f"Error cleaning up cache for {cache_key}: {e}")
            
    def cleanup_old_cache(self, max_age_days: int = 7) -> None:
        """
        Clean up cache files older than specified days.
        
        Args:
            max_age_days: Maximum age of cache files in days
        """
        try:
            current_time = time.time()
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > (max_age_days * 24 * 60 * 60):
                        os.remove(file_path)
        except Exception as e:
            print(f"Error cleaning up old cache: {e}")


def run_ffmpeg_command(cmd, step_description):
    """
    Execute an FFmpeg command, printing the step description and the command
    being run, then checking for errors.
    """
    print(f"\n=== {step_description} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def filter_filler_segments(transcript, min_length=5):
    """
    Remove filler segments from the transcript.
    """
    filler_pattern = re.compile(r"\b(?:um|uh|er|ah|hmm)\b", re.IGNORECASE)

    filtered_segments = []
    for seg in transcript["segments"]:
        text = seg["text"]
        text = text.replace("[*]", "")
        text = re.sub(filler_pattern, "", text)
        cleaned_text = " ".join(text.split())

        if len(cleaned_text) >= min_length:
            filtered_segments.append(
                {
                    "id": seg["id"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": cleaned_text,
                }
            )

    return {"segments": filtered_segments}


def extract_segments_transcript_data(transcript):
    """
    Extract segment-level data and filter filler segments.
    """
    relevant_segments = []
    for segment in transcript["segments"]:
        relevant_segments.append(
            {
                "id": segment["id"],
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
            }
        )
    per_line_transcript = {"segments": relevant_segments}
    per_line_transcript = filter_filler_segments(per_line_transcript)
    return per_line_transcript


def fallback_highlights(reduced_transcript, max_duration, crossfade_duration):
    """
    Fallback to a simple highlight selection by taking segments sequentially
    until max_duration is reached.
    Adjusts the last segment to not exceed max_duration and subtracts the
    crossfade_duration from non-final segments.
    """
    segments = sorted(reduced_transcript["segments"], key=lambda s: s["start"])
    highlights = []
    total = 0.0
    for seg in segments:
        seg_duration = seg["end"] - seg["start"]
        if total + seg_duration > max_duration:
            remaining = max_duration - total
            if remaining < 2:  # skip segments too short
                break
            new_seg = seg.copy()
            new_seg["end"] = seg["start"] + remaining
            highlights.append(new_seg)
            total += remaining
            break
        else:
            highlights.append(seg)
            total += seg_duration

    # Subtract crossfade_duration from non-final segments
    for i in range(len(highlights) - 1):
        highlights[i]["end"] = max(highlights[i]["start"] + 2, highlights[i]["end"] - crossfade_duration)
    return {"highlights": highlights}


def get_highlight_segments(transcript, max_duration, crossfade_duration=0):
    """
    Use a language model to pick highlight segments up to `max_duration`.
    Each highlight segment is padded by crossfade_duration seconds on non-final segments.
    """
    reduced_transcript = extract_segments_transcript_data(transcript)

    prompt = (
        f"Below I provided the transcript of a long audio/video.\n"
        "Your task is to produce a list of \"highlight segments\"; the segments that "
        "best describe all the important, interesting, insightful, noteworthy, or "
        "discussion-worthy parts.\n\n"
        "Cover the entire transcript content, and be diligent and accurate.\n"
        "Give each highlight segment a title that best describes in up to 4 words "
        "the context and meaning of what is discussed in this segment text.\n"
        "Make sure that the last highlight segment ends at a thought-provoking statement, "
        "making the user want more.\n"
        f"Each highlight segment must be at least 2 seconds long, and pad non-final segments "
        f"with {crossfade_duration} seconds for crossfade.\n"
        f"The max_duration is: {max_duration}.\n\n"
        "Return your answer as JSON:\n"
        "{\n"
        '  "highlights": [\n'
        "     {\n"
        "       \"start\": 10.5,\n"
        "       \"end\": 20.3,\n"
        "       \"text\": \"The text of this segment\",\n"
        "       \"segment_title\": \"Up to 4 words contextual title\"\n"
        "     },\n"
        "     ...\n"
        "  ]\n"
        "}\n\n"
        f"Transcript: {json.dumps(reduced_transcript)}"
    )

    response = litellm.completion(
        model=LITELLM_MODEL_STRING,
        messages=[
            {"role": "user", "content": prompt},
            {
                "role": "system",
                "content": (
                    "You are a JSON API for creating highlights in a video transcript. "
                    "You MUST fill up to ~100% of the user-provided max_duration, "
                    'Return your answer as a JSON object, like this: '
                    '"highlights": [{"start": 6.18,"end": 13.36, "text": "Summary"}]'
                ),
            },
        ],
        temperature=0,
    )

    try:
        highlights = json.loads(response.choices[0].message.content)
        # Check total duration and validity
        total_duration = sum(h["end"] - h["start"] for h in highlights.get("highlights", []))
        if total_duration < 0.5 * max_duration:
            raise ValueError("LLM returned insufficient highlight duration.")
        # Adjust crossfade: subtract crossfade_duration from non-final segments
        for i, h in enumerate(highlights.get("highlights", [])):
            if i < len(highlights["highlights"]) - 1:
                h["end"] = max(h["start"] + 2, h["end"] - crossfade_duration)
        return highlights
    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        print("Error parsing LLM response or insufficient highlights. Falling back to simple highlight selection.")
        return fallback_highlights(reduced_transcript, max_duration, crossfade_duration)


def trim_audio_segments(input_path, highlights_data, audio_output_path, video_output_path=None, crossfade_duration=1.0):
    """
    Extract each highlight clip from the input and crossfade them together.
    If video_output_path is provided, also extract and crossfade video segments.
    """
    highlights = highlights_data.get("highlights", [])
    if not highlights:
        print("No highlights found, skipping trim_segments.")
        return

    highlights.sort(key=lambda x: x["start"])
    tmp_dir = os.path.dirname(audio_output_path)
    clip_files = []
    video_clip_files = []

    # Calculate total duration for video padding
    total_duration = 0
    for i, seg in enumerate(highlights):
        if i > 0:
            total_duration -= crossfade_duration  # Account for overlap
        total_duration += seg["end"] - seg["start"]

    for i, seg in enumerate(highlights):
        start = seg["start"]
        end = seg["end"]
        duration = end - start
        clip_filename = os.path.join(tmp_dir, f"clip_{i}.m4a")
        video_clip_filename = os.path.join(tmp_dir, f"clip_{i}.mp4") if video_output_path else None

        # Extract both audio and video if video output is requested
        if video_output_path:
            # Extract video segment with audio
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(start),
                "-t", str(duration),
                "-i", input_path,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-b:a", "192k",
                video_clip_filename,
            ]
            run_ffmpeg_command(cmd, f"Extracting highlight clip {i}")
            video_clip_files.append(video_clip_filename)
            
            # Extract audio from the video clip for crossfading
            cmd = [
                "ffmpeg",
                "-y",
                "-i", video_clip_filename,
                "-vn",
                "-acodec", "aac",
                "-b:a", "192k",
                clip_filename,
            ]
            run_ffmpeg_command(cmd, f"Extracting audio from highlight clip {i}")
        else:
            # Audio only extraction
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(start),
                "-t", str(duration),
                "-i", input_path,
                "-vn",
                "-acodec", "aac",
                "-b:a", "192k",
                clip_filename,
            ]
            run_ffmpeg_command(cmd, f"Extracting highlight clip {i}")
        
        clip_files.append(clip_filename)

    if len(clip_files) == 1:
        os.rename(clip_files[0], audio_output_path)
        if video_output_path and video_clip_files:
            os.rename(video_clip_files[0], video_output_path)
        return

    # Crossfade audio clips
    num_clips = len(clip_files)
    audio_filter_lines = []
    for i in range(num_clips - 1):
        left_label = f"[{i}:a]" if i == 0 else f"[a{i}]"
        right_label = f"[{i+1}:a]"
        out_label = f"[a{i+1}]"
        line = (
            f"{left_label}{right_label}acrossfade="
            f"d={crossfade_duration}:curve1=tri:curve2=tri"
            f"{out_label}"
        )
        audio_filter_lines.append(line)

    final_audio_label = f"[a{num_clips-1}]"
    audio_filtergraph = ";".join(audio_filter_lines)

    # Crossfade audio
    audio_cmd = ["ffmpeg", "-y"]
    for cf in clip_files:
        audio_cmd += ["-i", cf]
    audio_cmd += [
        "-filter_complex", audio_filtergraph,
        "-map", final_audio_label,
        "-c:a", "aac",
        "-b:a", "192k",
        audio_output_path,
    ]
    run_ffmpeg_command(audio_cmd, "Crossfading audio clips")

    # Crossfade video if requested
    if video_output_path and video_clip_files:
        video_filter_lines = []
        current_offset = 0
        for i in range(num_clips - 1):
            left_label = f"[{i}:v]" if i == 0 else f"[v{i}]"
            right_label = f"[{i+1}:v]"
            out_label = f"[v{i+1}]"
            
            # Calculate duration of current clip
            clip_duration = highlights[i]["end"] - highlights[i]["start"]
            # Offset for next transition is current position minus crossfade duration
            transition_offset = current_offset + clip_duration - crossfade_duration
            
            line = (
                f"{left_label}{right_label}xfade="
                f"transition=fade:duration={crossfade_duration}:"
                f"offset={transition_offset}"
                f"{out_label}"
            )
            video_filter_lines.append(line)
            
            # Update offset for next clip
            current_offset = transition_offset + crossfade_duration

        final_video_label = f"[v{num_clips-1}]"
        video_filtergraph = ";".join(video_filter_lines)

        video_cmd = ["ffmpeg", "-y"]
        for vf in video_clip_files:
            video_cmd += ["-i", vf]
        video_cmd += [
            "-filter_complex", video_filtergraph,
            "-map", final_video_label,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            video_output_path,
        ]
        run_ffmpeg_command(video_cmd, "Crossfading video clips")


def chunk_into_subsegments(shifted_words, overall_start, overall_end, segment_id_start, max_chars=40, max_lines=2):
    """
    Break shifted_words for a highlight into multiple subtitle subsegments.
    """
    def line_len(words_in_line):
        return sum(len(w["text"]) for w in words_in_line) + (len(words_in_line) - 1) if words_in_line else 0

    subsegments = []
    current_id = segment_id_start
    current_lines = []

    def begin_line():
        current_lines.append([])

    begin_line()
    for w in shifted_words:
        active_line = current_lines[-1]
        new_len_if_added = line_len(active_line + [w])
        if new_len_if_added <= max_chars:
            active_line.append(w)
        else:
            if len(current_lines) < max_lines:
                begin_line()
                current_lines[-1].append(w)
            else:
                seg_words = [wd for line_wds in current_lines for wd in line_wds]
                if seg_words:
                    sub_start = max(min(x["start"] for x in seg_words), overall_start)
                    sub_end = min(max(x["end"] for x in seg_words), overall_end)
                    sub_text = " ".join(x["text"] for x in seg_words)
                    subsegments.append({
                        "id": current_id,
                        "start": sub_start,
                        "end": sub_end,
                        "text": sub_text,
                        "words": seg_words,
                    })
                    current_id += 1
                current_lines = []
                begin_line()
                current_lines[-1].append(w)
    seg_words = [wd for line_wds in current_lines for wd in line_wds]
    if seg_words:
        sub_start = max(min(x["start"] for x in seg_words), overall_start)
        sub_end = min(max(x["end"] for x in seg_words), overall_end)
        if sub_end > sub_start:
            sub_text = " ".join(x["text"] for x in seg_words)
            subsegments.append({
                "id": current_id,
                "start": sub_start,
                "end": sub_end,
                "text": sub_text,
                "words": seg_words,
            })
            current_id += 1
    return subsegments, current_id


def realign_transcript(original_segments, highlight_data, crossfade_duration=1.0):
    """
    Re-map word-level times to match the spliced audio timeline after trimming.
    Then chunk each highlight's words into smaller segments.
    Handles crossfade transitions by adjusting word timings in overlap regions.
    """
    highlights = highlight_data.get("highlights", [])
    if not highlights:
        return []

    # First pass: Calculate exact timeline positions
    timeline_positions = []
    current_time = 0.0
    
    for i, h in enumerate(highlights):
        segment_length = h["end"] - h["start"]
        if i < len(highlights) - 1:
            effective_length = segment_length - crossfade_duration
        else:
            effective_length = segment_length
            
        timeline_positions.append({
            "start": current_time,
            "orig_start": h["start"],
            "orig_end": h["end"],
            "end": current_time + segment_length,
            "effective_end": current_time + effective_length,
            "crossfade_start": current_time + effective_length if i < len(highlights) - 1 else None
        })
        current_time += effective_length

    all_segments = []
    id_counter = 1

    # Process each highlight segment
    for i, (highlight, timeline) in enumerate(zip(highlights, timeline_positions)):
        # Add title/section header
        notice_id = id_counter
        id_counter += 1
        title_text = highlight.get("segment_title", f"Highlight #{i+1}")
        # Create chapter title segment with guaranteed valid duration
        notice_start = round(timeline["start"], 2)
        notice_end = round(min(notice_start + 3, timeline["effective_end"] - 1), 2)
        if notice_end <= notice_start:
            # If the highlight is too short, at least show it for 1 second
            notice_end = notice_start + 1.0

        highlight_notice_seg = {
            "id": notice_id,
            "start": notice_start,
            "end": notice_end,
            "text": f"- {title_text} -",
            "words": [],
        }
        all_segments.append(highlight_notice_seg)

        # Process words
        matched_words = []
        for seg in original_segments:
            if seg["end"] <= highlight["start"] or seg["start"] >= highlight["end"]:
                continue
                
            for w in seg.get("words", []):
                if w["end"] <= highlight["start"] or w["start"] >= highlight["end"]:
                    continue
                if w["text"].strip() == "[*]":
                    continue

                # Calculate relative position in original segment
                orig_rel_start = (w["start"] - highlight["start"]) / (highlight["end"] - highlight["start"])
                orig_rel_end = (w["end"] - highlight["start"]) / (highlight["end"] - highlight["start"])

                # Map to new timeline
                new_start = timeline["start"] + (orig_rel_start * (timeline["effective_end"] - timeline["start"]))
                new_end = timeline["start"] + (orig_rel_end * (timeline["effective_end"] - timeline["start"]))

                # Handle crossfade regions
                if timeline["crossfade_start"] is not None and new_end > timeline["crossfade_start"]:
                    # Word extends into crossfade region
                    if new_start < timeline["crossfade_start"]:
                        # Word starts before crossfade
                        crossfade_portion = (new_end - timeline["crossfade_start"]) / crossfade_duration
                        new_end = timeline["crossfade_start"] + (crossfade_portion * crossfade_duration * 0.5)
                    else:
                        # Word entirely in crossfade
                        relative_pos = (new_start - timeline["crossfade_start"]) / crossfade_duration
                        compression = 0.8 - (relative_pos * 0.3)  # Gradually decrease duration
                        word_duration = new_end - new_start
                        new_duration = word_duration * compression
                        new_start = timeline["crossfade_start"] + (relative_pos * crossfade_duration)
                        new_end = new_start + new_duration

                matched_words.append({
                    "text": w["text"],
                    "start": round(new_start, 3),
                    "end": round(new_end, 3),
                })

        matched_words.sort(key=lambda x: x["start"])
        clipped_words = [w for w in matched_words if w["end"] <= timeline["effective_end"]]

        subsegments, id_counter = chunk_into_subsegments(
            clipped_words,
            overall_start=timeline["start"],
            overall_end=timeline["effective_end"],
            segment_id_start=id_counter,
            max_chars=40,
            max_lines=2,
        )
        all_segments.extend(subsegments)
    return all_segments


def get_ttf_font_name(ttf_path):
    """
    Extract a usable internal font name from the TTF file.
    """
    font = TTFont(ttf_path)
    name_record = None
    for record in font["name"].names:
        if record.nameID == 4 and record.platformID in (1, 3):
            name_str = record.toUnicode()
            if name_str:
                name_record = name_str
                break
    if not name_record:
        for record in font["name"].names:
            if record.nameID == 1 and record.platformID in (1, 3):
                name_str = record.toUnicode()
                if name_str:
                    name_record = name_str
                    break
    return name_record if name_record else "UnknownFont"


def assign_subtitle_layers(segments):
    """
    Assign each subtitle segment a layer to avoid on-screen overlap.
    Chapter titles are positioned at the top like the main title.
    """
    segments_sorted = sorted(segments, key=lambda s: s["start"])
    active_layers = {}
    layer_by_id = {}

    for seg in segments_sorted:
        seg_id = seg["id"]
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_text = seg["text"]

        # Normal layering for regular subtitles
        for lyr_idx, lyr_end in list(active_layers.items()):
            if lyr_end <= seg_start:
                del active_layers[lyr_idx]
        layer_index = 0
        while layer_index in active_layers:
            layer_index += 1
        layer_by_id[seg_id] = layer_index
        active_layers[layer_index] = seg_end

    return layer_by_id


def generate_chapter_title(
    seg_text,
    start_time,
    end_time,
    video_width,
    video_height,
    chapter_font_size=60
):
    """
    Generate an ASS subtitle line for a chapter title.
    Positioned at the top of the screen with enhanced styling.
    """
    y_pos = int(video_height * 0.3)  # 30% from top
    # Remove the formatting markers if they exist
    clean_text = seg_text
    if clean_text.startswith("- ") and clean_text.endswith(" -"):
        clean_text = clean_text[2:-2].strip()  # Remove "- " and " -"
        
    return (
        f"Dialogue: 0,{start_time},{end_time},ChapterStyle,,0,0,0,,"
        f"{{\\an8\\pos({video_width/2},{y_pos})"
        "\\fscx105\\fscy105\\bord2\\shad1\\blur0.5"
        f"\\fad(300,300)}}- {clean_text} -"
    )

def generate_caption_line(
    text,
    start_time,
    end_time,
    x_pos,
    y_pos,
    layer,
    style="WordStyle",
    effects=""
):
    """
    Generate an ASS subtitle line for a regular caption.
    """
    return (
        f"Dialogue: {layer},{start_time},{end_time},"
        f"{style},,0,0,0,,"
        f"{{\\r\\pos({int(x_pos)},{int(y_pos)}){effects}}}{text}"
    )

def generate_ass(
    segments,
    ass_output_path,
    video_width=1280,
    video_height=720,
    font_file="./OpenSansBold.ttf",
    font_size=48,
    max_line_width_ratio=0.9,
    line_spacing=60,
    chapter_font_size=60
):
    """
    Generate an .ass subtitle file with karaoke word-level effects.
    Handles main title, chapter titles, and captions separately.
    """
    # Setup font and measurements
    font_obj = ImageFont.truetype(font_file, size=font_size, encoding="unic")
    highlight_color = "&H00FFFF&"
    normal_color = "&H00F0F0F0&"
    highlight_fontadd = 6
    max_line_width_px = video_width * max_line_width_ratio
    baseline_y = int(video_height * 0.75)  # Base position for regular captions
    segment_layer_spacing = line_spacing  # Spacing between caption layers

    def measure_word_width(text_str):
        left, top, right, bottom = font_obj.getbbox(text_str)
        return right - left

    # Calculate subtitle layers
    layer_by_id = assign_subtitle_layers(segments)

    # Generate ASS header with styles
    internal_font_name = get_ttf_font_name(font_file)

    ass_header = (
        "[Script Info]\n"
        "Title: Karaoke with Crossfade Skips\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {video_width}\n"
        f"PlayResY: {video_height}\n"
        "Collisions: Normal\n"
        "Timer: 100.0000\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding\n"
        f"Style: WordStyle,{internal_font_name},{font_size},&H00F0F0F0,&HFFFFFF,"
        "&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,1,7,0,0,0,1\n"
        f"Style: ChapterStyle,{internal_font_name},{chapter_font_size},&H00FFFFFF,&HFFFFFF,"
        "&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,5,20,20,20,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )

    # Process segments and generate dialogue lines
    dialogue_lines = []

    for seg in segments:
        seg_id = seg["id"]
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_text = seg.get("text", "")
        seg_words = seg.get("words", [])
        this_layer = layer_by_id.get(seg_id, 0)
        segment_base_y = baseline_y - int(this_layer * segment_layer_spacing)
        start_ass = seconds_to_ass_time(seg_start)
        end_ass = seconds_to_ass_time(seg_end)

        # Handle chapter titles first
        if seg_text.startswith("- ") and seg_text.endswith(" -"):
            # Use helper function for chapter titles
            line_str = generate_chapter_title(
                seg_text=seg_text,
                start_time=start_ass,
                end_time=end_ass,
                video_width=video_width,
                video_height=video_height,
                chapter_font_size=chapter_font_size
            )
            dialogue_lines.append(line_str)
            continue

        # For non-chapter text, handle word-level timing
        if not seg_words:
            # Generate approximate word timings if not provided
            words = seg_text.split()
            if len(words) == 1:
                # Single word: display centered at bottom
                x_center = video_width / 2
                y_baseline = int(video_height * 0.90)
                line_str = generate_caption_line(
                    text=seg_text,
                    start_time=start_ass,
                    end_time=end_ass,
                    x_pos=x_center,
                    y_pos=y_baseline,
                    layer=this_layer,
                    effects="\\an5"  # Center alignment for single words
                )
                dialogue_lines.append(line_str)
                continue
            else:
                # Generate timing for multi-word segments
                total_duration = seg_end - seg_start
                word_duration = total_duration / len(words)
                seg_words = []
                current_time = seg_start
                for word in words:
                    seg_words.append({
                        "text": word,
                        "start": current_time,
                        "end": current_time + word_duration,
                    })
                    current_time += word_duration

        # Process words with karaoke timing
        current_line = []
        current_width = 0
        space_px = 6
        lines = []

        def flush_line():
            if current_line:
                lines.append(current_line[:])
                current_line.clear()

        for w in seg_words:
            w_text = w["text"]
            w_width = measure_word_width(w_text)
            w_start = w["start"]
            w_end = w["end"]
            w_start_ms = int(round((w_start - seg_start) * 1000))
            w_end_ms = int(round((w_end - seg_start) * 1000))
            needed = w_width + (space_px if current_line else 0)
            if current_width + needed <= max_line_width_px:
                current_line.append({
                    "text": w_text,
                    "start_ms": w_start_ms,
                    "end_ms": w_end_ms,
                    "width": w_width,
                })
                current_width += needed
            else:
                flush_line()
                current_line.append({
                    "text": w_text,
                    "start_ms": w_start_ms,
                    "end_ms": w_end_ms,
                    "width": w_width,
                })
                current_width = w_width
        if current_line:
            flush_line()

        # Lay out each line.
        for line_idx, line_words in enumerate(lines):
            total_line_width = sum(wd["width"] for wd in line_words) + space_px * (len(line_words) - 1)
            line_x = (video_width - total_line_width) / 2
            line_y = segment_base_y + line_idx * line_spacing
            x_cursor = line_x
            for wd in line_words:
                # Generate karaoke effects for each word
                effects = (
                    f"\\t({wd['start_ms']},{wd['end_ms']},"
                    f"\\c&H{highlight_color[2:-1]}&\\fs{font_size + highlight_fontadd})"
                    f"\\t({wd['end_ms']},{wd['end_ms']+1},"
                    f"\\c&H{normal_color[2:-1]}&\\fs{font_size})"
                )
                # Use caption helper for each word
                line_str = generate_caption_line(
                    text=wd['text'],
                    start_time=start_ass,
                    end_time=end_ass,
                    x_pos=x_cursor,
                    y_pos=line_y,
                    layer=this_layer,
                    effects=effects
                )
                dialogue_lines.append(line_str)
                x_cursor += wd["width"] + space_px

    with open(ass_output_path, "w", encoding="utf-8") as out:
        out.write(ass_header)
        for line in dialogue_lines:
            out.write(line + "\n")


def seconds_to_ass_time(seconds):
    """
    Convert seconds into an ASS time string (H:MM:SS.xx).
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def generate_waveform(audio_path, waveform_path, duration=None, width=1280, height=200, fps=30):
    """
    Generate a waveform video from an audio file using FFmpeg.
    """
    filter_complex = (
        f"[0:a]showwaves=s={width}x{height}:mode=cline:rate={fps}:scale=lin:"
        "colors=#0080FF,format=yuva420p,split=2[wave_main][wave_shadow];"
        "[wave_shadow]colorchannelmixer="
        "rr=1:rg=1:rb=1:"
        "gr=1:gg=1:gb=1:"
        "br=1:bg=1:bb=1:"
        "aa=0.2[shadow_colored];"
        "[wave_main][shadow_colored]overlay=x=5:y=5[final_output]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", audio_path,
        "-filter_complex", filter_complex,
        "-map", "[final_output]",
    ]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd.append(waveform_path)
    run_ffmpeg_command(cmd, "Generating waveform with offset shadow")


def wrap_text(text, max_width=40):
    """
    Word-wrap utility to split text ensuring no line exceeds max_width characters.
    """
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        candidate = " ".join(current_line + [word])
        if len(candidate) <= max_width:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
    return "\n".join(lines)


def overlay_waveform_and_title(background_path, waveform_path, intermediate_video_path,
                               title_text, font_file, video_width=1280, video_height=720,
                               waveform_height=200, duration=None):
    """
    Overlay a waveform on the background and add a centered title.
    """
    wave_box_y = (video_height - waveform_height) // 2
    wrapped_title = wrap_text(title_text, max_width=36)
    line_height = 60
    total_lines = len(wrapped_title.split("\n"))
    text_block_height = total_lines * line_height
    text_start_y = (video_height - text_block_height) // 8

    filter_complex = (
        f"[0:v]scale={video_width}:{video_height}[bg];"
        f"[1:v]colorkey=black:0.1:0.0[wave_nobg];"
        f"[bg][wave_nobg]overlay=x=(W-w)/2:y={wave_box_y}[tmp];"
        f"[tmp]drawtext="
        f"fontfile='{font_file}':"
        f"text='{wrapped_title}':"
        "text_align=center:"
        "fontcolor=white:"
        "fontsize=50:"
        "shadowx=2:"
        "shadowy=2:"
        "box=1:"
        "boxcolor=black@0.3:"
        "boxborderw=10:"
        "line_spacing=5:"
        f"x=(w/2-text_w/2):"
        f"y={text_start_y}"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", background_path,
        "-i", waveform_path,
        "-filter_complex", filter_complex,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
    ]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd.append(intermediate_video_path)
    run_ffmpeg_command(cmd, "Overlaying waveform & title")


def add_karaoke_subtitles(intermediate_video_path, audio_path, ass_path,
                          final_output_path, font_path, audio_bitrate="192k", duration=None):
    """
    Burn .ass subtitles into the video while combining the provided audio.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", intermediate_video_path,
        "-i", audio_path,
        "-shortest",
        "-vf", f"ass={ass_path}:fontsdir={os.path.dirname(os.path.abspath(font_path))}",
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
    ]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd.append(final_output_path)
    run_ffmpeg_command(cmd, "Burning final .ass subtitles with original audio")


def main():
    """
    Main entry point:
    1) Parse CLI args.
    2) Extract audio (or reuse cached).
    3) Load transcript.
    4) Generate highlights and trim audio if --duration is specified.
    5) Generate ASS subtitles.
    6) Generate (or reuse) waveform (unless waveform_height=0).
    7) Overlay waveform (if enabled) and title.
    8) Burn final subtitles.
    """
    args = parse_arguments()
    cache_manager = CacheManager(args.temp_dir)
    
    # Clean up old cache files (older than 7 days)
    cache_manager.cleanup_old_cache(7)

    if not args.video_input and not (args.audio and args.background):
        raise ValueError("Either --video_input OR both --audio and --background must be provided")

    # Use video input: extract audio or reuse cached version
    if args.video_input:
        print("Using video input file for karaoke processing. Extracting audio...")
        cached_audio = cache_manager.get_cached_file("extracted_audio.m4a", args.video_input)
        if cached_audio:
            print("Using cached extracted audio file")
            audio_path = cached_audio
        else:
            extracted_audio_path = os.path.join(args.temp_dir, "extracted_audio.m4a")
            cmd = [
                "ffmpeg",
                "-y",
                "-i", args.video_input,
                "-vn",
                "-acodec", "aac",
                "-b:a", "192k",
                extracted_audio_path
            ]
            run_ffmpeg_command(cmd, "Extracting audio from video input")
            if cache_manager.save_file_hash(args.video_input, "extracted_audio.m4a"):
                audio_path = extracted_audio_path
            else:
                print("Warning: Failed to save cache hash for extracted audio")
                audio_path = extracted_audio_path
        background_path = args.video_input
    else:
        audio_path = args.audio
        background_path = args.background

    ass_path = os.path.join(args.temp_dir, "karaoke.ass")
    waveform_path = os.path.join(args.temp_dir, "waveform.mp4")
    intermediate_video_path = os.path.join(args.temp_dir, "overlayed.mp4")

    with open(args.transcript, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    original_segments = transcript.get("segments", [])
    full_audio_duration = sum(seg["end"] - seg["start"] for seg in original_segments)

    # If duration is specified and less than full duration, generate highlights
    if args.duration is not None and args.duration < full_audio_duration:
        print(f"Generating highlights up to ~{args.duration} seconds.")
        # Use a cached highlights file keyed by duration.
        highlight_cache_path = os.path.join(args.temp_dir, f"highlight_segments_{args.duration}.json")
        if os.path.exists(highlight_cache_path):
            print("Using cached highlight segments")
            with open(highlight_cache_path, 'r', encoding="utf-8") as f:
                highlight_data = json.load(f)
        else:
            highlight_data = get_highlight_segments(
                transcript,
                args.duration,
                crossfade_duration=args.crossfade_duration,
            )
            with open(highlight_cache_path, 'w', encoding="utf-8") as f:
                json.dump(highlight_data, f)

        highlight_hash = hashlib.sha256(json.dumps(highlight_data, sort_keys=True).encode()).hexdigest()[:16]
        trimmed_cache_key = f"trimmed_audio_{highlight_hash}.m4a"
        cached_trimmed = cache_manager.get_cached_file(trimmed_cache_key, audio_path)
        if cached_trimmed:
            print("Using cached trimmed audio file")
            trimmed_audio_path = cached_trimmed
        else:
            trimmed_audio_path = os.path.join(args.temp_dir, trimmed_cache_key)
            trimmed_video_path = os.path.join(args.temp_dir, f"trimmed_video_{highlight_hash}.mp4")
            trim_audio_segments(
                input_path=args.video_input if args.video_input else audio_path,
                highlights_data=highlight_data,
                audio_output_path=trimmed_audio_path,
                video_output_path=trimmed_video_path if args.video_input else None,
                crossfade_duration=args.crossfade_duration,
            )
            if cache_manager.save_file_hash(audio_path, trimmed_cache_key):
                print("Cached trimmed audio file")
            else:
                print("Warning: Failed to save cache hash for trimmed audio")
        final_segments_for_ass = realign_transcript(original_segments, highlight_data, crossfade_duration=args.crossfade_duration)
        audio_path = trimmed_audio_path
        if args.video_input:
            background_path = trimmed_video_path
    else:
        print("Using full audio (no highlight trimming).")
        final_segments_for_ass = original_segments

    print("\n=== Final segments for ASS ===")
    pprint(final_segments_for_ass)

    generate_ass(
        segments=final_segments_for_ass,
        ass_output_path=ass_path,
        video_width=args.video_width,
        video_height=args.video_height,
        font_file=args.font_file,
        font_size=50,
    )

    # Skip waveform generation if height is 0
    if args.waveform_height > 0:
        cached_waveform = cache_manager.get_cached_file("waveform.mp4", audio_path)
        if cached_waveform:
            print("Using cached waveform file")
            waveform_path = cached_waveform
        else:
            generate_waveform(
                audio_path=audio_path,
                waveform_path=waveform_path,
                duration=args.duration,
                width=args.video_width,
                height=args.waveform_height,
                fps=args.waveform_fps,
            )
            if os.path.exists(waveform_path) and os.path.getsize(waveform_path) > 0:
                if cache_manager.save_file_hash(audio_path, "waveform.mp4"):
                    print("Cached waveform file")
                else:
                    print("Warning: Failed to save cache hash for waveform")
            else:
                print("Warning: Waveform generation may have failed")
        
        # Overlay waveform and title
        overlay_waveform_and_title(
            background_path=background_path,
            waveform_path=waveform_path,
            intermediate_video_path=intermediate_video_path,
            title_text=args.title,
            font_file=args.font_file,
            video_width=args.video_width,
            video_height=args.video_height,
            waveform_height=args.waveform_height,
            duration=args.duration,
        )
    else:
        print("\nSkipping waveform generation (height=0)")
        # Just add title without waveform
        filter_complex = (
            f"[0:v]scale={args.video_width}:{args.video_height}[bg];"
            f"[bg]drawtext="
            f"fontfile='{args.font_file}':"
            f"text='{wrap_text(args.title, max_width=36)}':"
            "text_align=center:"
            "fontcolor=white:"
            "fontsize=50:"
            "shadowx=2:"
            "shadowy=2:"
            "box=1:"
            "boxcolor=black@0.3:"
            "boxborderw=10:"
            "line_spacing=5:"
            "x=(w/2-text_w/2):"
            "y=(h/8)"
        )
        
        cmd = [
            "ffmpeg",
            "-y",
            "-i", background_path,
            "-filter_complex", filter_complex,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
        ]
        if args.duration is not None:
            cmd += ["-t", str(args.duration)]
        cmd.append(intermediate_video_path)
        run_ffmpeg_command(cmd, "Adding title without waveform")

    add_karaoke_subtitles(
        intermediate_video_path=intermediate_video_path,
        audio_path=audio_path,
        ass_path=ass_path,
        final_output_path=args.output,
        font_path=args.font_file,
        duration=args.duration,
    )

    print(f"\nDone! Your final karaoke-style video is at: {args.output}")


if __name__ == "__main__":
    main()
