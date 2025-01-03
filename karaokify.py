#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import re
import subprocess
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
            "highlight reels, crossfade audio segments, overlay a waveform, "
            "and burn in subtitles for a final MP4."
        ),
        epilog=(
            "Example:\n"
            "  python karaokify.py --audio=my_podcast.mp3 --transcript=transcript.json \\\n"
            "       --background=background.mp4 --output=final_video.mp4 --title=\"My Podcast\"\n\n"
            "For a short highlight reel (90 seconds) with crossfade:\n"
            "  python karaokify.py --audio=my_podcast.mp3 --transcript=transcript.json \\\n"
            "       --background=background.mp4 --duration=90 --crossfade_duration=1.5\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to the audio file (e.g., .mp3 or .wav)."
    )
    parser.add_argument(
        "--transcript",
        required=True,
        help="Path to the JSON transcript file."
    )
    parser.add_argument(
        "--background",
        required=True,
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
        default="./OpenSansBold.ttf",
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
            "Max duration of highlight reel (for testing). "
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

    1) Strips obvious short disfluencies ("um", "uh", "er", "ah", "hmm").
    2) Removes the placeholder token "[*]".
    3) Excludes entire segments whose cleaned text is < min_length characters.

    :param transcript: A dict with "segments" key, each segment has {id, start, end, text}.
    :param min_length: Minimum length of cleaned text for a segment to be kept.
    :return: A new dict with "segments" filtered and cleaned.
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
    Extract only segment-level data (id, start, end, text) from the transcript,
    then filter filler segments.
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


def get_highlight_segments(transcript, max_duration, crossfade_duration=0):
    """
    Use a language model to pick highlight segments up to `max_duration`.
    Each highlight segment is padded by crossfade_duration seconds on both ends.
    """
    reduced_transcript = extract_segments_transcript_data(transcript)

    prompt = (
        f"Below I provided the transcript of a long audio/video.\n"
        "Your task is produce a list of \"highlight segments\"; the segments that "
        "best describe all the important, interesting, insightful, noteworthy, or "
        "discussion-worthy parts.\n\n"
        "Cover the entire transcript content, and be diligent and accurate.\n"
        "Give each highlight segment a title that best describes in up to 4 words "
        "the context and meaning of what is discussed in this segment text.\n"
        "Make sure that the last Highlight segment ends at the best thought-provoking statement, making the user want more.\n"
        f"Each highlight segment must be at least 2 seconds long, and always pad it "
        f"with {crossfade_duration} seconds to account for crossfades.\n"
        f"The max_duration is: {max_duration}.\n\n"
        "Return your answer as JSON:\n"
        "{\n"
        "  \"highlights\": [\n"
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
        return json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, AttributeError):
        print("Error parsing LLM response. Falling back to entire transcript.")
        return {"highlights": reduced_transcript["segments"]}


def trim_audio_segments(audio_path, highlights_data, output_path, crossfade_duration=1.0):
    """
    1) Sort highlights.
    2) Extract each highlight clip from the audio.
    3) Crossfade them all together.
    => Output to output_path.
    """
    highlights = highlights_data.get("highlights", [])
    highlights.sort(key=lambda x: x["start"])

    if not highlights:
        print("No highlights found, skipping trim_audio_segments.")
        return

    clip_files = []
    tmp_dir = os.path.dirname(output_path)

    for i, seg in enumerate(highlights):
        start = seg["start"]
        end = seg["end"]
        duration = end - start
        clip_filename = os.path.join(tmp_dir, f"clip_{i}.m4a")

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-i",
            audio_path,
            "-acodec",
            "aac",
            "-b:a",
            "192k",
            clip_filename,
        ]
        run_ffmpeg_command(cmd, f"Extracting highlight clip {i}")
        clip_files.append(clip_filename)

    if len(clip_files) == 1:
        os.rename(clip_files[0], output_path)
        return

    num_clips = len(clip_files)
    filter_lines = []

    for i in range(num_clips - 1):
        left_label = f"[{i}:a]" if i == 0 else f"[a{i}]"
        right_label = f"[{i+1}:a]"
        out_label = f"[a{i+1}]"
        line = (
            f"{left_label}{right_label}acrossfade="
            f"d={crossfade_duration}:curve1=tri:curve2=tri"
            f"{out_label}"
        )
        filter_lines.append(line)

    final_label = f"[a{num_clips-1}]"
    filtergraph = ";".join(filter_lines)

    crossfade_cmd = ["ffmpeg", "-y"]
    for cf in clip_files:
        crossfade_cmd += ["-i", cf]
    crossfade_cmd += [
        "-filter_complex",
        filtergraph,
        "-map",
        final_label,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        output_path,
    ]

    run_ffmpeg_command(crossfade_cmd, "Crossfading highlight clips")


def chunk_into_subsegments(
    shifted_words,
    overall_start,
    overall_end,
    segment_id_start,
    max_chars=40,
    max_lines=2,
):
    """
    Break the entire set of 'shifted_words' for a highlight into multiple
    smaller subtitle segments, each with up to 2 lines, each line up to
    ~max_chars characters. Return a list of new segments, each with
    "id", "start", "end", "text", "words".
    """

    def line_len(words_in_line):
        """
        Approximate the length of a single line by summing word lengths + spaces.
        """
        if not words_in_line:
            return 0
        return sum(len(w["text"]) for w in words_in_line) + (len(words_in_line) - 1)

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
            # line is "full"; try to begin a new line
            if len(current_lines) < max_lines:
                begin_line()
                current_lines[-1].append(w)
            else:
                # we've hit max_lines for this sub-segment => finalize and start new
                seg_words = [wd for line_wds in current_lines for wd in line_wds]
                if seg_words:
                    sub_start = max(min(x["start"] for x in seg_words), overall_start)
                    sub_end = min(max(x["end"] for x in seg_words), overall_end)
                    sub_text = " ".join(x["text"] for x in seg_words)

                    subsegments.append(
                        {
                            "id": current_id,
                            "start": sub_start,
                            "end": sub_end,
                            "text": sub_text,
                            "words": seg_words,
                        }
                    )
                    current_id += 1

                # start new sub-segment
                current_lines = []
                begin_line()
                current_lines[-1].append(w)

    # Flush any leftover lines
    seg_words = [wd for line_wds in current_lines for wd in line_wds]
    if seg_words:
        sub_start = max(min(x["start"] for x in seg_words), overall_start)
        sub_end = min(max(x["end"] for x in seg_words), overall_end)
        if sub_end > sub_start:
            sub_text = " ".join(x["text"] for x in seg_words)
            subsegments.append(
                {
                    "id": current_id,
                    "start": sub_start,
                    "end": sub_end,
                    "text": sub_text,
                    "words": seg_words,
                }
            )
            current_id += 1

    return subsegments, current_id


def realign_transcript(original_segments, highlight_data, crossfade_duration=1.0):
    """
    Re-map word-level times to match the new spliced audio timeline after
    extracting + concatenating highlight segments with crossfade.

    Then chunk each highlight's words into smaller segments:
    - up to 2 lines per sub-segment, each ~35-40 chars.
    - skip any words == "[*]"
    - add a brief "---- Highlight #N ----" sub-segment at the start of each highlight
    - skip subtitles during crossfade overlap by reducing the highlight's end time
      by `crossfade_duration` (except the last highlight).
    """
    highlights = highlight_data.get("highlights", [])
    if not highlights:
        return []

    # Compute offsets in the final timeline
    highlight_offsets = []
    for i, h in enumerate(highlights):
        if i == 0:
            highlight_offsets.append(0.0)
        else:
            prev = highlights[i - 1]
            prev_len = prev["end"] - prev["start"]
            highlight_offsets.append(
                highlight_offsets[i - 1] + prev_len - crossfade_duration
            )

    all_segments = []
    id_counter = 1

    for i, highlight in enumerate(highlights):
        orig_start = highlight["start"]
        orig_end = highlight["end"]
        seg_dur = orig_end - orig_start

        offset = highlight_offsets[i]
        if i < len(highlights) - 1:
            local_end = offset + seg_dur
        else:
            local_end = offset + seg_dur - crossfade_duration

        matched_words = []
        for seg in original_segments:
            if seg["end"] <= orig_start or seg["start"] >= orig_end:
                continue
            for w in seg.get("words", []):
                if w["end"] <= orig_start or w["start"] >= orig_end:
                    continue
                if w["text"].strip() == "[*]":
                    continue

                w_start = max(w["start"], orig_start)
                w_end = min(w["end"], orig_end)
                matched_words.append(
                    {
                        "text": w["text"],
                        "start": round(offset + (w_start - orig_start), 3),
                        "end": round(offset + (w_end - orig_start), 3),
                    }
                )

        matched_words.sort(key=lambda x: x["start"])
        clipped_words = [w for w in matched_words if w["end"] <= local_end]

        notice_id = id_counter
        id_counter += 1
        title_text = highlight.get("segment_title", f"Highlight #{i+1}")
        highlight_notice_seg = {
            "id": notice_id,
            "start": round(offset, 2),
            "end": round(min(offset + 3, local_end - 2), 2),
            "text": f"- {title_text} -",
            "words": [],
        }
        all_segments.append(highlight_notice_seg)

        subsegments, id_counter = chunk_into_subsegments(
            clipped_words,
            overall_start=offset,
            overall_end=local_end,
            segment_id_start=id_counter,
            max_chars=40,
            max_lines=2,
        )
        all_segments.extend(subsegments)

    return all_segments


def get_ttf_font_name(ttf_path):
    """
    Read the font's 'name' table to extract a usable internal name.
    Typically, nameID=4 (Full Name) or nameID=1 (Family Name).
    We'll try nameID=4 first, then fallback to nameID=1.
    """
    font = TTFont(ttf_path)
    name_record = None

    # Try Full Name (nameID = 4)
    for record in font["name"].names:
        if record.nameID == 4 and record.platformID in (1, 3):
            name_str = record.toUnicode()
            if name_str:
                name_record = name_str
                break

    # Fallback to Family Name (nameID = 1) if needed
    if not name_record:
        for record in font["name"].names:
            if record.nameID == 1 and record.platformID in (1, 3):
                name_str = record.toUnicode()
                if name_str:
                    name_record = name_str
                    break

    if not name_record:
        name_record = "UnknownFont"

    return name_record


def assign_subtitle_layers(segments):
    """
    Given a list of segments with .start and .end, assign each segment to a
    'layer' to avoid on-screen overlap.

    Returns a dict: layer_by_id = { segment_id: integer_layer }
    """
    segments_sorted = sorted(segments, key=lambda s: s["start"])
    active_layers = {}
    layer_by_id = {}

    for seg in segments_sorted:
        seg_id = seg["id"]
        seg_start = seg["start"]
        seg_end = seg["end"]

        # Clear out layers whose end_time <= this seg's start
        for lyr_idx, lyr_end in list(active_layers.items()):
            if lyr_end <= seg_start:
                del active_layers[lyr_idx]

        # Find the lowest free layer index
        layer_index = 0
        while layer_index in active_layers:
            layer_index += 1

        # Assign this segment to that layer
        layer_by_id[seg_id] = layer_index
        active_layers[layer_index] = seg_end

    return layer_by_id


def generate_ass(
    segments,
    ass_output_path,
    video_width=1280,
    video_height=720,
    font_file="./OpenSansBold.ttf",
    font_size=48,
    max_line_width_ratio=0.9,
    line_spacing=60,
):
    """
    Generate an .ass subtitle file. Each segment can have word-level karaoke:
    - Each word "lights up" (highlights) between its start/end times.
    - If two segments overlap, shift one upward so they don't collide on-screen.
    """
    layer_by_id = assign_subtitle_layers(segments)

    segment_layer_spacing = 1.3 * line_spacing

    dummy_im = Image.new("RGB", (video_width, video_height), (0, 0, 0))
    draw = ImageDraw.Draw(dummy_im)
    font_obj = ImageFont.truetype(font_file, size=font_size, encoding="unic")

    highlight_color = "&H00FFFF&"
    normal_color = "&H00F0F0F0&"
    highlight_fontadd = 6

    def measure_word_width(text_str):
        left, top, right, bottom = font_obj.getbbox(text_str)
        return right - left

    def seconds_to_ass_time(sec):
        hh = int(sec // 3600)
        mm = int((sec % 3600) // 60)
        ss = sec % 60
        return f"{hh}:{mm:02d}:{ss:05.2f}"

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
        "&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,1,7,0,0,0,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )

    dialogue_lines = []
    max_line_width_px = video_width * max_line_width_ratio
    baseline_y = int(video_height * 0.80)

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

        # If no word-level data, just place the entire text in one line
        if not seg_words:
            left, top, right, bottom = font_obj.getbbox(seg_text)
            text_width = right - left
            x_center = video_width / 2
            y_baseline = int(video_height * 0.30)
            piece = f"{{\\r\\an5\\pos({x_center},{y_baseline})}}{seg_text}"
            line_str = (
                f"Dialogue: 0,{start_ass},{end_ass},WordStyle,,0,0,0,,{piece}"
            )
            dialogue_lines.append(line_str)
            continue

        # Karaoke approach with word-level times
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
                current_line.append(
                    {
                        "text": w_text,
                        "start_ms": w_start_ms,
                        "end_ms": w_end_ms,
                        "width": w_width,
                    }
                )
                current_width += needed
            else:
                flush_line()
                current_line.append(
                    {
                        "text": w_text,
                        "start_ms": w_start_ms,
                        "end_ms": w_end_ms,
                        "width": w_width,
                    }
                )
                current_width = w_width

        if current_line:
            flush_line()

        # Lay out each line
        for line_idx, line_words in enumerate(lines):
            total_line_width = sum(wd["width"] for wd in line_words)
            total_line_width += space_px * (len(line_words) - 1)
            line_x = (video_width - total_line_width) / 2
            line_y = segment_base_y + line_idx * line_spacing
            x_cursor = line_x

            for wd in line_words:
                piece = (
                    f"{{\\r\\pos({int(x_cursor)},{int(line_y)})"
                    f"\\t({wd['start_ms']},{wd['end_ms']},"
                    f"\\c&H{highlight_color[2:-1]}&\\fs{font_size + highlight_fontadd})"
                    f"\\t({wd['end_ms']},{wd['end_ms']+1},"
                    f"\\c&H{normal_color[2:-1]}&\\fs{font_size})}}{wd['text']}"
                )
                line_str = (
                    f"Dialogue: 0,{start_ass},{end_ass},"
                    f"WordStyle,,0,0,0,,{piece}"
                )
                dialogue_lines.append(line_str)
                x_cursor += wd["width"] + space_px

    with open(ass_output_path, "w", encoding="utf-8") as out:
        out.write(ass_header)
        for line in dialogue_lines:
            out.write(line + "\n")


def seconds_to_ass_time(seconds):
    """
    Convert a float number of seconds into an ASS time string (H:MM:SS.xx).
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def generate_waveform(
    audio_path,
    waveform_path,
    duration=None,
    width=1280,
    height=200,
    fps=30,
):
    """
    Generate a waveform video from an audio file using FFmpeg.
    If duration is provided, limit the output.
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
        "-i",
        audio_path,
        "-filter_complex",
        filter_complex,
        "-map",
        "[final_output]",
    ]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd.append(waveform_path)
    run_ffmpeg_command(cmd, "Generating waveform with offset shadow")


def wrap_text(text, max_width=40):
    """
    A simple word-wrap utility that splits text by words, and ensures
    no line exceeds max_width characters.
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


def overlay_waveform_and_title(
    background_path,
    waveform_path,
    intermediate_video_path,
    title_text,
    font_file,
    video_width=1280,
    video_height=720,
    waveform_height=200,
    duration=None,
):
    """
    Overlay a waveform on the background video/image and place a centered
    title at the top. Output is written to intermediate_video_path.
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
        "-i",
        background_path,
        "-i",
        waveform_path,
        "-filter_complex",
        filter_complex,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
    ]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd.append(intermediate_video_path)
    run_ffmpeg_command(
        cmd,
        "Overlaying keyed-out waveform & title (centered, top-down)"
    )


def add_karaoke_subtitles(
    intermediate_video_path,
    audio_path,
    ass_path,
    final_output_path,
    font_path,
    audio_bitrate="192k",
    duration=None,
):
    """
    Combine the intermediate video, original (or trimmed) audio,
    and ASS subtitles into the final karaoke-style MP4 video.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        intermediate_video_path,
        "-i",
        audio_path,
        "-shortest",
        "-vf",
        f"ass={ass_path}:fontsdir={os.path.dirname(os.path.abspath(font_path))}",
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
    ]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd.append(final_output_path)
    run_ffmpeg_command(cmd, "Burning final .ass subtitles with original audio")


def main():
    """
    Main entry point:
    1) Parse CLI args.
    2) Possibly trim highlights from audio.
    3) Generate .ass subtitles.
    4) Generate waveform video.
    5) Overlay background + waveform.
    6) Burn final subtitles.
    """
    args = parse_arguments()
    os.makedirs(args.temp_dir, exist_ok=True)

    ass_path = os.path.join(args.temp_dir, "karaoke.ass")
    waveform_path = os.path.join(args.temp_dir, "waveform.mp4")
    intermediate_video_path = os.path.join(args.temp_dir, "overlayed.mp4")

    # 1) Load transcript
    with open(args.transcript, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    original_segments = transcript.get("segments", [])
    full_audio_duration = sum(seg["end"] - seg["start"] for seg in original_segments)
    audio_path = args.audio
    final_segments_for_ass = original_segments  # default to entire transcript

    # 2) If user requests a highlight reel shorter than total
    if args.duration and args.duration < full_audio_duration:
        print(f"Generating highlights up to ~{args.duration} seconds.")
        highlight_data = get_highlight_segments(
            transcript,
            args.duration,
            crossfade_duration=args.crossfade_duration,
        )

        trimmed_audio_path = os.path.join(args.temp_dir, "trimmed_audio.m4a")
        trim_audio_segments(
            audio_path=args.audio,
            highlights_data=highlight_data,
            output_path=trimmed_audio_path,
            crossfade_duration=args.crossfade_duration,
        )

        final_segments_for_ass = realign_transcript(
            original_segments,
            highlight_data,
            crossfade_duration=args.crossfade_duration,
        )

        audio_path = trimmed_audio_path
    else:
        print("Using full audio (no highlight trimming).")

    print("\n=== Final segments for ASS ===")
    pprint(final_segments_for_ass)

    # 3) Generate .ass
    generate_ass(
        segments=final_segments_for_ass,
        ass_output_path=ass_path,
        video_width=args.video_width,
        video_height=args.video_height,
        font_file=args.font_file,
        font_size=50,
    )

    # 4) Waveform
    generate_waveform(
        audio_path=audio_path,
        waveform_path=waveform_path,
        width=args.video_width,
        height=args.waveform_height,
        fps=args.waveform_fps,
    )

    # 5) Overlay background + waveform
    overlay_waveform_and_title(
        background_path=args.background,
        waveform_path=waveform_path,
        intermediate_video_path=intermediate_video_path,
        title_text=args.title,
        font_file=args.font_file,
        video_width=args.video_width,
        video_height=args.video_height,
        waveform_height=args.waveform_height,
    )

    # 6) Burn in subtitles
    add_karaoke_subtitles(
        intermediate_video_path=intermediate_video_path,
        audio_path=audio_path,
        ass_path=ass_path,
        final_output_path=args.output,
        font_path=args.font_file,
    )

    print(f"\nDone! Your final karaoke-style video is at: {args.output}")


if __name__ == "__main__":
    main()
