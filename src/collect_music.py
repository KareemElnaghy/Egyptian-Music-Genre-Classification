import argparse
import csv
import concurrent.futures
import json
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults – override via CLI args
# ---------------------------------------------------------------------------
DEFAULT_GENRE = "Mahraganat"
DEFAULT_NUM_CLIPS = 100
DEFAULT_SEGMENT_LENGTH = 30        # seconds
DEFAULT_OUTPUT_DIR = "dataset_new"
DEFAULT_SAMPLE_RATE = 32000        # matches PANNs/CNN14 pretrained checkpoint
DEFAULT_FORMAT = "wav"             # lossless; change to "mp3" if space matters
DEFAULT_JOBS = 3                   # parallel clip exports per song
SEARCH_BATCH_SIZE = 50             # YouTube search results per query batch
MAX_VIDEO_DURATION = 900           # seconds – skip videos longer than 15 minutes

GLOBAL_IDS_FILENAME = "_global_downloaded_ids.json"

# ---------------------------------------------------------------------------
# Genre-specific Arabic + English search queries
# Fall back to generic variants for genres not listed here.
# ---------------------------------------------------------------------------
GENRE_QUERIES = {
    "Tarab": [
        "طرب مصري اصيل",
        "اغاني طرب كلاسيك مصري",
        "Egyptian classical tarab music",
        "ام كلثوم عبد الحليم فيروز",
        "موسيقى عربية اصيلة",
        "classic Egyptian Arabic music",
        "tarab Egyptian songs",
    ],
    "Egyptian Pop": [
        "اغاني بوب مصرية",
        "Egyptian pop music hits",
        "اغاني مصرية شعبية حديثة",
        "egyptian pop songs 2020 2021 2022",
        "اغاني مصرية جديدة pop",
        "اكثر اغاني مصرية انتشارا",
        "Egyptian pop stars music",
        "اغاني مصرية دويتو",
        "اغاني مصرية فيت",
        "collab مصري بوب",
        "اغاني مصرية اوسكار ميوزيك",
        "اغاني مصرية روتانا",
    ],
    "Mahraganat": [
        "مهرجانات مصرية",
        "مهرجانات 2022 2023",
        "Egyptian mahraganat music",
        "مهرجانات شعبية",
        "فيلو حمو بيكا مهرجان",
        "اغاني مهرجانات جديدة",
        "وصلة مهرجانات كاملة",
        "مهرجانات ميكس ساعة",
        "قناة مهرجانات مصرية",
        "مهرجانات بدون موسيقى",
        "توزيع مهرجانات مصري",
    ],
    "Shaabi": [
        "اغاني شعبي مصري",
        "موسيقى شعبي مصري",
        "Egyptian shaabi music",
        "احمد عدوية شعبي",
        "شعبي بلدي مصري",
        "egyptian sha3bi songs",
        "اغاني بلدي مصرية",
        "مزمار بلدي مصري",
        "طبلة بلدي شعبي",
        "فرح شعبي مصري",
        "زفة شعبي مصري",
        "موسيقى زفة مصرية",
        "مزمار الشيخ احمد",
        "شعبي مصري اوريجينال",
    ],
    "Egyptian Rap": [
        "راب مصري",
        "Egyptian Arabic rap",
        "هيب هوب مصري",
        "egyptian rap music",
        "ريمي والجوكر مصري راب",
        "arab Egyptian hip hop",
        "راب عربي مصري جديد",
        "درتي فينيل راب مصري",
        "MTM راب مصري",
        "راب مصري كولكتيف",
        "فري فاير راب مصري",
        "راب مصري ساوند كلاود",
        "راب مصري مستقل",
    ],
}


def ensure_dependencies():
    """Install yt-dlp if it is missing."""
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        print("[setup] Installing yt-dlp ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])


# ---------------------------------------------------------------------------
# Global ID registry – shared across all genre runs
# ---------------------------------------------------------------------------
def load_global_ids(root_dir: Path) -> set[str]:
    """Load the set of video IDs already used in any genre run."""
    path = root_dir / GLOBAL_IDS_FILENAME
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def save_global_ids(root_dir: Path, global_ids: set[str]):
    """Persist the global video ID registry."""
    path = root_dir / GLOBAL_IDS_FILENAME
    path.write_text(json.dumps(list(global_ids), indent=2))


# ---------------------------------------------------------------------------
# Progress tracking – allows resuming a single genre run after a crash
# ---------------------------------------------------------------------------
def load_progress(progress_path: Path) -> dict:
    if progress_path.exists():
        return json.loads(progress_path.read_text())
    return {"clipped_files": [], "song_index": 0, "total_clips": 0}


def save_progress(progress_path: Path, progress: dict):
    progress_path.write_text(json.dumps(progress, indent=2))


def load_used_source_files(csv_path: Path, genre: str) -> set[str]:
    """Read existing metadata and return source files already used for this genre."""
    if not csv_path.exists():
        return set()

    used: set[str] = set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("label") == genre and row.get("source_filename"):
                used.add(row["source_filename"])
    return used


# ---------------------------------------------------------------------------
# Search – gather candidate videos without downloading
# ---------------------------------------------------------------------------
def gather_candidates(genre: str, cookies_file: Path | None) -> list[dict]:
    """
    Search YouTube with multiple query variants and return a de-duplicated
    list of candidate videos (id + title).
    """
    import yt_dlp

    search_variants = GENRE_QUERIES.get(genre, [
        f"{genre} Egyptian music",
        f"{genre} اغاني",
        f"{genre} songs",
        f"{genre} mix",
        f"best {genre}",
    ])

    seen_ids: set[str] = set()
    candidates: list[dict] = []

    for variant in search_variants:
        query = f"ytsearch{SEARCH_BATCH_SIZE}:{variant}"
        print(f"  [search] \"{variant}\"")

        opts = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "ignoreerrors": True,
            "skip_download": True,
        }
        if cookies_file:
            opts["cookiefile"] = str(cookies_file)

        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(query, download=False)

        if not info or "entries" not in info:
            continue

        for entry in info["entries"]:
            if entry is None:
                continue
            vid = entry.get("id", "")
            duration = entry.get("duration") or 0
            if duration > MAX_VIDEO_DURATION:
                continue
            if vid and vid not in seen_ids:
                seen_ids.add(vid)
                candidates.append({"id": vid, "title": entry.get("title", "unknown")})

    print(f"  [search] Found {len(candidates)} unique candidates\n")
    return candidates


def gather_playlist_candidates(playlist_url: str, cookies_file: Path | None) -> list[dict]:
    """Load playlist entries and return de-duplicated video candidates."""
    import yt_dlp

    opts = {
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "extract_flat": True,
    }
    if cookies_file:
        opts["cookiefile"] = str(cookies_file)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)

    if not info or "entries" not in info:
        return []

    seen_ids: set[str] = set()
    candidates: list[dict] = []

    for entry in info["entries"]:
        if entry is None:
            continue
        vid = entry.get("id", "")
        title = entry.get("title", "unknown")
        if vid and vid not in seen_ids:
            seen_ids.add(vid)
            candidates.append({"id": vid, "title": title})

    print(f"  [playlist] Found {len(candidates)} unique playlist entries\n")
    return candidates


# ---------------------------------------------------------------------------
# Artist diversity guard
# ---------------------------------------------------------------------------
def check_artist_diversity(title: str, genre: str, metadata_csv: Path, max_pct: float = 0.10) -> bool:
    """Returns True if this artist is within the per-genre diversity limit."""
    artist = title.split(" - ")[0].split(" | ")[0].strip().lower()
    if not metadata_csv.exists():
        return True
    total = 0
    artist_count = 0
    with open(metadata_csv) as f:
        for row in csv.DictReader(f):
            if row.get("label") == genre:
                total += 1
                src = row.get("source_filename", "").lower()
                if artist in src:
                    artist_count += 1
    if total < 20:  # don't enforce until we have enough data
        return True
    return (artist_count / total) < max_pct


# ---------------------------------------------------------------------------
# Download a single video
# ---------------------------------------------------------------------------
def download_one(video_id: str, title: str, download_dir: Path, sample_rate: int, cookies_file: Path | None) -> Path | None:
    """Download a single YouTube video as WAV. Returns the path or None on failure."""
    import yt_dlp

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(download_dir / "%(title)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "postprocessor_args": ["-ar", str(sample_rate), "-ac", "1"],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "ignoreerrors": True,
        "concurrent_fragment_downloads": 4,
        "retries": 3,
    }
    if cookies_file:
        ydl_opts["cookiefile"] = str(cookies_file)

    # Track files before download so we only return a file created or updated now.
    before_wavs = {p: p.stat().st_mtime for p in download_dir.glob("*.wav")}

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    except Exception as e:
        print(f"    [skip] Failed: {e}")
        return None

    safe_title = yt_dlp.utils.sanitize_filename(title)
    candidate = download_dir / f"{safe_title}.wav"
    after_wavs = list(download_dir.glob("*.wav"))

    # Prefer exact expected file if it was created or updated in this call.
    if candidate.exists():
        candidate_mtime = candidate.stat().st_mtime
        if candidate not in before_wavs or candidate_mtime > before_wavs[candidate]:
            return candidate

    updated_wavs = []
    for wav in after_wavs:
        mtime = wav.stat().st_mtime
        if wav not in before_wavs or mtime > before_wavs[wav]:
            updated_wavs.append(wav)

    if not updated_wavs:
        print("    [skip] No new audio file produced for this video.")
        return None

    return max(updated_wavs, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Clip
# ---------------------------------------------------------------------------
def get_audio_duration(audio_path: Path) -> float:
    """Get duration of an audio file in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(audio_path),
        ],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def clip_audio(
    audio_path: Path,
    genre: str,
    output_dir: Path,
    segment_length: int,
    out_format: str,
    song_index: int,
    jobs: int,
    playlist_mode: bool = False,
    max_clips: int | None = None,
) -> list[dict]:
    """
    Extract 1-2 segments from different parts of the audio file using ffmpeg.
    Returns list of metadata dicts for clips created (empty list if song is too short).
    """
    duration = get_audio_duration(audio_path)

    INTRO_SKIP = 30
    OUTRO_SKIP = 30
    working_start = INTRO_SKIP
    working_end = duration - OUTRO_SKIP
    usable_duration = working_end - working_start

    if usable_duration < segment_length:
        print(f"  [skip] {audio_path.name} too short after trimming intros/outros")
        return []

    num_clips = 2 if usable_duration >= segment_length * 2 else 1

    if num_clips == 1:
        start_times = [working_start + usable_duration * 0.50]
    else:
        start_times = [
            working_start + usable_duration * 0.33,
            working_start + usable_duration * 0.67,
        ]

    if max_clips is not None:
        if max_clips <= 0:
            return []
        start_times = start_times[:max_clips]
        num_clips = len(start_times)

    def export_segment(seg_idx_start: tuple[int, float]) -> dict | None:
        seg_idx, start = seg_idx_start
        filename = f"{genre}_{song_index:03d}_seg{seg_idx:02d}.{out_format}"
        out_path = output_dir / filename

        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(audio_path),
                "-ss", str(start),
                "-t", str(segment_length),
                "-acodec", "pcm_s16le" if out_format == "wav" else "libmp3lame",
                str(out_path),
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            return None

        rms_result = subprocess.run([
            "ffprobe", "-v", "quiet", "-f", "lavfi",
            "-i", f"amovie={out_path},astats=metadata=1:reset=1",
            "-show_entries", "frame_tags=lavfi.astats.Overall.RMS_level",
            "-of", "csv=p=0"
        ], capture_output=True, text=True)
        try:
            rms_db = float(rms_result.stdout.strip().split('\n')[0])
            if rms_db < -50:
                out_path.unlink(missing_ok=True)
                print(f"    [skip] Clip {filename} rejected: silence (RMS {rms_db:.1f}dB)")
                return None
        except (ValueError, IndexError):
            pass  # if ffprobe fails the check, accept the clip

        return {
            "clip_filename": filename,
            "label": genre,
            "source_filename": audio_path.name,
            "segment_index": seg_idx,
            "start_time": start,
            "end_time": start + segment_length,
        }

    clips_meta = []
    seg_jobs = min(max(1, jobs), len(start_times))
    with concurrent.futures.ThreadPoolExecutor(max_workers=seg_jobs) as executor:
        for segment_meta in executor.map(export_segment, enumerate(start_times)):
            if segment_meta is not None:
                clips_meta.append(segment_meta)

    print(f"  [clip] {audio_path.name} -> {len(clips_meta)} clips from different positions")
    return clips_meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Collect & clip music by genre")
    parser.add_argument("--genre", type=str, default=DEFAULT_GENRE, help="Genre search term")
    parser.add_argument("--num_clips", type=int, default=DEFAULT_NUM_CLIPS, help="Target number of clips to collect")
    parser.add_argument("--segment_length", type=int, default=DEFAULT_SEGMENT_LENGTH, help="Clip length in seconds")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Root output directory")
    parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Audio sample rate (Hz)")
    parser.add_argument("--format", type=str, default=DEFAULT_FORMAT, choices=["wav", "mp3"], help="Output audio format")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS, help="Parallel clip-export workers per song")
    parser.add_argument("--playlist_url", type=str, default="", help="Optional YouTube playlist URL. If set, skips search and clips each playlist song into 1-2 clips (capped by --num_clips).")
    parser.add_argument("--cookies_file", type=str, default=None, help="Path to a Netscape format cookies file for yt-dlp authentication.")

    # Keep notebook compatibility while still honoring real CLI args in terminal.
    if "ipykernel" in sys.modules:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    ensure_dependencies()

    root_dir = Path(args.output_dir)
    genre_dir = root_dir / args.genre
    raw_dir = root_dir / "_raw" / args.genre

    genre_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Load the global registry (shared across all genre runs)
    global_ids = load_global_ids(root_dir)
    print(f"[dedup] Global registry loaded: {len(global_ids)} video IDs already used across all genres.\n")

    # Load per-genre resume progress (no longer stores downloaded IDs)
    progress_path = raw_dir / ".progress.json"
    progress = load_progress(progress_path)
    metadata_csv_path = root_dir / "metadata.csv"
    used_source_files = load_used_source_files(metadata_csv_path, args.genre)

    total_clips = progress["total_clips"]
    playlist_mode = bool(args.playlist_url)
    clips_needed = args.num_clips - total_clips

    if clips_needed <= 0:
        print(f"[done] Already have {total_clips} clips (target: {args.num_clips}).")
        return

    # --- Step 1: Initial search for candidates ------------------------------
    print(f"\n{'='*60}")
    if playlist_mode:
        print("  Loading playlist entries ...")
    else:
        print(f"  Searching for '{args.genre}' songs  (need {clips_needed} clips) ...")
    print(f"{'='*60}\n")

    # Pass cookies_file to gather functions
    cookies_path = Path(args.cookies_file) if args.cookies_file else None
    if playlist_mode:
        candidates = gather_playlist_candidates(args.playlist_url, cookies_path)
    else:
        candidates = gather_candidates(args.genre, cookies_path)

    if not candidates:
        if playlist_mode:
            print("[error] No playlist entries found. Check URL or provide cookies for authentication.")
        else:
            print("[error] No candidates found. Try a different search term or provide cookies for authentication.")
        sys.exit(1)

    # --- Step 2: Download → clip → repeat until target met ------------------
    print(f"{'='*60}")
    if playlist_mode:
        print(f"  Downloading & clipping playlist  (entries: {len(candidates)}) ...")
    else:
        print(f"  Downloading & clipping  (target: {args.num_clips} clips) ...")
    print(f"{'='*60}\n")

    already_clipped = set(progress["clipped_files"])
    song_index = progress["song_index"]
    all_meta: list[dict] = []
    songs_used = 0
    candidate_idx = 0

    while True:
        if total_clips >= args.num_clips:
            break

        # If we've exhausted current candidates, search for more
        if candidate_idx >= len(candidates):
            if playlist_mode:
                break

            print(f"\n  [search] Need more candidates (current: {len(candidates)}, clips: {total_clips}/{args.num_clips})")
            print("  [search] Expanding search...\n")
            new_candidates = gather_candidates(args.genre, cookies_path)
            # Filter out any video already used in any genre
            new_unique = [c for c in new_candidates if c["id"] not in global_ids]
            if not new_unique:
                print("  [warn] No new unique candidates found. Stopping.")
                break
            candidates.extend(new_unique)
            print(f"  [search] Added {len(new_unique)} new unique candidates\n")

        if candidate_idx >= len(candidates):
            break

        candidate = candidates[candidate_idx]
        candidate_idx += 1

        vid = candidate["id"]
        title = candidate["title"]

        # Skip if this video was used in any genre (global check)
        if vid in global_ids:
            print(f"  [skip] Already used in another genre: {title}")
            continue

        # Artist diversity check — keep any single artist below max_pct of the genre
        if not check_artist_diversity(title, args.genre, metadata_csv_path):
            print(f"  [skip] Artist diversity limit reached for: {title}")
            continue

        print(f"  [{total_clips}/{args.num_clips} clips] Downloading: {title}")

        # Pass cookies_file to download_one
        wav_path = download_one(vid, title, raw_dir, args.sample_rate, cookies_path)

        if wav_path is None:
            continue

        if wav_path.name in used_source_files:
            print(f"  [skip] Duplicate source for {args.genre}: {wav_path.name}")
            global_ids.add(vid)
            save_global_ids(root_dir, global_ids)
            continue

        clips = clip_audio(
            wav_path, args.genre, genre_dir,
            30 if playlist_mode else args.segment_length,
            args.format,
            song_index,
            args.jobs,
            playlist_mode=playlist_mode,
            max_clips=(args.num_clips - total_clips),
        )

        if clips:
            all_meta.extend(clips)
            total_clips += len(clips)
            songs_used += 1
            used_source_files.add(wav_path.name)
            global_ids.add(vid)
            save_global_ids(root_dir, global_ids)
        else:
            # Still mark this ID as processed to avoid retry loops.
            global_ids.add(vid)
            save_global_ids(root_dir, global_ids)

        if wav_path and wav_path.exists():
            wav_path.unlink()

        already_clipped.add(wav_path.name)
        song_index += 1
        progress["clipped_files"] = list(already_clipped)
        progress["song_index"] = song_index
        progress["total_clips"] = total_clips
        save_progress(progress_path, progress)

    # --- Step 3: Write metadata CSV -----------------------------------------
    csv_path = metadata_csv_path
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["clip_filename", "label", "source_filename", "segment_index", "start_time", "end_time"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(all_meta)

    # --- Step 4: Cleanup raw downloads --------------------------------------
    shutil.rmtree(raw_dir, ignore_errors=True)

    # --- Summary ------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"  Songs downloaded : {songs_used}")
    print(f"  Clips created    : {len(all_meta)}")
    print(f"  Total clips      : {total_clips}")
    print(f"  Segments saved   : {genre_dir}/")
    print(f"  Metadata CSV     : {csv_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
