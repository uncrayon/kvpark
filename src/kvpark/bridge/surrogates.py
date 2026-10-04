"""Text stand-ins ("surrogates") for media a target model cannot take: images, audio and video.

A conversation can hold media of a kind the target's server has no encoder for: an image moving
from Qwen3.8 (vision) to Muse (text only), a voice note or a video moving to a model without
audio or video input. llama-server refuses such a request outright (HTTP 500 "image input is not
supported", tools/server/server-common.cpp oaicompat_chat_params_parse), and the media's cache
cannot be translated either (every model has its own encoder). So the bridge replaces each such
part with text before it renders the target's prompt (apply), and the Hermes middleware rewrites
the real request the same way (rewrite_request), so the target processes exactly the prompt that
was translated for it:

    image  ->  [Image description: ...]
    audio  ->  [Audio transcript: ...]          the speech verbatim, or "No speech." and the sounds
    video  ->  [Video description: ...]         what happens, in order: from a producer with video
               [Video audio transcript: ...]    input, or from keyframes a vision producer reads;
                                                its sound track when it can be transcribed
    video, for a target with vision but no video input: its keyframes as images (when ffmpeg is
           installed), after "[Video keyframes at ...]"

A message left without media becomes one string, its text parts joined by newlines. A video's
sound track that no producer could transcribe is noted as such ("[Video audio transcript:
unavailable ...]", report "partial"); a later resolve with a producer for audio completes it.

Each surrogate comes from the first of these that has it:

1. the conversation: a text part right after the media part that already starts with the
   modality's marker (a client that sends both); the media part is dropped;
2. the surrogate cache (Cache): one made before for the same media bytes (sha256), kind and
   prompt version, by any producer, so later switches and later turns get the same text and the
   target's prompt keeps matching its cache;
3. the switch's source model, when its server has the modality: a side request (temperature 0,
   thinking off). Its slot is saved before and restored after (switch.py reuses that save as
   the engine's input), so its cached conversation is unchanged;
4. the describer configured for the modality (kvpark bridge media --describer): a registered
   bridge model, whose slot is saved and restored the same way, a local llama-server URL, or for
   audio a local OpenAI-compatible speech-to-text endpoint (.../v1/audio/transcriptions, e.g.
   Lemonade's Whisper or a faster-whisper server; Transcriber);
5. none: the switch is skipped (target-no-vision, target-no-audio, target-no-video), as it is
   for a kind of media whose policy is "skip" (kvpark bridge media --image skip, ...).

The cache is a private directory beside the bridge configuration (surrogates/, entries 0600)
trimmed to a size cap; it holds surrogate texts and a video's keyframes, keyed by the media's
digests, never the media. Standard library only; ffmpeg and ffprobe are used when installed,
for a video's keyframes and sound track.
"""

import base64
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

from . import messages, registry
from .client import LlamaServer, ServerError

PROMPT_VERSION = 1                 # part of every cache key: a new prompt makes new surrogates
MARKERS = {"vision": "Image description", "audio": "Audio transcript", "video": "Video description"}
VIDEO_AUDIO_MARKER = "Video audio transcript"
VIDEO_FRAMES_MARKER = "Video keyframes at"
# in place of the transcript of a video's sound track no producer could transcribe
SOUND_UNTRANSCRIBED = "unavailable (the video has a sound track that could not be transcribed)"
PROMPTS = {
    "vision": ("Describe this image for a reader who cannot see it and will be asked about it later. Transcribe "
               "all visible text exactly. Give the important details: objects, people, counts, colors, "
               "positions, charts with their values, and the layout. Be factual; do not guess beyond what is "
               "visible. Answer with the description only."),
    "audio": ("Transcribe the speech in this audio verbatim. If there is no speech, write \"No speech.\" and "
              "describe the sounds in one sentence; if there is speech and also notable other sound (music, "
              "noise), add one short sentence about it in parentheses at the end. Answer with the transcript "
              "only."),
    "video": ("Describe this video for a reader who cannot see it and will be asked about it later: what happens, "
              "in order, with approximate times; transcribe all visible text exactly; mention people, objects, "
              "counts and colors. Answer with the description only."),
    "keyframes": ("These images are frames of one video, each after its time. Describe the video for a reader "
                  "who cannot see it and will be asked about it later: what each frame shows and what changes "
                  "between them; transcribe all visible text exactly. Answer with the description only."),
}
MAX_TOKENS = {"vision": 768, "audio": 4096, "video": 1024, "keyframes": 1024}
REQUEST_TIMEOUT = 600
TRANSCRIPTION_MODEL = "whisper-1"  # the "model" field of a transcription request, unless set
AUDIO_TYPES = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac", "ogg": "audio/ogg", "m4a": "audio/mp4"}
CACHE_DIR = "surrogates"
CACHE_MAX_BYTES = 64 << 20
FRAMES = "video-frames"            # the cache kind of a video's keyframes (for targets with vision)
MAX_KEYFRAMES = 8
KEYFRAME_STEP_S = 2.0              # one keyframe per this many seconds of video, at most MAX_KEYFRAMES
KEYFRAME_WIDTH = 512               # keyframes are scaled down to this width (projectors resize anyway)
KEYFRAME_QUALITY = 3               # their JPEG quality (ffmpeg -q:v: 2 best, 31 worst)
FFMPEG_TIMEOUT = 120


def policy(config, model=None):
    """{"image": ..., "audio": ..., "video": ...}: what the bridge does with each kind of media a
    target cannot take: the defaults (caption, transcribe, describe), `kvpark bridge media`'s
    settings, then those for `model` as a target."""
    settings = config.get("media") or {}
    out = {key: values[0] for key, values in registry.MEDIA_POLICIES.values()}
    out.update({key: value for key, value in (settings.get("policy") or {}).items() if key in out})
    if model is not None:
        overrides = (settings.get("models") or {}).get(model) or {}
        out.update({key: value for key, value in overrides.items() if key in out})
    return out


def converts(config, model, modality):
    """Whether media of this modality become surrogates for `model` as a target (policy not "skip")."""
    return policy(config, model)[registry.MEDIA_POLICIES[modality][0]] != "skip"


def wrap(marker, text):
    return f"[{marker}: {text}]"


# -- the cache ------------------------------------------------------------------------------------

class Cache:
    """Surrogates by (media sha256, kind, prompt version, producer) in a private directory,
    trimmed to `max_bytes` (least recently used first). get() returns the oldest entry made for
    the media by any producer: every later switch and turn reuses the first surrogate made (a
    complete one before one whose video sound track is missing, "sound_missing")."""

    def __init__(self, directory, max_bytes=CACHE_MAX_BYTES):
        self.directory, self.max_bytes = Path(directory), int(max_bytes)

    def _stem(self, digest, kind):
        return f"{digest}-{kind}-v{PROMPT_VERSION}"

    def get(self, digest, kind):
        found = []
        for path in self.directory.glob(self._stem(digest, kind) + "-*.json"):
            try:
                found.append((json.loads(path.read_text(encoding="utf-8")), path))
            except (OSError, ValueError):
                continue
        if not found:
            return None
        entry, path = min(found, key=lambda item: (bool(item[0].get("sound_missing")), item[0].get("created", 0),
                                                   item[1].name))
        with contextlib.suppress(OSError):
            os.utime(path)             # recently used: trimmed last
        return entry

    def put(self, digest, kind, producer, entry):
        """Store a surrogate made by `producer` (a model name or describer URL); returns the entry."""
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        entry = {**entry, "media_sha256": digest, "kind": kind, "producer": producer,
                 "prompt_version": PROMPT_VERSION, "created": time.time()}
        name = f"{self._stem(digest, kind)}-{hashlib.sha256(producer.encode()).hexdigest()[:12]}.json"
        pending = self.directory / f".{uuid.uuid4().hex}.tmp"
        fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(entry, stream)
        os.replace(pending, self.directory / name)
        self.trim()
        return entry

    def _files(self):
        out = []
        for path in self.directory.glob("*.json"):
            with contextlib.suppress(OSError):
                stat = path.stat()
                out.append((stat.st_mtime, stat.st_size, path))
        return sorted(out)

    def trim(self):
        files = self._files()
        total = sum(size for _, size, _ in files)
        for _, size, path in files[:-1]:
            if total <= self.max_bytes:
                break
            total -= size
            path.unlink(missing_ok=True)

    def stats(self):
        files = self._files()
        return {"directory": str(self.directory), "entries": len(files),
                "bytes": sum(size for _, size, _ in files), "max_bytes": self.max_bytes}

    def clear(self):
        """Delete every cached surrogate; returns how many there were."""
        files = self._files()
        for _, _, path in files:
            path.unlink(missing_ok=True)
        return len(files)


def cache_for(config, bridge_dir):
    mib = (config.get("media") or {}).get("cache_mib")
    return Cache(Path(bridge_dir) / CACHE_DIR, CACHE_MAX_BYTES if mib is None else int(mib * 2**20))


# -- media parts -----------------------------------------------------------------------------------

def _media_key(part):
    """(modality, sha256) of a media part in either form (an OpenAI part, or a placeholder from
    messages.conversation_from_openai); None for any other part."""
    if not isinstance(part, dict):
        return None
    kind = part.get("type")
    if kind == "media":
        return part.get("modality"), part.get("sha256")
    if kind in messages.MEDIA_PARTS:
        return messages.MEDIA_PARTS[kind], messages.media_digest(part)
    return None


def request_media(message_list):
    """The modalities of the media parts in OpenAI messages (an empty set for text only), cheaply."""
    found = set()
    for message in message_list if isinstance(message_list, list) else ():
        content = message.get("content") if isinstance(message, dict) else None
        for part in content if isinstance(content, list) else ():
            if isinstance(part, dict) and part.get("type") in messages.MEDIA_PARTS:
                found.add(messages.MEDIA_PARTS[part["type"]])
    return found


def _covered(content, index, modality):
    """Whether the part after content[index] is text that already starts with the media's surrogate."""
    following = content[index + 1] if index + 1 < len(content) else None
    return (isinstance(following, dict) and following.get("type") == "text" and isinstance(following.get("text"), str)
            and following["text"].lstrip().startswith(f"[{MARKERS[modality]}:"))


class Part:
    """A media part the target cannot take: its message, modality, digest, the OpenAI part (with
    its data; None when the caller does not have it) and whether its surrogate follows it."""

    def __init__(self, message, modality, digest, part, covered):
        self.message, self.modality, self.digest, self.part, self.covered = message, modality, digest, part, covered


def lacking_parts(message_list, lacking, media_parts=None):
    """[Part] for every media part whose modality is in `lacking`, in order: of OpenAI messages, or
    of a converted conversation's messages with its "media_parts" (digest -> OpenAI part)."""
    out = []
    for number, message in enumerate(message_list):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for index, part in enumerate(content):
            key = _media_key(part)
            if key is None or key[0] not in lacking:
                continue
            raw = (media_parts or {}).get(key[1]) if part.get("type") == "media" else part
            out.append(Part(number, key[0], key[1], raw, _covered(content, index, key[0])))
    return out


# -- producers ---------------------------------------------------------------------------------------

class Producer:
    """A server that makes surrogates: `label` (a model name or a describer URL) for the cache and
    reports, its client, the modalities it is asked for (None: any its server has) and `hold`, a
    context manager factory entered once around its side requests (a slot save and restore)."""

    def __init__(self, label, server, *, modalities=None, allowed=None, hold=None):
        self.label, self.server, self.hold = label, server, hold
        self._modalities, self.allowed, self.error = modalities, allowed, None

    def has(self, modality):
        if self.allowed is not None and modality not in self.allowed:
            return False
        if self._modalities is None:
            try:
                self._modalities = self.server.props().get("modalities") or {}
            except ServerError as error:
                self.error, self._modalities = str(error)[:200], {}
        return bool(self._modalities.get(modality))

    def answer(self, content, prompt):
        """(text, finish reason) of one side request: the media `content`, then PROMPTS[prompt].
        On slot 0, the slot a hold saves and restores (a describer URL may serve several)."""
        reply, _ = self.server.chat(
            [{"role": "user", "content": [*content, {"type": "text", "text": PROMPTS[prompt]}]}],
            timeout=REQUEST_TIMEOUT, max_tokens=MAX_TOKENS[prompt], temperature=0, top_k=1, seed=0, id_slot=0,
            chat_template_kwargs={"enable_thinking": False})
        choice = (reply.get("choices") or [{}])[0]
        return ((choice.get("message") or {}).get("content") or "").strip(), choice.get("finish_reason")


def _multipart(fields, filename, content_type, data):
    """(body, Content-Type) of a multipart/form-data upload: `fields`, then `data` as "file"."""
    boundary = f"kvpark-{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
             f"Content-Type: {content_type}\r\n\r\n").encode()
    body += data + f"\r\n--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


class Transcriber(Producer):
    """A local OpenAI-compatible speech-to-text endpoint (POST .../v1/audio/transcriptions with the
    audio as a file, e.g. Lemonade's Whisper or a faster-whisper server): audio only, inline
    audio only, no conversation to keep. Its text is the transcript ("No speech." when empty)."""

    def __init__(self, url, model=TRANSCRIPTION_MODEL):
        super().__init__(url, None, allowed={"audio"})
        self.model = model

    def has(self, modality):
        return modality == "audio"

    def answer(self, content, prompt):
        part = next((p for p in content if isinstance(p, dict) and p.get("type") == "input_audio"), None)
        data = messages.media_bytes(messages.media_reference(part)) if part else None
        if data is None:
            raise ServerError(f"{self.label}: a transcription endpoint takes inline audio only")
        kind = str((part.get("input_audio") or {}).get("format") or "wav").lower()
        body, content_type = _multipart({"model": self.model, "response_format": "json", "temperature": "0"},
                                        f"audio.{kind}", AUDIO_TYPES.get(kind, f"audio/{kind}"), data)
        request = Request(self.label, data=body, method="POST", headers={"Content-Type": content_type})
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT) as reply:
                payload = json.loads(reply.read() or b"null")
        except HTTPError as error:
            with error:
                detail = error.read()[:300]
            raise ServerError(f"{self.label}: HTTP {error.code}: {detail!r}") from None
        except URLError as error:
            raise ServerError(f"{self.label}: {error.reason}") from None
        except ValueError:
            raise ServerError(f"{self.label}: the answer is not JSON") from None
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str):
            raise ServerError(f"{self.label}: the answer has no text")
        return text.strip() or "No speech.", "stop"


def slot_hold(server, slot_dir, log=lambda event: None):
    """A context manager factory for a registered model's side requests: its slot is saved into
    `slot_dir` first and restored after (the files then removed), so the conversation it holds
    is unchanged."""
    @contextlib.contextmanager
    def hold():
        name = f"kvbridge-hold-{uuid.uuid4().hex}.bin"
        files = [Path(slot_dir) / name, Path(slot_dir) / (name + ".resume")]
        try:
            server.slot_action("save", name)
            try:
                yield
            finally:
                try:
                    server.slot_action("erase")
                    server.slot_action("restore", name)
                except ServerError as error:       # the side request's answer stands; report the loss
                    log({"step": "surrogate-restore-failed", "error": str(error)[:200]})
        finally:
            for path in files:
                path.unlink(missing_ok=True)
    return hold


def model_producer(config, name, *, allowed=None, log=lambda event: None):
    cfg = registry.model(config, name)
    server = LlamaServer(cfg["url"], timeout=REQUEST_TIMEOUT)
    return Producer(name, server, allowed=allowed, hold=slot_hold(server, cfg["slot_dir"], log))


def describers(config, *, exclude=(), log=lambda event: None):
    """Producers for the describers set per modality (kvpark bridge media --describer), each asked
    only for the modalities it is set for; models in `exclude` are left out."""
    settings = config.get("media") or {}
    chosen = {}
    for modality, name in (settings.get("describers") or {}).items():
        if name and name not in exclude and modality in registry.MEDIA_POLICIES:
            chosen.setdefault(name, set()).add(modality)
    out = []
    for name, modalities in chosen.items():
        if name in config["models"]:
            out.append(model_producer(config, name, allowed=modalities, log=log))
        elif registry.is_transcription_url(name):
            out.append(Transcriber(name, settings.get("transcription_model") or TRANSCRIPTION_MODEL))
        else:
            out.append(Producer(name, LlamaServer(name, timeout=REQUEST_TIMEOUT), allowed=modalities))
    return out


# -- ffmpeg (optional) -----------------------------------------------------------------------------

def ffmpeg_tools():
    """(ffmpeg, ffprobe) when both are installed, else None."""
    tools = shutil.which("ffmpeg"), shutil.which("ffprobe")
    return tools if all(tools) else None


def _run(argv):
    try:
        result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                timeout=FFMPEG_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 and result.stdout else None


class Video:
    """A video's keyframes and sound track, read with ffmpeg from its bytes in a private file."""

    def __init__(self, tools, path):
        self.tools, self.path = tools, str(path)

    def duration(self):
        out = _run([self.tools[1], "-v", "error", "-show_entries", "format=duration", "-of",
                    "default=noprint_wrappers=1:nokey=1", self.path])
        try:
            value = float(out.decode().strip()) if out else 0.0
        except ValueError:
            return 0.0
        return value if math.isfinite(value) and value > 0 else 0.0

    def keyframes(self):
        """[(seconds, JPEG bytes)]: one frame per KEYFRAME_STEP_S seconds (at most MAX_KEYFRAMES),
        each in the middle of its stretch, scaled to KEYFRAME_WIDTH."""
        duration = self.duration()
        count = max(1, min(MAX_KEYFRAMES, math.ceil(duration / KEYFRAME_STEP_S))) if duration else 1
        frames = []
        for index in range(count):
            at = round((index + 0.5) * duration / count, 2)
            jpeg = _run([self.tools[0], "-v", "error", "-ss", f"{at:.2f}", "-i", self.path, "-frames:v", "1",
                         "-vf", f"scale='min({KEYFRAME_WIDTH},iw)':-2", "-map_metadata", "-1", "-fflags", "+bitexact",
                         "-c:v", "mjpeg", "-q:v", str(KEYFRAME_QUALITY), "-f", "image2pipe", "pipe:1"])
            if jpeg:
                frames.append((at, jpeg))
        return frames

    def sound(self):
        """The sound track as 16 kHz mono WAV bytes, or None (a video without sound)."""
        wav = _run([self.tools[0], "-v", "error", "-i", self.path, "-vn", "-ac", "1", "-ar", "16000",
                    "-map_metadata", "-1", "-fflags", "+bitexact", "-f", "wav", "pipe:1"])
        return wav if wav and len(wav) > 44 else None


def _data_url(jpeg):
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def _seconds(at):
    return f"{at:g} s"


# -- making surrogates -------------------------------------------------------------------------------

class _Makers:
    """Side requests to the producers, in order, each producer's hold entered once (on `stack`)."""

    def __init__(self, producers, stack, tools, workdir, log):
        self.producers, self.stack, self.tools, self.workdir, self.log = producers, stack, tools, workdir, log
        self.held, self.errors, self.videos, self.sounds = set(), [], {}, {}

    def can(self, modality):
        """Whether some producer has `modality` (asks each server's /props once)."""
        return any(producer.has(modality) for producer in self.producers)

    def ask(self, modality, content, prompt):
        """(text, producer label) from the first producer with `modality` that answers, or None."""
        for producer in self.producers:
            if not producer.has(modality):
                continue
            try:
                if producer.hold is not None and producer.label not in self.held:
                    self.stack.enter_context(producer.hold())
                    self.held.add(producer.label)
                text, finish = producer.answer(content, prompt)
                if not text:
                    raise ServerError(f"{producer.label} answered with no text")
                if finish == "length":
                    text += " [...]"
                return text, producer.label
            except (ServerError, OSError) as error:
                self.errors.append(f"{producer.label}: {error}"[:300])
                self.log({"step": "surrogate-failed", "producer": producer.label, "error": str(error)[:200]})
        return None

    def video(self, part, digest):
        """A Video for an inline video part (ffmpeg installed), else None."""
        if digest in self.videos:
            return self.videos[digest]
        video, data = None, messages.media_bytes(messages.media_reference(part))
        if self.tools and data:
            Path(self.workdir).mkdir(parents=True, exist_ok=True, mode=0o700)
            directory = self.stack.enter_context(tempfile.TemporaryDirectory(prefix=".media-", dir=self.workdir))
            path = Path(directory) / "video"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
            video = Video(self.tools, path)
        self.videos[digest] = video
        return video

    def sound_transcript(self, digest, video):
        """(transcript and producer label or None, whether the video has sound) for a video's sound
        track; the transcript is None when no producer for audio answered. Asked once per video;
        a video ffmpeg cannot read (or no ffmpeg) counts as without sound."""
        if digest not in self.sounds:
            wav = video.sound() if video else None
            part = {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav or b"").decode(),
                                                           "format": "wav"}}
            self.sounds[digest] = (self.ask("audio", [part], "audio") if wav else None), bool(wav)
        return self.sounds[digest]

    def add_sound(self, entry, part):
        """A video's entry with its sound track ("audio"): its transcript, or, when the video has
        sound no producer transcribed, SOUND_UNTRANSCRIBED and "sound_missing"."""
        got, sound = self.sound_transcript(part.digest, self.video(part.part, part.digest))
        if got:
            entry = {key: value for key, value in entry.items() if key != "sound_missing"}
            return {**entry, "audio": wrap(VIDEO_AUDIO_MARKER, got[0]), "audio_producer": got[1]}
        if sound:
            return {**entry, "audio": wrap(VIDEO_AUDIO_MARKER, SOUND_UNTRANSCRIBED), "sound_missing": True}
        return entry

    def make(self, part, kind):
        """(entry, producer label) for one media part as `kind` (a modality, or FRAMES), or None."""
        if kind == "vision":
            got = self.ask("vision", [part.part], "vision")
            return got and ({"text": wrap(MARKERS["vision"], got[0])}, got[1])
        if kind == "audio":
            got = self.ask("audio", [part.part], "audio")
            return got and ({"text": wrap(MARKERS["audio"], got[0])}, got[1])
        video = self.video(part.part, part.digest)
        if kind == FRAMES:
            frames = video.keyframes() if video else []
            if not frames:
                return None
            entry = {"text": f"[{VIDEO_FRAMES_MARKER} {', '.join(_seconds(at) for at, _ in frames)}]",
                     "frames": [{"at": at, "jpeg": base64.b64encode(jpeg).decode()} for at, jpeg in frames]}
            return self.add_sound(entry, part), "ffmpeg"
        reference = messages.media_reference(part.part)
        got = self.ask("video", [{"type": "input_video", "input_video": {"url": reference}}], "video")
        if got is None and video:
            # no producer takes video: a vision producer describes its keyframes
            content = []
            for at, jpeg in video.keyframes():
                content += [{"type": "text", "text": f"[{_seconds(at)}]"},
                            {"type": "image_url", "image_url": {"url": _data_url(jpeg)}}]
            got = self.ask("vision", content, "keyframes") if content else None
        if got is None:
            return None
        return self.add_sound({"text": wrap(MARKERS["video"], got[0])}, part), got[1]


class Resolution:
    """Surrogates for one conversation and target: `entries` by (sha256, kind) for apply() and
    rewrite_messages(); `parts`, one record per media part (message, modality, producer, cached,
    seconds; "duplicate" for media met earlier in the conversation, "sound" when a video's sound
    track is missing); `missing` {modality: note} for media nothing could replace; `partial`
    {modality: note} for surrogates that lack something (a video's sound track); `errors` of
    producers."""

    def __init__(self):
        self.entries, self.parts, self.missing, self.partial, self.errors, self.seconds = {}, [], {}, {}, [], 0.0

    def report(self):
        out = {"parts": self.parts, "seconds": round(self.seconds, 3)}
        for key in ("missing", "partial", "errors"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        return out


def _missing_note(modality, producers):
    what = {"vision": "image", "audio": "audio", "video": "video"}[modality]
    asked = ", ".join(producer.label for producer in producers)
    return (f"no {what} surrogate could be made: none is cached, and no producer{f' ({asked})' if asked else ''} "
            f"has a {modality} encoder that answered (set one with kvpark bridge media --describer)")


def _sound_note(producers):
    asked = ", ".join(producer.label for producer in producers)
    return (f"a video's sound track could not be transcribed: no producer{f' ({asked})' if asked else ''} has "
            "audio input that answered (set one with kvpark bridge media --describer audio=...)")


def resolve(parts, *, frames, producers, cache, workdir, log=lambda event: None):
    """The surrogates of `parts` (lacking_parts): from the conversation, the cache, else the
    producers in order (side requests, each producer's hold entered once around all of its own).
    A cached video surrogate whose sound track was missing is completed when a producer for
    audio is there now. `frames`: the target has vision, so a video becomes its keyframes when
    ffmpeg can read it."""
    started = time.perf_counter()
    resolution = Resolution()
    tools = ffmpeg_tools() if any(part.modality == "video" for part in parts) else None
    with contextlib.ExitStack() as stack:
        makers = _Makers(producers, stack, tools, workdir, log)
        for part in parts:
            record = {"message": part.message, "modality": part.modality}
            if part.covered:
                resolution.parts.append({**record, "producer": "conversation", "cached": False, "seconds": 0.0})
                continue
            kinds = [FRAMES, "video"] if part.modality == "video" and frames and tools else [part.modality]
            for kind in kinds:
                # the same media earlier in this conversation, the cache, else a producer
                entry, how, seconds = resolution.entries.get((part.digest, kind)), "duplicate", 0.0
                if entry is None:
                    entry, how = cache.get(part.digest, kind), "cached"
                    if (entry is not None and entry.get("sound_missing") and part.part is not None
                            and makers.can("audio")):
                        # made when nothing could transcribe its sound track: try again
                        t = time.perf_counter()
                        completed = makers.add_sound(entry, part)
                        seconds = round(time.perf_counter() - t, 3)
                        if not completed.get("sound_missing"):
                            entry, how = cache.put(part.digest, kind, entry["producer"], {
                                **completed, "seconds": round((entry.get("seconds") or 0) + seconds, 3)}), "made"
                if entry is None and part.part is not None:
                    t = time.perf_counter()
                    made = makers.make(part, kind)
                    seconds = round(time.perf_counter() - t, 3)
                    if made is not None:
                        entry, how = cache.put(part.digest, kind, made[1], {**made[0], "seconds": seconds}), "made"
                if entry is not None:
                    resolution.entries[(part.digest, kind)] = entry
                    record.update(kind=kind, producer=entry.get("producer"), cached=how == "cached",
                                  seconds=seconds if how == "made" else 0.0,
                                  chars=len(entry.get("text", "")) + len(entry.get("audio", "")))
                    if how == "duplicate":
                        record["duplicate"] = True
                    if entry.get("frames"):
                        record["frames"] = len(entry["frames"])
                    if entry.get("audio_producer"):
                        record["audio_producer"] = entry["audio_producer"]
                    if entry.get("sound_missing"):
                        record["sound"] = "not transcribed"
                        resolution.partial.setdefault("video", _sound_note(producers))
                    resolution.parts.append(record)
                    break
            else:
                note = (_missing_note(part.modality, producers) if part.part is not None
                        else f"the {part.modality} part's data is not available to make a surrogate")
                resolution.missing.setdefault(part.modality, note)
        resolution.errors = makers.errors
    resolution.seconds = time.perf_counter() - started
    log({"step": "surrogates", "parts": len(resolution.parts), "missing": sorted(resolution.missing),
         "seconds": round(resolution.seconds, 3)})
    return resolution


# -- applying them -------------------------------------------------------------------------------------

def _swap(content, entries, lacking, frame_part):
    """New content for one message: each media part of a `lacking` modality replaced by its
    surrogate (or dropped when its surrogate follows it); unchanged content is returned as is."""
    if not isinstance(content, list):
        return content
    out, changed = [], False
    for index, part in enumerate(content):
        key = _media_key(part)
        if key is None or key[0] not in lacking:
            out.append(part)
            continue
        modality, digest = key
        if _covered(content, index, modality):
            changed = True
            continue
        entry = entries.get((digest, FRAMES)) or entries.get((digest, modality))
        if entry is None:
            out.append(part)
            continue
        changed = True
        out.append({"type": "text", "text": entry["text"]})
        for frame in entry.get("frames") or ():
            out.append(frame_part(base64.b64decode(frame["jpeg"])))
        if entry.get("audio"):           # a video's sound track
            out.append({"type": "text", "text": entry["audio"]})
    if not changed:
        return content
    if any(_media_key(part) for part in out):
        return out
    # no media left: one string (what the bridge renders and the real request carries alike)
    return "\n".join(part.get("text") or "" for part in out if isinstance(part, dict))


def apply(conversation, resolution, lacking):
    """The conversation (messages.conversation_from_openai form) as the target gets it: each media
    part of a `lacking` modality replaced by its surrogate; a video's keyframes become image
    placeholders (their parts kept in "media_parts")."""
    kept = dict(conversation.get("media_parts") or {})

    def frame_part(jpeg):
        digest = hashlib.sha256(jpeg).hexdigest()
        kept[digest] = {"type": "image_url", "image_url": {"url": _data_url(jpeg)}}
        return {"type": "media", "modality": "vision", "sha256": digest}
    out = {**conversation, "messages": [{**message, "content": _swap(message.get("content"), resolution.entries,
                                                                      lacking, frame_part)}
                                        for message in conversation["messages"]]}
    if kept:
        out["media_parts"] = kept
    return out


def rewrite_messages(message_list, resolution, lacking):
    """OpenAI messages with the same replacements as apply() (the real request for the target)."""
    def frame_part(jpeg):
        return {"type": "image_url", "image_url": {"url": _data_url(jpeg)}}
    return [({**message, "content": _swap(message["content"], resolution.entries, lacking, frame_part)}
             if isinstance(message, dict) and isinstance(message.get("content"), list) else message)
            for message in message_list]


def rewrite_request(config, request, target, *, bridge_dir, source=None, log=lambda event: None):
    """(request, summary) for a chat request to `target` (a registered bridge model): each media
    part its server cannot take replaced by its surrogate (the conversation, the cache, the
    `source` model when named, then the describers), as switch() renders it. request is None
    when nothing needs replacing or not every such part could be replaced (the request then goes
    out unchanged); summary is None when the request has no media the target lacks, else
    {"target", "lacking", "parts", "seconds", "missing"?, "partial"?, "errors"?, "skipped"?}."""
    message_list = request.get("messages")
    present = request_media(message_list)
    if not present:
        return None, None
    server = LlamaServer(registry.model(config, target)["url"])
    modalities = server.props().get("modalities") or {}
    lacking = {modality for modality in present if not modalities.get(modality)}
    if not lacking:
        return None, None
    summary = {"target": target, "lacking": sorted(lacking)}
    skipped = sorted(modality for modality in lacking if not converts(config, target, modality))
    if skipped:
        # the policy keeps these media: the request goes out as it is (and the target refuses it)
        return None, {**summary, "skipped": skipped}
    producers = describers(config, exclude={target, source}, log=log)
    if source and source != target:
        producers.insert(0, model_producer(config, source, log=log))
    resolution = resolve(lacking_parts(message_list, lacking), frames=bool(modalities.get("vision")),
                         producers=producers, cache=cache_for(config, bridge_dir), workdir=bridge_dir, log=log)
    summary.update(resolution.report())
    if resolution.missing:
        return None, summary
    return {**request, "messages": rewrite_messages(message_list, resolution, lacking)}, summary
