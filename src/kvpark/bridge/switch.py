"""Move a live conversation from one llama-server to another, translating its cache.

    report = switch(config, conversation, "qwen38", "muse")

0. The switch is skipped (``report["skipped"]`` = a reason code, ``report["note"]``) when the
   source is registered as a target only ("source-refused": kvpark bridge add --no-source, for a model
   the hub translates out of worse than no translation), or when the translator cannot handle it: an architecture the engine cannot translate, an embedding model
   or a LongRoPE model, a vocabulary the engine cannot read (with an engine older than the
   "vocabularies" feature: any but GPT-2 byte-level BPE; see capabilities.py), media the target
   cannot take that get no text surrogate (the media policy says skip, or nothing could make
   one; or, with an engine older than the "media-source" feature, media in the source's
   cache), a prompt too short to be worth translating, a DSA target without indexer keys for a
   prompt this long, or a draft model whose state the bridge cannot synthesize; or, at step 4,
   when the engine daemon's memory guard refuses the job ("engine-memory"), the hub has no
   indexer keys after all ("no-indexer-keys"), or the engine cannot reproduce the target's
   rendering of the conversation ("template-mismatch"; with the request's template options it
   first falls back to the default rendering, see switch(), and "template-options" means too
   little of that rendering is shared with the request's). The target then simply prefills the
   conversation natively.
   Media the target's server has no encoder for (an image going to a text-only model) are
   replaced first by text surrogates (surrogates.py: "[Image description: ...]" from the cache,
   the conversation, the source model or a describer; report["surrogates"]); a side request to
   the source saves its slot first (step 3's save) and restores it after. The real request must
   carry the same surrogates (surrogates.rewrite_request; the Hermes middleware does it).
1. The target server renders the conversation with its own template (/apply-template, with the
   generation prompt and the request's template options) and tokenizes it (/tokenize with
   pieces): exactly what it will process. Media parts are rendered as the target's media
   marker, as the server renders them.
2. A layout template of the target (a one-token native slot save, with its .resume on
   kvpark-patched servers) is made once per target model, server build and speculative
   configuration. Its .resume tells whether the server has a draft model and of which kind.
3. The source server saves its slot (its cache for the conversation) into its slot directory.
4. The engine translates the source slot into a target slot file holding [0, T0),
   T0 = min(P - tail, start of the newest message, start of the first message with media the
   target reads or the source's cache holds);
   see experiments/universal_kv/engine.py (it renders the conversation as the server does,
   the request's template options included, and checks that the target's tokens spell it).
   It runs on the resident engine daemon (the job is sent over its socket, so the
   conversation never touches the disk) or, where no daemon can run, in a one-shot process
   (engine_client.py). The job names each server's llama.cpp build (/props build_info), whose
   pin's layouts the engine's specs take. A hybrid target's recurrent cell at T0 - 1 starts
   from its registered mean (kvpark bridge add --recurrent-mean; report["recurrent"]), else
   from zeros. The engine's T0 is final (a DeepSeek V4 target rounds it down to whole
   128-token blocks). On a kvpark-patched target the bridge then writes the matching .resume
   (resume.py: the draft state of a translated restore).
5. The target slot is erased and the translated file restored; every bridge file is deleted
   as soon as it has been used.
6. Optionally (never with media the target reads itself) a warm request (one generated token)
   prefills the native tail and reports how many prompt tokens the target reused (`cache_n`,
   expected T0).

The target's resident conversation is replaced: park it first if it matters. Output stays
correct whatever the translation quality, because llama-server only reuses cells whose tokens
match the prompt; translation changes the values of the reused cells, never which tokens the
target conditions on.
"""

import contextlib
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from . import capabilities, engine_client, messages, registry, resume, surrogates
from .client import LlamaServer, ServerError

TEMPLATE_TEXT = "Hello"
TEMPLATE_FORMAT = 2            # 2: the template keeps its .resume; the identity includes the draft setup
MIN_HANDOVER_TOKENS = 1024     # below tail + this, a switch translates nothing and the target prefills
DRAFT_WINDOW = 2048            # sliding window of the DFlash drafts in use (dflash.attention.sliding_window)
MEDIA_STAND_IN = "kvpark-media-message"
# /slots speculative.types that run each draft kind resume.classify_draft tells from the state
# ("shared": a draft that writes no cells of its own, Gemma 4's MTP head over the target's cache)
DRAFT_TYPES = {"dflash": {"draft-dflash", "draft-dspark"}, "mtp": {"draft-mtp"}, "shared": {"draft-mtp"}}
# how the engine refuses a DSA target without indexer keys (translate.IndexerKeysUnavailable)
INDEXER_REFUSAL = "no indexer keys"


def _bridge_dir(config_path=None):
    directory = Path(config_path or registry.default_path()).parent
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return directory


def _private_write(path, text):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(text)


def _keep_job(directory, job, files):
    """KVPARK_BRIDGE_KEEP_JOB (debugging only): copy the job and its inputs to `directory`. They
    hold conversation text and caches, so the directory is private."""
    import shutil
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    _private_write(directory / f"job-{uuid.uuid4().hex}.json", json.dumps(job))
    for item in files:
        shutil.copy2(item, directory / Path(item).name)


def speculative_config(server):
    """The server's speculative decoding setup as /slots reports it ("none", "draft-dflash", ...).

    /props always reports "none" (it copies only the sampling settings); a slot reports the
    real setup in its params once a task has run on it, so an untouched slot of a server that
    speculates gets a one-token completion first (its cache is empty anyway)."""
    def slot():
        slots = server.slots()
        return slots[0] if isinstance(slots, list) and slots else {}
    try:
        info = slot()
        if "speculative.types" not in (info.get("params") or {}):
            if not info.get("speculative"):
                return "none"
            server.complete([token for token, _ in server.tokenize(TEMPLATE_TEXT, add_special=True)], n_predict=1)
            info = slot()
        return str((info.get("params") or {}).get("speculative.types", "unknown"))
    except ServerError:      # the server runs without the /slots endpoint
        return "unknown"


def layout_template(server, model_cfg, bridge_dir):
    """(slot file, .resume or None on upstream servers, speculative setup): a one-token native
    slot save of the target, cached per weights file, server build, slot count and speculative
    setup."""
    props = server.props()
    speculative = speculative_config(server)
    identity = hashlib.sha256(json.dumps([TEMPLATE_FORMAT, model_cfg["gguf"], props.get("build_info"),
                                          Path(model_cfg["gguf"]).stat().st_size, props.get("total_slots"),
                                          speculative]).encode()).hexdigest()[:16]
    cached = bridge_dir / "templates" / f"{identity}.bin"
    cached_resume = Path(str(cached) + ".resume")
    if cached.is_file():
        return cached, cached_resume if cached_resume.is_file() else None, speculative
    cached.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    server.slot_action("erase")
    tokens = [token for token, _ in server.tokenize(TEMPLATE_TEXT, add_special=True)]
    server.complete(tokens, n_predict=1)
    name = f"kvbridge-template-{uuid.uuid4().hex}.bin"
    server.slot_action("save", name)
    saved = Path(model_cfg["slot_dir"]) / name
    saved_resume = Path(str(saved) + ".resume")
    # a kvpark-patched server writes the draft state next to the slot; it is the layout of
    # every translated restore's .resume, so it is kept (moved first: the slot file's presence
    # marks a complete template)
    if saved_resume.is_file():
        os.replace(saved_resume, cached_resume)
    else:
        cached_resume.unlink(missing_ok=True)
    os.replace(saved, cached)
    server.slot_action("erase")
    return cached, cached_resume if cached_resume.is_file() else None, speculative


def _template_tokens(template):
    """The template's prompt length (settles draft layouts the bytes alone leave open), if readable."""
    try:
        return resume.slot_token_count(template)
    except (OSError, resume.StateBlobError):
        return None


def _drop_template(template, template_resume):
    for path in (template, template_resume):
        if path is not None:
            Path(path).unlink(missing_ok=True)


def native_prefix(server, model_cfg, conversation, tokens, bridge_dir, *, minimum=4, options=None):
    """(slot file, H): the target's own cache for the prompt's first H tokens, cached by token digest.

    H covers the system message (rendered alone, with the request's template options, its tokens
    are a prefix of the full prompt) or at least `minimum` tokens (the attention-sink start of the
    prompt). K/V of a prefix depend only on the prefix, so these values are exact; a system prompt
    shared by many conversations is prefilled once. The file is only read by the engine, never
    restored, so a patched server's .resume for it is not needed."""
    ids = [token for token, _ in tokens]
    H = 0
    messages_ = conversation["messages"]
    if messages_ and messages_[0]["role"] == "system" and len(messages_) > 1:
        head = server.apply_template(messages_[:1], conversation.get("tools"), options=options)
        head_ids = [token for token, _ in server.tokenize(head, add_special=True)]
        while H < min(len(head_ids), len(ids)) and head_ids[H] == ids[H]:
            H += 1
        H = max(0, H - 2)      # stay clear of a template tail that renders differently in context
    H = min(max(H, minimum), len(ids) - 1)
    props = server.props()
    digest = hashlib.sha256(json.dumps([model_cfg["gguf"], props.get("build_info"), ids[:H]]).encode()).hexdigest()
    cached = bridge_dir / "prefix" / f"{digest[:32]}.bin"
    if not cached.is_file():
        cached.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        server.slot_action("erase")
        server.complete(ids[:H], n_predict=1)
        name = f"kvbridge-prefix-{uuid.uuid4().hex}.bin"
        server.slot_action("save", name)
        saved = Path(model_cfg["slot_dir"]) / name
        os.replace(saved, cached)
        Path(str(saved) + ".resume").unlink(missing_ok=True)
        _trim_prefix_cache(cached.parent)
    return cached, H


def _trim_prefix_cache(directory, max_bytes=2 * 1024**3):
    files = sorted(directory.glob("*.bin"), key=lambda path: path.stat().st_mtime)
    total = sum(path.stat().st_size for path in files)
    for path in files[:-1]:
        if total <= max_bytes:
            break
        total -= path.stat().st_size
        path.unlink(missing_ok=True)


def _text_join(server, conversation):
    """How the target's template joins a message's text parts (llama-server passes text-only
    parts to the template as a list), or None when no plain string renders the same."""
    if not messages.has_text_parts(conversation):
        return ""
    first, second = "kvpark-part-one", "kvpark-part-two"
    try:
        rendered = server.apply_template([{"role": "user", "content": [{"type": "text", "text": first},
                                                                         {"type": "text", "text": second}]}])
        start, end = rendered.find(first), rendered.find(second)
        if start < 0 or end < start + len(first):
            return None
        join = rendered[start + len(first):end]
        same = server.apply_template([{"role": "user", "content": first + join + second}]) == rendered
    except ServerError:       # a template that refuses a lone user message
        return None
    return join if same else None


def _shared_prefix(a, b):
    """How many leading tokens two token id lists share."""
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def _source_saw(conversation):
    """Index of the last message the source server has processed: the conversation up to its
    last assistant message (what follows is new in this request)."""
    roles = [m["role"] for m in conversation["messages"]]
    return max((i for i, role in enumerate(roles) if role == "assistant"), default=-1)


def _media_start(server, conversation, index, ids, *, prompt, marker, options=None):
    """Target tokens before the content of message `index`, at most: the common prefix of the
    full rendering and the conversation rendered up to that message with a stand-in content
    (both with the request's template options), cut where the stand-in starts.

    The stand-in keeps the message's role (media sit in user and tool messages), so the
    rendering does not end with the assistant message before it, which the server would
    continue (and refuses to continue when it has tool calls). A rendering the server refuses
    anyway falls back to the text before the first media marker of `prompt`."""
    if index == 0:
        return 0
    messages_ = conversation["messages"]
    stand_in = {key: value for key, value in messages_[index].items() if key != "tool_calls"}
    stand_in["content"] = MEDIA_STAND_IN
    try:
        head = server.apply_template(messages_[:index] + [stand_in], conversation.get("tools"), options=options)
        cut = head.find(MEDIA_STAND_IN)
    except ServerError:
        cut = -1
    if cut < 0:
        head, cut = prompt, prompt.find(marker) if marker else -1
    if cut < 0:
        return 0              # no rendering shows where the media starts
    head_ids = [token for token, _ in server.tokenize(head[:cut], add_special=True)]
    n = 0
    while n < min(len(head_ids), len(ids)) and head_ids[n] == ids[n]:
        n += 1
    return n


def switch(config, conversation, source, target, *, tail=None, fill=None, warm=True, config_path=None,
           engine_timeout=900, prefix=None, template_options=None, log=lambda event: None):
    """See the module docstring. template_options: the real request's fields that change its
    rendering (messages.template_options: chat_template_kwargs, reasoning_effort). When the
    request renders otherwise than the template's defaults and the engine renders with the
    options too ("template-options" in its features), the engine translates the request's own
    rendering (report["template_options"]["engine"] true). Otherwise, and when the engine cannot
    reproduce the server's bytes with them (an EngineError of kind "template": the server runs
    settings the engine cannot see), the engine translates the default rendering: T0 then stops
    at the tokens both renderings share, and a switch left with too little to translate is
    skipped ("template-options")."""
    if source == target:
        raise ValueError("source and target are the same model")
    if not config.get("engine") or not config.get("hub"):
        raise ValueError("no bridge engine configured; run kvpark bridge engine")
    src_cfg, tgt_cfg = registry.model(config, source), registry.model(config, target)
    if src_cfg["url"] == tgt_cfg["url"]:
        raise ValueError("source and target are the same server")
    report = {"source": source, "target": target, "timings": {}}
    started = time.perf_counter()

    def skip(code, note):
        report.update(skipped=code, note=note + "; the target prefills the conversation natively")
        report["timings"]["total_s"] = round(time.perf_counter() - started, 3)
        log({"step": "skipped", "reason": code})
        return report

    # 0. a model registered as a target only (kvpark bridge add --no-source): its cache is never translated
    if src_cfg.get("source") is False:
        return skip("source-refused", f"{source} is registered as a target only (kvpark bridge add --no-source): "
                                      "its cache is not translated")
    # what the translator can handle (first: a model it cannot handle has no hub adapter either)
    source_arch = registry.model_arch(config, source, path=config_path)
    target_arch = registry.model_arch(config, target, path=config_path)
    # embedding models and LongRoPE first: their GGUF, not their architecture, rules them out
    rule = (capabilities.model(source, source_arch, src_cfg["gguf"])
            or capabilities.model(target, target_arch, tgt_cfg["gguf"])
            or capabilities.check((source, source_arch), (target, target_arch))
            # a DeepSeek V4 target's pooled caches need the hub's head decoders
            or capabilities.heads(tgt_cfg["hub_name"], target_arch, registry.hub_manifest(config)))
    if rule:
        return skip(*rule)
    known = registry.hub_models(config)
    for cfg, name in ((src_cfg, source), (tgt_cfg, target)):
        if cfg["hub_name"] not in known:
            raise ValueError(f"the hub has no adapter for {name} ({cfg['hub_name']})")
    A, B = LlamaServer(src_cfg["url"]), LlamaServer(tgt_cfg["url"])
    bridge_dir = _bridge_dir(config_path)
    asked = {}
    source_name = f"kvbridge-{uuid.uuid4().hex}.bin"
    source_file = Path(src_cfg["slot_dir"]) / source_name
    created = []

    def engine_features():
        """What the engine that will run the job handles beyond older ones (asked once, when needed)."""
        if "features" not in asked:
            asked["features"] = engine_client.features(config, bridge_dir, log=log)
            report["engine_features"] = sorted(asked["features"])
        return asked["features"]

    def build_info(role, server):
        """The server's llama.cpp build (/props build_info, asked once; None when it reports none)."""
        key = f"build_info:{role}"
        if key not in asked:
            try:
                props = server.props()
            except (ServerError, OSError, ValueError):     # the engine then uses its default pin
                props = None
            info = props.get("build_info") if isinstance(props, dict) else None
            asked[key] = info if isinstance(info, str) else None
        return asked[key]

    def save_source():
        """Step 3, once: the source's slot (its cache for the conversation) into its slot directory."""
        if "source_saved" in asked:
            return
        if source_file not in created:
            created.extend([source_file, Path(str(source_file) + ".resume")])
        reply, seconds = A.slot_action("save", source_name)
        asked["source_saved"] = True
        report.update(source_tokens=reply.get("n_saved"), source_bytes=reply.get("n_written"))
        report["timings"]["source_save_s"] = round(seconds, 3)
        log({"step": "source-saved", "tokens": reply.get("n_saved")})

    @contextlib.contextmanager
    def hold_source():
        """Around side requests to the source (surrogates): its slot is saved first (the save the
        engine reads at step 4) and restored from that file after, so its cache is unchanged. A
        failed save fails the switch (step 3 needs it anyway) instead of passing the media on to
        the next producer."""
        try:
            save_source()
        except ServerError as error:
            raise RuntimeError(f"{source}'s slot save failed: {error}") from None
        try:
            yield
        finally:
            t = time.perf_counter()
            try:
                A.slot_action("erase")
                A.slot_action("restore", source_name)
            except ServerError as error:     # the surrogates stand; the source re-reads its conversation
                report["source_restore_error"] = str(error)[:300]
            report["timings"]["source_restore_s"] = round(time.perf_counter() - t, 3)

    # vocabularies the engine reads (an engine older than the "vocabularies" feature: GPT-2 byte-level BPE)
    vocabularies = [(name, (capabilities.gguf_facts(cfg["gguf"]) or {}).get("tokenizer"))
                    for name, cfg in ((source, src_cfg), (target, tgt_cfg))]
    if any(model not in (None, *capabilities.TOKENIZERS) for _, model in vocabularies):
        rule = capabilities.tokenizer(*vocabularies, engine_features())
        if rule:
            return skip(*rule)

    try:
        media = messages.media(conversation)
        marker, cap = None, None
        if media:
            target_props = B.props()
            target_modalities = target_props.get("modalities") or {}
            source_modalities = A.props().get("modalities") or {}
            saw = _source_saw(conversation)
            # media the target has no encoder for become text surrogates, unless the policy says skip
            replaced = {modality for modality in media if not target_modalities.get(modality)
                        and surrogates.converts(config, target, modality)}
            rule = capabilities.media(media, source=source, target=target, source_modalities=source_modalities,
                                      target_modalities=target_modalities, source_saw=saw, target_arch=target_arch,
                                      replaced=replaced)
            source_media = bool(rule) and rule[0] == "media-source" and "media-source" in engine_features()
            if source_media:
                rule = None         # the engine reads the source's cache up to its first medium
            if rule:
                return skip(*rule)
            report["media"] = {"modalities": sorted(media)}
            if replaced:
                t = time.perf_counter()
                producers = [surrogates.Producer(source, A, modalities=source_modalities, hold=hold_source),
                             *surrogates.describers(config, exclude={source, target}, log=log)]
                resolution = surrogates.resolve(
                    surrogates.lacking_parts(conversation["messages"], replaced, conversation.get("media_parts")),
                    frames=bool(target_modalities.get("vision")), producers=producers,
                    cache=surrogates.cache_for(config, bridge_dir), workdir=bridge_dir, log=log)
                report["surrogates"] = resolution.report()
                report["timings"]["surrogates_s"] = round(time.perf_counter() - t, 3)
                if resolution.missing:
                    modality = sorted(resolution.missing)[0]
                    return skip(f"target-no-{modality}", f"the conversation has {modality} input, {target}'s "
                                f"server has no {modality} encoder (mmproj), and {resolution.missing[modality]}")
                conversation = surrogates.apply(conversation, resolution, replaced)
                rule = capabilities.pooled_media(target, target_arch, set(messages.media(conversation)))
                if rule:            # a video's keyframes, as images
                    return skip(*rule)
            # T0 stops at the first message whose media the target reads itself, and at the first
            # one whose media the source's cache holds (its encoder's cells: the engine reads the
            # source's text before them; their surrogates are prefilled natively)
            native = messages.media(conversation)
            held = [first for modality, first in media.items()
                    if modality in replaced and source_modalities.get(modality) and first <= saw]
            cap = min([*native.values(), *held], default=None)
            if cap is not None:
                report["media"]["first_message"] = cap
            if native:
                report["media"]["native"] = sorted(native)
                marker = target_props.get("media_marker")
                if not marker:
                    return skip("target-no-media-marker", f"{target}'s server does not report its media marker")
            if source_media:
                report["media"]["source_media"] = True
            media = native
        join = _text_join(B, conversation)
        if join is None:
            return skip("text-parts", f"{target}'s chat template renders text parts differently from plain text")
        n_media = messages.media_count(conversation)
        conversation = messages.flatten(conversation, marker=marker, text_join=join)

        # 1. the target's prompt, as the target will tokenize it
        t = time.perf_counter()
        prompt = B.apply_template(conversation["messages"], conversation.get("tools"))
        # as the chat endpoint does: BOS (when the model wants one) is added as a token, not text
        tokens = B.tokenize(prompt, add_special=True, parse_special=True)
        # renderings the engine can translate: (prompt, tokens, template options for the engine)
        default = request = (prompt, tokens, None)
        shared = None
        if template_options:
            rendered = B.apply_template(conversation["messages"], conversation.get("tools"), options=template_options)
            if rendered != prompt:
                # the real request renders otherwise (enable_thinking, another reasoning effort)
                request = (rendered, B.tokenize(rendered, add_special=True, parse_special=True), None)
                shared = _shared_prefix([i for i, _ in tokens], [i for i, _ in request[1]])
                report["template_options"] = {"options": sorted(template_options), "shared_tokens": shared,
                                              "engine": "template-options" in engine_features()}
                if report["template_options"]["engine"]:
                    request = (*request[:2], template_options)
        request_tokens = request[1]
        report["target_prompt_tokens"] = len(request_tokens)
        # /tokenize reads each media marker as text, while the server puts the media's own cells
        # there (often fewer): the markers count on top of the tail, so the tail stays native
        markers = n_media * len(B.tokenize(marker)) if media else 0
        base_tail = int(tail or tgt_cfg.get("tail") or config.get("tail") or 2048) + markers
        minimum = int(tgt_cfg.get("min_handover", config.get("min_handover", MIN_HANDOVER_TOKENS)))
        media_starts = {}

        def plan(rendering, window=None):
            """(effective tail, tokens to hand over, skip or None) for the engine translating `rendering`."""
            prompt_, tokens_, options_ = rendering
            effective_tail = base_tail
            handover = len(tokens_) - effective_tail
            if cap is not None:
                # the translated cache stops before the first message with media (or a surrogate of
                # media the source's cache holds): the target prefills that message and everything
                # after natively
                key = id(rendering)
                if key not in media_starts:
                    media_starts[key] = _media_start(B, conversation, cap, [i for i, _ in tokens_],
                                                     prompt=prompt_, marker=marker, options=options_)
                report["media"]["tokens_before"] = media_starts[key]
                handover = min(handover, media_starts[key])
            if handover < minimum:
                # the target renders this conversation short (e.g. its template drops earlier reasoning):
                # a native prefill is cheap and exact, so there is nothing worth translating
                return effective_tail, handover, ("short", f"{handover} tokens to hand over (target prompt of "
                                                           f"{len(tokens_)} tokens)")
            if shared is not None and rendering[2] is None and shared < handover:
                # the engine renders with the template's defaults, the request renders otherwise: only
                # the tokens both share can be reused (a classic Qwen3 template's enable_thinking changes
                # only the generation prompt, after every message; Qwen3.8's reasoning effort sits in
                # its system prompt), so a longer tail stops T0 there (the engine's T0 <= P - tail)
                effective_tail, handover = len(tokens_) - shared, shared
                if handover < minimum:
                    return effective_tail, handover, (
                        "template-options", f"the request's {', '.join(sorted(template_options))} change {target}'s "
                        f"rendering from token {shared} on, and the engine renders with the template's defaults")
            if window is not None:
                # the draft attends over a sliding window of target features: with a native tail
                # of window - 1 tokens its whole visible window is native at the first draft
                effective_tail = max(effective_tail, window - 1 + markers)
                handover = min(handover, len(tokens_) - effective_tail)
                if handover < minimum:
                    return effective_tail, handover, ("short", f"{handover} tokens to hand over with the "
                                                               f"{effective_tail}-token native tail the DFlash draft "
                                                               "needs")
            return effective_tail, handover, None

        translating = request if request[2] is not None else default
        effective_tail, handover, why = plan(translating)
        report["timings"]["target_tokenize_s"] = round(time.perf_counter() - t, 3)
        if why:
            return skip(*why)
        # a DSA target reads its indexer keys once a query sees more than top_k cells
        rule = capabilities.indexer(tgt_cfg["hub_name"], tgt_cfg["gguf"], registry.hub_indexer_ridges(config),
                                    len(request_tokens))
        if rule:
            return skip(*rule)

        # 2. the target's state layout and draft model
        t = time.perf_counter()
        template, template_resume, speculative = layout_template(B, tgt_cfg, bridge_dir)
        report["timings"]["template_s"] = round(time.perf_counter() - t, 3)
        report["resume_sidecar"] = template_resume is not None
        kinds = set(speculative.split(","))
        if template_resume is not None:
            # kvpark-patched server: its own save says which draft model it runs
            try:
                template_draft = resume.parse_resume(template_resume.read_bytes()).draft
            except (OSError, resume.StateBlobError) as error:
                # e.g. a save cut short: drop it so the next switch makes a new one
                _drop_template(template, template_resume)
                raise RuntimeError(f"{target}'s cached layout template is unreadable ({error}); "
                                   "it was dropped") from None
            draft = resume.classify_draft(template_draft, n_tokens=_template_tokens(template))
            named = {kind for kind in kinds if kind.startswith("draft-")}
            if draft.kind in DRAFT_TYPES and named and not named & DRAFT_TYPES[draft.kind]:
                # the same state shape, another draft (a one-layer EAGLE3 or plain draft model)
                described = "; ".join(section.describe() for section in draft.blob.sections)
                draft = resume.DraftLayout("other", f"{speculative} draft whose state the bridge cannot "
                                           f"synthesize: {described}", draft.blob)
        else:
            # upstream servers never restore draft state; the draft is rebuilt from the native tail
            kind = next((kind for kind, types in DRAFT_TYPES.items() if kinds & types), "none")
            draft = resume.DraftLayout(kind, f"speculative types {speculative}; the server keeps no draft state")
        report["draft"] = {"kind": draft.kind, "note": draft.note, "speculative": speculative}
        # architecture rules first (a DeepSeek V4 target restores without a draft only; an upstream
        # server's layout comes from its speculative types, with no blob)
        rule = capabilities.draft(target, target_arch, draft)
        if rule:
            return skip(*rule)
        if draft.kind == "other":
            return skip("draft-unsupported", f"{target}'s server runs a {draft.note}")
        window = None
        if draft.kind == "dflash":
            window = int(tgt_cfg.get("draft_window") or DRAFT_WINDOW)
            report["draft"]["window"] = window
            effective_tail, handover, why = plan(translating, window)
            if why:
                return skip(*why)

        # 3. the source's cache (saved already when the source made a surrogate)
        save_source()

        # 4. translate
        def run_engine(rendering, effective_tail):
            """The engine's report for a job translating `rendering` (raises EngineError)."""
            _, tokens_, options_ = rendering
            prefix_file, prefix_tokens = None, 0
            # per target: measured to help some targets (rescues a badly filled prompt start) and to
            # hurt others (Muse), so it is opt-in per model (kvpark bridge add --native-prefix)
            use_prefix = prefix if prefix is not None else tgt_cfg.get("native_prefix", False)
            if use_prefix and target_arch in capabilities.TARGET_ONLY:
                # the engine synthesizes a DeepSeek V4 cache whole and refuses a native prefix (an
                # entry registered before add_model refused it, or prefix=True)
                use_prefix = False
                report["native_prefix_tokens"] = 0
            if use_prefix:
                t = time.perf_counter()
                prefix_file, prefix_tokens = native_prefix(B, tgt_cfg, conversation, tokens_, bridge_dir,
                                                           options=options_)
                report["timings"]["native_prefix_s"] = round(time.perf_counter() - t, 3)
                report["native_prefix_tokens"] = prefix_tokens
            output_name = f"kvbridge-{uuid.uuid4().hex}.bin"
            output = Path(tgt_cfg["slot_dir"]) / output_name
            created.extend([output, Path(str(output) + ".resume")])
            target_job = {"name": tgt_cfg["hub_name"], "gguf": tgt_cfg["gguf"], "template_state": str(template),
                          "output": str(output), "tokens": tokens_,
                          "prefix_state": str(prefix_file) if prefix_file else None, "prefix_tokens": prefix_tokens}
            if cap is not None:
                # T0 stops at the start of this message (the first one holding media, or the surrogate
                # of media the source's cache holds)
                target_job["max_t0_message"] = cap
            if tgt_cfg.get("template_kwargs") is not None:
                # the target server's own template flags (kvpark bridge add --reasoning-effort, ...)
                target_job["template_kwargs"] = tgt_cfg["template_kwargs"]
            if tgt_cfg.get("recurrent_mean") and target_arch in capabilities.HYBRID:
                # the hybrid target's recurrent cell at T0 - 1 starts from its mean (kvpark bridge add
                # --recurrent-mean): the engine picks the conversation's group
                target_job["recurrent_mean"] = tgt_cfg["recurrent_mean"]
                if tgt_cfg.get("recurrent_tools"):
                    target_job["recurrent_tools"] = True
            source_job = {"name": src_cfg["hub_name"], "gguf": src_cfg["gguf"], "state": str(source_file)}
            for job_side, cfg, role, server in ((source_job, src_cfg, "source", A), (target_job, tgt_cfg, "target", B)):
                if cfg.get("runtime"):
                    # the server's RoPE settings (kvpark bridge add --rope-*/--yarn-*): its cache layout
                    job_side["runtime"] = cfg["runtime"]
                if build_info(role, server):
                    # the server's llama.cpp build: the engine takes the layouts of the pin it names
                    # (an engine that does not know the field ignores it)
                    job_side["build_info"] = build_info(role, server)
            job_conversation = {"messages": conversation["messages"], "tools": conversation.get("tools")}
            if options_:
                # the engine renders the request as the server does, with its template options
                job_conversation["template_options"] = options_
            job = {"hub": config["hub"], "tail": effective_tail, "fill": fill or config.get("fill", "mean"),
                   "cache_dir": str(bridge_dir / "profiles"), "conversation": job_conversation,
                   "source": source_job, "target": target_job}
            if os.environ.get("KVPARK_BRIDGE_KEEP_JOB"):
                _keep_job(os.environ["KVPARK_BRIDGE_KEEP_JOB"], job, (source_file, template))
            t = time.perf_counter()
            try:
                return engine_client.run_job(config, job, bridge_dir=bridge_dir, timeout=engine_timeout,
                                             log=log), output_name
            finally:
                report["timings"]["engine_s"] = round(report["timings"].get("engine_s", 0)
                                                      + time.perf_counter() - t, 3)

        try:
            try:
                engine, output_name = run_engine(translating, effective_tail)
            except engine_client.EngineError as error:
                if error.kind != engine_client.TEMPLATE or translating[2] is None:
                    raise
                # the engine cannot reproduce the server's rendering of the request (the server runs
                # template settings it cannot see): translate the default rendering instead, up to the
                # tokens it shares with the request's
                report["template_options"].update(engine=False, engine_error=str(error)[:300])
                log({"step": "template-fallback", "error": str(error)[:300]})
                translating = default
                effective_tail, handover, why = plan(translating, window)
                if why:
                    return skip(*why)
                engine, output_name = run_engine(translating, effective_tail)
        except engine_client.EngineError as error:
            detail = str(error).replace("bridge engine failed: ", "", 1)
            if error.kind == "memory":
                # the daemon's memory guard refused the job: a native prefill needs no extra memory
                report["engine_mode"] = "daemon"
                return skip("engine-memory", f"the engine's memory guard refused the translation ({detail})")
            if error.kind == engine_client.TEMPLATE:
                return skip("template-mismatch", f"the engine's rendering of the conversation does not match "
                            f"{target}'s server ({detail})")
            if INDEXER_REFUSAL in detail:
                return skip("no-indexer-keys", detail)
            raise
        except RuntimeError as error:
            # the one-shot engine's refusal (capabilities.indexer decides it before the job, from the
            # GGUF; the engine is the final word)
            if INDEXER_REFUSAL in str(error):
                return skip("no-indexer-keys", str(error).replace("bridge engine failed: ", "", 1))
            raise
        report["engine"] = engine
        report["engine_mode"] = engine.get("engine_mode")
        if tgt_cfg.get("recurrent_mean") and target_arch in capabilities.HYBRID:
            # what the hybrid target's recurrent cell started from (an engine older than the
            # "recurrent-mean" feature ignores the mean and writes zeros, as it always did)
            report["recurrent"] = engine.get("recurrent") or {"strategy": "zero",
                                                              "note": "the engine takes no recurrent mean"}
        if engine.get("daemon_error"):
            # the daemon was unavailable and the one-shot engine ran instead
            report["daemon_error"] = engine["daemon_error"]
        log({"step": "translated", "t0": engine["t0"], "seconds": engine["engine_seconds"],
             "engine_mode": engine.get("engine_mode")})
        output_resume = Path(tgt_cfg["slot_dir"]) / (output_name + ".resume")
        if template_resume is not None:
            # a patched server restores the draft state with the slot: synthesize it for [0, T0)
            data, _ = resume.translated_resume(template_resume.read_bytes(), engine["t0"],
                                               n_tokens=_template_tokens(template))
            resume.write_private(output_resume, data)
            report["draft"]["blob_bytes"] = len(resume.parse_resume(data).draft)

        # 5. restore into the target (the erase also clears the draft model's cache)
        t = time.perf_counter()
        B.slot_action("erase")
        try:
            reply, _ = B.slot_action("restore", output_name)
        except Exception:
            B.slot_action("erase")
            # a template from before a server restart with other settings (flash attention,
            # draft model) no longer matches; the next switch makes a new one
            _drop_template(template, template_resume)
            raise
        report["timings"]["restore_s"] = round(time.perf_counter() - t, 3)
        report["restored_tokens"] = reply.get("n_restored")
        if reply.get("n_restored") != engine["t0"]:
            raise RuntimeError(f"the target restored {reply.get('n_restored')} tokens, expected {engine['t0']}")

        # 6. warm: prefill the native tail now and measure the reuse (the token list would carry
        # the media marker as text, so never with media; the real request prefills it). The
        # request's own rendering: what the real request will send
        if warm and not media:
            reply, seconds = B.complete([token for token, _ in request_tokens], n_predict=1)
            timings = reply.get("timings", {})
            report["warm"] = {"cache_n": timings.get("cache_n"), "prompt_n": timings.get("prompt_n"),
                              "prompt_ms": timings.get("prompt_ms"), "wall_s": round(seconds, 3)}
            report["reused_tokens"] = timings.get("cache_n")
            report["reuse_ok"] = timings.get("cache_n") == engine["t0"]
        report["timings"]["total_s"] = round(time.perf_counter() - started, 3)
        return report
    finally:
        for path in created:
            Path(path).unlink(missing_ok=True)


def undo(config, target):
    """Drop the translated cache from the target; its next request prefills natively."""
    LlamaServer(registry.model(config, target)["url"]).slot_action("erase")
