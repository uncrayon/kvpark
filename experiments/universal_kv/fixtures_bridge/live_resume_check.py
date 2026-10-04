"""Live check of the bridge's .resume sidecars on a kvpark-patched llama-server (tiny fixtures).

    python live_resume_check.py --fixtures DIR --work DIR [--port 18510] [--server PATH] [--fa on|off]

DIR holds the GGUFs of make_fixtures.py. For each draft setup (none, an M-RoPE DFlash draft
on a Qwen3.5 hybrid, a NEOX DFlash draft on a dense Qwen3, MTP on a Qwen3.5 hybrid) it starts
a throwaway patched server and runs what a bridge switch does, with the product code
(kvpark.bridge.switch.layout_template and kvpark.bridge.resume):

1. the layout template (a one-token save and its .resume) and its draft classification;
2. a native conversation prefilled and saved, then cut to a "translated-style" slot of T0
   tokens: attention cells below T0 kept, the recurrent state reset to zero at T0 - 1 (what
   the translator's R0 writes), the token list cut to T0;
3. the synthesized .resume for T0, an erase, the restore;
4. a completion of the full prompt, which must reuse exactly T0 tokens (timings cache_n) with
   speculative decoding running (draft_n);
5. a save afterwards, whose draft state shows where the draft's cells sit.

Negative checks per setup: a restore without .resume, and (with a draft) one with an empty
draft blob, must both fail.

The "media" case runs the 5120-wide Qwen3.5 fixture with the local Qwen3.8 mmproj (read
only; --mmproj PATH) and checks the bridge's media handling: /apply-template of the bridge's
flattened messages (media parts as the server's media marker) must equal the server's
rendering of the real request with image parts, and after restoring a translated-style cache
that stops at the start of the newest message (which holds an image), the real chat request
must reuse exactly that prefix while the server prefills the image natively.

The "switch-dflash" and "switch-mtp" cases run kvpark.bridge.switch.switch itself between two
patched servers of the same tiny Qwen3.5 model (the source without a draft, the target with a
DFlash or MTP draft), with stub_engine.py standing in for the engine (same model: the exact
translation is the source slot cut to T0). The warm request must reuse exactly T0.

Prints one JSON report; use the experiment venv (numpy, for experiments/linear_transfer/
kvstate.py). Servers are killed by PID when done.
"""

import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/linear_transfer"))

try:
    import psutil  # noqa: F401
except ImportError:      # kvpark.platforms imports it for process checks the bridge code here never uses
    import types
    sys.modules["psutil"] = types.ModuleType("psutil")

import kvstate  # noqa: E402
from kvpark.bridge import messages, registry, resume  # noqa: E402
from kvpark.bridge import switch as bridge_switch  # noqa: E402
from kvpark.bridge.client import LlamaServer, ServerError  # noqa: E402
from kvpark.bridge.switch import layout_template  # noqa: E402

CASES = {
    "none": ("qwen35-target.gguf", []),
    "dflash-mrope": ("qwen35-target.gguf", ["-md", "{fx}/dflash-mrope.gguf", "--spec-draft-n-max", "3"]),
    "dflash-neox": ("qwen3-target.gguf", ["-md", "{fx}/dflash-neox.gguf", "--spec-draft-n-max", "3"]),
    "mtp": ("qwen35-mtp.gguf", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]),
}
PROMPT = list(b"The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
              b"Sphinx of black quartz, judge my vow. How vexingly quick daft zebras jump!")
T0 = 64


def start(server, model, extra, port, slots, log, fa="on"):
    # flash attention decides the V layout of every cache: off stores V transposed (v_trans 1)
    command = [str(server), "-m", str(model), *extra, "--host", "127.0.0.1", "--port", str(port), "-c", "512",
               "-np", "1", "-fa", fa, "--slot-save-path", str(slots), "-t", "4", "--no-warmup"]
    process = subprocess.Popen(command, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    client = LlamaServer(f"http://127.0.0.1:{port}", timeout=60)
    for _ in range(120):
        try:
            client.props()
            return process, client
        except ServerError:
            if process.poll() is not None:
                raise RuntimeError(f"server exited: {Path(log).read_text()[-2000:]}")
            time.sleep(0.25)
    process.kill()
    raise RuntimeError("server did not start")


def translated_style(native, spec, t0, out):
    """The native save cut to t0 tokens as the translator lays it out (R0 recurrent reset)."""
    state = kvstate.read_file(native, spec)
    parts = []
    for part in state.state.parts:
        if isinstance(part, kvstate.KVCache):
            streams = []
            for s in part.streams:
                if s is None:
                    streams.append(None)
                    continue
                keep = np.asarray(s.positions) < t0
                streams.append(kvstate.KVStream(
                    np.asarray(s.positions)[keep], tuple(i for i, k in zip(s.seq_ids, keep) if k),
                    None if s.ext is None else np.asarray(s.ext)[keep], s.v_trans,
                    {il: np.asarray(a)[keep] for il, a in s.k.items()},
                    {il: np.asarray(a)[keep] for il, a in s.v.items()}, s.k_types, s.v_types))
            parts.append(kvstate.KVCache(part.name, part.layers, streams))
        else:
            zero = {key: {il: kvstate.OpaqueRows(rows.ggml_type, rows.row_bytes, bytes(rows.row_bytes))
                          for il, rows in getattr(part, key).items()} for key in ("r", "s")}
            parts.append(kvstate.RecurrentState(part.name, part.layers, np.asarray([t0 - 1], dtype=np.int32), ((),),
                                                part.n_layer, zero["r"], zero["s"]))
    words = kvstate.token_words(np.asarray(state.tokens)[:t0], server_packed=state.server_packed)
    kvstate.write_file(out, kvstate.StateFile(words, kvstate.SequenceState(spec, state.state.flags, parts)))


def draft_positions(resume_path, n_tokens):
    """Where the draft's cells sit, per section, in a save's .resume."""
    draft = resume.parse_resume(Path(resume_path).read_bytes()).draft
    if not draft:
        return []
    out = []
    for section in resume.parse_state_blob(draft, n_tokens=n_tokens).sections:
        streams = section.streams if isinstance(section, resume.KVSection) else [section]
        positions = sorted(p for s in streams if s is not None for p in s.positions)
        out.append({"cells": len(positions), "min": positions[0] if positions else None,
                    "max": positions[-1] if positions else None,
                    "layers": None if not isinstance(section, resume.KVSection) or section.layers is None
                    else len(section.layers)})
    return out


def expect_failure(client, name):
    try:
        client.slot_action("restore", name)
    except ServerError as error:
        return str(error)[-160:]
    return None


def run_case(name, args, port):
    model_name, extra = CASES[name]
    fx, work = Path(args.fixtures), Path(args.work) / f"{name}-fa-{args.fa}"
    slots = work / "slots"
    slots.mkdir(parents=True, exist_ok=True)
    for stale in slots.iterdir():
        stale.unlink()
    model = fx / model_name
    extra = [e.format(fx=fx) for e in extra]
    process, client = start(args.server, model, extra, port, slots, work / "server.log", args.fa)
    report = {"case": name, "model": model_name, "args": extra, "flash_attn": args.fa}
    try:
        bridge_dir = work / "bridge"
        bridge_dir.mkdir(exist_ok=True)
        template, template_resume, speculative = layout_template(client, {"gguf": str(model), "slot_dir": str(slots)},
                                                                 bridge_dir)
        report["speculative"] = speculative
        n_template = resume.slot_token_count(template)
        layout = resume.classify_draft(resume.parse_resume(template_resume.read_bytes()).draft, n_tokens=n_template)
        report["template"] = {"tokens": n_template, "resume_bytes": template_resume.stat().st_size,
                              "draft_kind": layout.kind, "draft_note": layout.note}

        # a native conversation turn, saved
        spec = kvstate.ModelSpec.from_gguf(str(model))
        client.slot_action("erase")
        first, _ = client.complete(PROMPT, n_predict=8)
        client.slot_action("save", "native.bin")
        native_tokens = resume.slot_token_count(slots / "native.bin")
        full = list(PROMPT) + [int(t) for t in first["tokens"][:7]] + list(b" And then? Tell me more.")
        report["native"] = {"tokens": native_tokens, "draft": draft_positions(slots / "native.bin.resume", native_tokens)}

        # the translated-style slot of T0 tokens and its synthesized .resume
        translated_style(slots / "native.bin", spec, T0, slots / "translated.bin")
        data, kind = resume.translated_resume(template_resume.read_bytes(), T0, n_tokens=n_template)
        report["synthesized"] = {"kind": kind.kind, "draft_bytes": len(resume.parse_resume(data).draft)}

        # negative checks: no .resume, then an empty draft blob
        client.slot_action("erase")
        report["restore_without_resume"] = expect_failure(client, "translated.bin")
        if layout.kind != "none":
            resume.write_private(slots / "translated.bin.resume", resume.Resume().to_bytes())
            client.slot_action("erase")
            report["restore_with_empty_draft_blob"] = expect_failure(client, "translated.bin")
            (slots / "translated.bin.resume").unlink()

        # the bridge's restore
        resume.write_private(slots / "translated.bin.resume", data)
        client.slot_action("erase")
        restored, _ = client.slot_action("restore", "translated.bin")
        report["restored_tokens"] = restored.get("n_restored")
        reply, _ = client.complete(full, n_predict=16)
        timings = reply["timings"]
        report["after_restore"] = {key: timings.get(key) for key in ("cache_n", "prompt_n", "predicted_n", "draft_n",
                                                                     "draft_n_accepted")}
        report["reuse_ok"] = timings.get("cache_n") == T0
        client.slot_action("save", "after.bin")
        after_tokens = resume.slot_token_count(slots / "after.bin")
        report["after_restore"]["tokens"] = after_tokens
        report["after_restore"]["draft"] = draft_positions(slots / "after.bin.resume", after_tokens)

        # the same request natively, for comparison
        client.slot_action("erase")
        reply, _ = client.complete(full, n_predict=16)
        report["native_same_request"] = {key: reply["timings"].get(key) for key in
                                         ("cache_n", "prompt_n", "predicted_n", "draft_n", "draft_n_accepted")}
        log = (work / "server.log").read_text(errors="replace")
        report["log_warnings"] = [line.strip()[-200:] for line in log.splitlines()
                                  if "pos_max" in line or "failed to process speculative" in line][:5]
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    return report


MMPROJ = "/srv/models/qwen3.8-27b-q8/mmproj-Qwen3.8-27B-F16.gguf"
IMAGE = ROOT / "runtime/source/tools/mtmd/test-1.jpeg"


def run_media(args, port):
    fx, work = Path(args.fixtures), Path(args.work) / "media"
    slots = work / "slots"
    slots.mkdir(parents=True, exist_ok=True)
    for stale in slots.iterdir():
        stale.unlink()
    model = fx / "qwen35-vision.gguf"
    extra = ["--mmproj", args.mmproj, "--image-max-tokens", "16", "--jinja"]
    process, client = start(args.server, model, extra, port, slots, work / "server.log")
    report = {"case": "media", "model": model.name, "args": extra}
    try:
        props = client.props()
        report["modalities"] = props.get("modalities")
        image = {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(IMAGE.read_bytes()).decode()}}
        history = [{"role": "system", "content": "You are terse."},
                   {"role": "user", "content": "Tell me about caches, " * 12 + "please."}]

        def chat(conversation, n_predict=8):
            reply, _ = client._call("POST", "/v1/chat/completions", {
                "messages": conversation, "max_tokens": n_predict, "temperature": 0, "top_k": 1, "id_slot": 0,
                "cache_prompt": True})
            return reply

        # rendering: real requests vs the bridge's flattened messages
        shapes = {"text, image, text": [{"role": "user", "content": [{"type": "text", "text": "Look:"}, image,
                                                                     {"type": "text", "text": "what is it?"}]}],
                  "texts then image": [{"role": "user", "content": [{"type": "text", "text": "a"},
                                                                    {"type": "text", "text": "b"}, image]}],
                  "two images first": [{"role": "user", "content": [image, image, {"type": "text", "text": "x"}]}],
                  "text parts only": [{"role": "user", "content": [{"type": "text", "text": "find "},
                                                                   {"type": "text", "text": "caches"}]}]}
        rendering = {}
        for name, shape in shapes.items():
            conversation = messages.conversation_from_openai(shape)
            join = bridge_switch._text_join(client, conversation)
            flat = messages.flatten(conversation, marker=props["media_marker"], text_join=join)
            rendering[name] = {"identical": client.apply_template(shape) == client.apply_template(flat["messages"]),
                               "text_join": join}
        report["rendering"] = rendering

        # media in a tool result right after the assistant's tool calls (a screenshot tool): the
        # server refuses to continue that assistant message, so the bridge renders a stand-in
        call = {"id": "c1", "type": "function", "function": {"name": "screenshot", "arguments": "{}"}}
        tool_shape = history + [{"role": "assistant", "content": "", "tool_calls": [call]},
                                {"role": "tool", "tool_call_id": "c1", "content": [image]}]
        flat = messages.flatten(messages.conversation_from_openai(tool_shape), marker=props["media_marker"])
        prompt = client.apply_template(flat["messages"])
        ids = [i for i, _ in client.tokenize(prompt, add_special=True)]
        before_marker = len(client.tokenize(prompt[:prompt.find(props["media_marker"])], add_special=True))
        try:
            client.apply_template(flat["messages"][:3], add_generation_prompt=False)
            continued = "rendered"
        except ServerError as error:
            continued = str(error)[-90:]
        cap = bridge_switch._media_start(client, flat, 3, ids, prompt=prompt, marker=props["media_marker"])
        report["tool_result_media"] = {
            "identical": client.apply_template(tool_shape) == prompt, "conversation_up_to_it": continued,
            "media_start": cap, "tokens_before_marker": before_marker,
            "pieces_before_cap": [piece for _, piece in client.tokenize(prompt, add_special=True)[cap - 6:cap]],
            "ok": 0 < cap <= before_marker}

        # a native first turn, saved
        template, template_resume, _ = layout_template(client, {"gguf": str(model), "slot_dir": str(slots)},
                                                       work / "bridge")
        client.slot_action("erase")
        first = chat(history)
        answer = first["choices"][0]["message"]["content"]
        client.slot_action("save", "native.bin")
        conversation = history + [{"role": "assistant", "content": answer},
                                  {"role": "user", "content": [{"type": "text", "text": "And this picture?"}, image]}]

        # what the bridge computes for this conversation: the cap at the newest message
        bridged = messages.conversation_from_openai(conversation)
        flat = messages.flatten(bridged, marker=props["media_marker"])
        prompt = client.apply_template(flat["messages"])
        ids = [i for i, _ in client.tokenize(prompt, add_special=True)]
        cap = bridge_switch._media_start(client, flat, min(messages.media(bridged).values()), ids, prompt=prompt,
                                         marker=props["media_marker"])
        # the native save stands in for a translation, so T0 also stays within what it holds
        # (it ends before the end-of-turn token the template adds after the stored answer)
        spec = kvstate.ModelSpec.from_gguf(str(model))
        native = [int(t) for t in kvstate.read_file(slots / "native.bin", spec).tokens]
        common = next((i for i, (a, b) in enumerate(zip(native, ids)) if a != b), min(len(native), len(ids)))
        t0 = min(cap, common)
        report.update(media_start_cap=cap, native_common_prefix=common, t0=t0, flat_prompt_tokens=len(ids))
        translated_style(slots / "native.bin", spec, t0, slots / "translated.bin")
        data, _ = resume.translated_resume(template_resume.read_bytes(), t0, n_tokens=resume.slot_token_count(template))
        resume.write_private(slots / "translated.bin.resume", data)
        client.slot_action("erase")
        restored, _ = client.slot_action("restore", "translated.bin")
        report["restored_tokens"] = restored.get("n_restored")
        reply = chat(conversation)
        timings = reply.get("timings", {})
        report["after_restore"] = {key: timings.get(key) for key in ("cache_n", "prompt_n", "predicted_n")}
        report["reuse_ok"] = timings.get("cache_n") == t0
        client.slot_action("save", "after.bin")
        report["after_restore"]["slot_tokens"] = resume.slot_token_count(slots / "after.bin")
        # natively, for comparison
        client.slot_action("erase")
        reply = chat(conversation)
        report["native_same_request"] = {key: reply.get("timings", {}).get(key) for key in ("cache_n", "prompt_n")}
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    return report


SWITCH_TARGETS = {"switch-dflash": CASES["dflash-mrope"][1], "switch-mtp": ["--spec-type", "draft-mtp"]}


def run_switch(name, args, port):
    fx, work = Path(args.fixtures), Path(args.work) / name
    model = fx / ("qwen35-mtp.gguf" if name == "switch-mtp" else "qwen35-target.gguf")
    slots = {side: work / f"slots-{side}" for side in ("a", "b")}
    for path in slots.values():
        path.mkdir(parents=True, exist_ok=True)
        for stale in path.iterdir():
            stale.unlink()
    extra = [e.format(fx=fx) for e in SWITCH_TARGETS[name]]
    processes = []
    report = {"case": name, "target_args": extra}
    try:
        process, A = start(args.server, model, [], port, slots["a"], work / "server-a.log")
        processes.append(process)
        process, B = start(args.server, model, extra, port + 1, slots["b"], work / "server-b.log")
        processes.append(process)
        hub = work / "hub"
        hub.mkdir(exist_ok=True)
        (hub / "manifest.json").write_text(json.dumps({"format_version": 1, "models": {"a": {}, "b": {}}}))
        config_path = work / "bridge" / "bridge.json"
        config = registry.empty()
        registry.add_model(config, "a", url=A.url, gguf=model, slot_dir=slots["a"])
        registry.add_model(config, "b", url=B.url, gguf=model, slot_dir=slots["b"], tail=16, min_handover=8,
                           draft_window=8)
        registry.set_engine(config, python=sys.executable, script=Path(__file__).with_name("stub_engine.py"), hub=hub)
        registry.save(config, config_path)
        history = [{"role": "system", "content": "You are terse."},
                   {"role": "user", "content": "Tell me about caches, " * 8 + "please."}]
        first, _ = A._call("POST", "/v1/chat/completions", {"messages": history, "max_tokens": 12, "temperature": 0,
                                                            "top_k": 1, "id_slot": 0, "cache_prompt": True})
        conversation = messages.conversation_from_openai(
            history + [{"role": "assistant", "content": first["choices"][0]["message"]["content"]},
                       {"role": "user", "content": "And then? Go on, at length."}])
        result = bridge_switch.switch(registry.load(config_path), conversation, "a", "b", config_path=config_path,
                                      log=lambda event: None)
        report["switch"] = {key: result.get(key) for key in ("skipped", "note", "draft", "resume_sidecar",
                                                             "target_prompt_tokens", "restored_tokens",
                                                             "reused_tokens", "reuse_ok", "warm")}
        report["switch"]["t0"] = result.get("engine", {}).get("t0")
        report["switch"]["tail"] = result.get("engine", {}).get("P", 0) - (result.get("engine", {}).get("t0") or 0)
        # the next request continues from the restored cache, with the draft running
        flat = messages.flatten(conversation)
        reply, _ = B._call("POST", "/v1/chat/completions", {"messages": flat["messages"], "max_tokens": 16,
                                                            "temperature": 0, "top_k": 1, "id_slot": 0,
                                                            "cache_prompt": True})
        report["next_request"] = {key: reply.get("timings", {}).get(key)
                                  for key in ("cache_n", "prompt_n", "predicted_n", "draft_n", "draft_n_accepted")}
        report["bridge_files_left"] = sorted(p.name for side in slots.values() for p in side.iterdir())
    finally:
        for process in processes:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--port", type=int, default=18510)
    parser.add_argument("--server", default=str(ROOT / "runtime/build/bin/llama-server"))
    parser.add_argument("--case", action="append", choices=sorted(CASES) + ["media", *SWITCH_TARGETS])
    parser.add_argument("--mmproj", default=MMPROJ, help="a qwen3vl_merger mmproj 5120 wide (media case)")
    parser.add_argument("--fa", choices=("on", "off"), default="on", help="flash attention (off: transposed V)")
    args = parser.parse_args()
    reports, port = [], args.port
    for name in args.case or list(CASES) + ["media", *SWITCH_TARGETS]:
        if name == "media":
            reports.append(run_media(args, port))
        elif name in SWITCH_TARGETS:
            reports.append(run_switch(name, args, port))
            port += 1
        else:
            reports.append(run_case(name, args, port))
        port += 1
    print(json.dumps(reports, indent=1))


if __name__ == "__main__":
    main()
