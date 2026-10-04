# kvpark bridge (experimental): switch a conversation between models

`kvpark bridge` moves a live conversation's cache from one local model to another, so the
new model continues without re-reading the whole conversation. It is experimental and off
unless you configure it.

kvpark already parks and restores a model's **own** cache. The bridge adds **translation**:
the cache of model A is encoded into a shared, model-neutral latent (the *hub*) and decoded
into model B's cache format. Model B restores that cache for the older part of the
conversation and reads only the most recent ~2,000 tokens itself (the *native tail*).

Design and evidence: [experiments/universal_kv/ARCHITECTURE.md](../experiments/universal_kv/ARCHITECTURE.md).

## What it can and cannot do

- **Output is always well-formed.** llama-server reuses only cells whose tokens match the
  prompt, so the target conditions on exactly the right tokens. Translation only changes the
  *values* of the reused cells; imperfect values change what the model writes, never break it.
- **Quality depends on the conversation.** Plain chat translates well; long documents and
  agent work with large tool outputs lose noticeably more (measurements in
  [experiments/universal_kv/RESULTS.md](../experiments/universal_kv/RESULTS.md)). Use
  `kvpark bridge undo MODEL` to drop a translated cache; the next request then prefills natively.
- **Speed.** On an EVO-X2-class machine (Ryzen AI Max+ 395) with the engine daemon on the
  integrated GPU, a switch of an 11-14K-token conversation took 10-17 s between Qwen3.8-27B and
  Muse-Glimmer-30B and 8 s into Qwen3-4B, against 47-86 s and 15 s of native prefill (every
  ordered pair of the three, on a machine shared with other work). Most of it is the target
  re-reading its native tail (2,048 tokens: 8-11 s on Qwen3.8, 12-16 s on Muse, 4 s on
  Qwen3-4B). The translation itself takes 0.5-1.1 s on the GPU on a quiet machine (up to 3.7 s
  under load; a translation into the dense Qwen3-4B now takes 0.8-1.1 s) and 2-14 s on the CPU. Muse → the
  production Qwen3.8 build, the latest run: 0.9-3.2 s for the switch plus an 8 s tail, against
  41-59 s of native prefill. Staging the switch in Hermes loads the engine ahead (20 s the very
  first time, while the bridge caches the models' metadata). Measurements:
  [RESULTS.md](../experiments/universal_kv/RESULTS.md), sections 6 and 8.
- **Models.** A model can take part only if the hub bundle has an adapter for it. The engine
  translates these llama.cpp architectures (`TRANSLATABLE` in `src/kvpark/bridge/capabilities.py`):
  - dense GQA `qwen3` and sliding-window + global `muse-glimmer` (measured live);
  - the hybrids with recurrent layers, whose recurrent state starts at the handover point from
    zeros or from the model's mean (see [Hybrid targets](#hybrid-targets-the-recurrent-state)) and
    is carried on by the native tail: `qwen35` (Qwen3.5/3.8 and Ornith, measured live),
    `qwen35moe`, `qwen3next`, `qwen4exp`, `minimax-01`, `jamba`, `falcon-h1`, `granitehybrid`,
    `nemotron_h`, `nemotron_h_moe`, `plamo2`, `lfm2`, `lfm2moe`;
  - the MLA family, K-only caches: `deepseek2` (DeepSeek V2/V3, Kimi K2.x), `mistral4`, `hy_v4`,
    the DSA models with indexer keys `deepseek32`, `glm-dsa`, `dots3note`, and the K-only hybrids
    `kimi-linear`, `kimi-k3`, `bailingmoe3`;
  - `deepseek4` (DeepSeek V4) as a target only: see [DeepSeek V4 targets](#deepseek-v4-targets).

  Real models have been measured for `qwen3`, `muse-glimmer` and `qwen35` (Qwen3-4B,
  Muse-Glimmer-30B, Qwen3.8-27B and Ornith-1.5-9B). Every other architecture is verified on tiny random-weight models only (no
  such model is downloaded here): its cache layout, the restore and decoding are checked, the
  translation quality is not. Pure recurrent models (Mamba, Mamba2, RWKV6/7, ARWKV7) keep no
  per-token cache and are always prefilled natively. Other architectures (Gemma 3, gpt-oss,
  MiMo, MiniMax-M3, ...) are not supported yet. A switch the translator cannot handle is
  skipped, never forced: see [Skipped switches](#skipped-switches).
- **Servers.** Each model needs its own llama-server with `--slot-save-path` and one slot
  (`--parallel 1`). Upstream llama-server and kvpark's patched server (with its `.resume`
  sidecars) both work as targets, with or without a DFlash or MTP draft model; see
  [Draft models](#draft-models-dflash-mtp). The patched path is verified on the CPU build in
  `runtime/` with small test models and live on the production Qwen3.8 build (ROCm, DFlash2
  draft, mmproj; and the same build with MTP), in and out of it.
  The target's resident conversation is replaced by the switch; park it first if you need it.
- **Images and audio.** The translated cache stops before the first message that holds an
  image or audio; the target reads that message and everything after it itself. See
  [Images and audio](#images-and-audio).

## Set up

1. Run each model's llama-server with a slot directory, for example:

   ```bash
   llama-server --model /path/to/Muse-Glimmer-30B-Q8_0.gguf --port 18083 --parallel 1 --slot-save-path ~/.local/share/kvpark/slots/muse --jinja
   ```

2. Register the models with the bridge (names must match the hub's model names, or pass
   `--hub-name`):

   ```bash
   kvpark bridge add muse --url http://127.0.0.1:18083 --gguf /path/to/Muse-Glimmer-30B-Q8_0.gguf --slot-dir ~/.local/share/kvpark/slots/muse
   ```

   Per-target settings, from the measurements:

   | Setting | Use it for | Why |
   | --- | --- | --- |
   | `--native-prefix` | Qwen-family targets (Qwen3, Qwen3.5/3.8, Ornith) | they need their own values for the first tokens of the prompt (the attention sink); Muse does better without; refused for DeepSeek V4 |
   | `--recurrent-mean NPZ` | hybrid targets (Qwen3.5/3.8, Ornith, ...) | the translated state's recurrent cell starts from the model's mean of native cells instead of zeros: see [Hybrid targets](#hybrid-targets-the-recurrent-state) |
   | `--recurrent-tools` | with `--recurrent-mean`, when the file's `tools` group was calibrated on your own agent prompts | conversations with tools then use that group instead of the pooled mean |
   | `--min-handover N` | small models (e.g. 4B) | their native prefill is already fast; translate only very long prompts |
   | `--tail N` | targets that need more native context | a longer native tail is slower but closer to native quality |
   | `--draft-window N` | targets whose server runs a DFlash draft with a sliding window other than 2048 | the native tail is kept at least `N - 1` tokens long, so the draft sees only native context |

   `kvpark bridge add` records the model's GGUF architecture (`arch` in `kvpark bridge status`);
   entries registered before that get it on their next switch.

   A model whose server runs with RoPE flags that change its cache (`--rope-scaling`,
   `--rope-scale`, `--rope-freq-base`, `--rope-freq-scale`, `--yarn-orig-ctx`,
   `--yarn-ext-factor`, `--yarn-attn-factor`, `--yarn-beta-slow`, `--yarn-beta-fast`) is
   registered with the same flags and values; every engine job carries them, so the engine
   reads and writes that model's cache with them. The state codec models these settings for
   the MLA family and the generic hybrids only; for today's translated architectures (Qwen3,
   Qwen3.5/3.8, Muse) it takes RoPE from the GGUF, so `kvpark bridge add` refuses the flags
   for them (a switch would save the source's cache and then fail in the engine).

   Likewise a model whose server runs with chat template flags (`--chat-template-kwargs`,
   `--reasoning on|off`, `--reasoning-effort`, `--reasoning-preserve` or
   `--no-reasoning-preserve`) is registered with the same flags and values: as a target, the
   engine then renders the conversation as that server does (see
   [Request template settings](#request-template-settings)). Without them a switch into such a
   server usually ends in `template-mismatch`. Flags that keep the template's own default (the
   production Qwen3.8's `--reasoning-effort xhigh --reasoning-preserve`) need not be given.

3. Point the bridge at the translation engine and a hub bundle (a directory written by
   `experiments/universal_kv/fit_hub.py`, ideally converted once to format 2 with
   `experiments/universal_kv/convert_hub.py HUB HUB-v2`, which loads a model pair in a
   fraction of a second instead of several seconds):

   ```bash
   kvpark bridge engine --python ~/venvs/rocm/bin/python --script experiments/universal_kv/engine.py --hub /path/to/hub-v2 --device auto
   ```

   The engine Python needs numpy with an optimized BLAS (OpenBLAS or MKL, as pip's numpy
   wheels bundle; a distribution numpy on the reference BLAS makes a switch take minutes) and,
   for the GPU, torch with ROCm, CUDA or Apple MPS. `--device gpu` uses torch (float16
   weights with float32 accumulation), `--device cpu` numpy, and `--device auto` (the default)
   the GPU when the engine Python's torch has one. Without torch (for example
   `experiments/linear_transfer/.venv`) the engine runs on the CPU. Other options:
   `--mode daemon|oneshot`, `--idle-unload S`, `--idle-exit S`, `--threads N`; see
   [The engine](#the-engine). Settings not given keep their values, so
   `kvpark bridge engine --device cpu` later changes only the device.

4. Check:

   ```bash
   kvpark bridge status
   ```

   It shows the models, the hub and the engine daemon (`engine_daemon`: running or not,
   device, jobs). `engine_check` reports the engine Python's numpy BLAS and whether it sees
   a GPU (a running daemon answers; otherwise the engine Python is asked directly, which takes
   a few seconds when it imports torch). `warnings` flags an engine Python that cannot be
   asked (the check failed), a reference-BLAS numpy, a GPU asked for but missing, a GPU the
   daemon stopped using after an error, a daemon that cannot run here (no `engine_daemon.py`
   beside the engine script, or a bridge directory whose socket path would be too long:
   every switch then runs the slower one-shot engine), and a format 1 hub.

## Switch

Write the conversation as JSON in the OpenAI chat format (content parts, tool calls and tool
results as in a chat request), ending with the message the new model should answer:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."}, {"role": "user", "content": "now you"}],
 "tools": null}
```

```bash
kvpark bridge switch --from qwen38 --to muse --conversation conversation.json
```

The report shows the tokens restored (`restored_tokens`, = T0), how many prompt tokens the
target reused on the warm request (`reused_tokens`, expected T0), the target's draft model
(`draft`: kind, and for DFlash the window), whether the target writes `.resume` sidecars
(`resume_sidecar`), how the engine ran (`engine_mode`: `daemon` or `oneshot`, with
`daemon_error` when the daemon was unavailable; `engine.backend` and `engine.device`), the
native prompt start (`native_prefix_tokens`), what a hybrid target's recurrent cell started from
(`recurrent`), how the request's template settings were handled (`template_options`, see
[Request template settings](#request-template-settings)), the engine's `features` when the
switch needed to know them (`engine_features`), and the time of each step. Send the
conversation to the target's server as usual afterwards; it reuses the cache. A switch the
bridge does not translate reports `"skipped"` with a reason code and a `"note"`; the target
then simply prefills the conversation.

## The engine

The translation runs in an engine daemon, `experiments/universal_kv/engine_daemon.py`,
started by the first switch (or by staging one in Hermes) with the configured engine Python.
It keeps the hub's model sides and the GPU (or numpy's BLAS threads) warm between switches,
so a switch pays only the arithmetic. For 10-13K translated tokens between Qwen3.8-27B and
Muse-Glimmer-30B that was 0.45-0.9 s (Qwen3.8 → Muse) and 0.55-2.4 s (Muse → Qwen3.8) on the
Radeon 8060S GPU, and 2.1-6.4 s on the CPU, against 13 s or more for a fresh engine process.
Dense models cost more: Qwen3-4B keeps 36 full KV layers (1.3-1.6 GB of cache at 11-13K
tokens), and switches into or out of it took 1.1-3.7 s on the GPU and 4.9-13.6 s on the CPU in
the first live runs, 1.4-2.0 s of it the assemble stage (RoPE at the target positions, the
cache cast and the copy into the output arrays). The GPU backend now rotates keys in float32
with fused multiply-adds (on the 8060S exactly the bits of its earlier float64 rotation; every
rotation is checked against the CPU reference on first use) and touches the fresh output pages
on four threads while the GPU decodes: assemble takes 0.13-0.15 s on the same jobs (0.23-0.36 s
before) and the engine step into Qwen3-4B 0.77-1.10 s live. With another process busy on the GPU every stage slows down (assemble 0.6-2.1 s
measured before this change). A DeepSeek V4 target always runs on the CPU.

- **Where.** One daemon per bridge directory (the directory of the bridge configuration),
  listening on `engine.sock` there (owner-only directory and socket; on Linux every client's
  user id is checked). The job travels over the socket, so the conversation is never written
  to disk. Its log, `engine.log` in the same directory (owner-only, rotated at 8 MB), holds
  timings and sizes only.
- **Status.** `kvpark bridge engine-status` prints the daemon's state: device, resident hub
  sides and their memory, jobs and failures, calibration of its memory guard, idle time,
  and the engine Python's BLAS. `kvpark bridge engine-stop` stops it; the next switch starts
  it again. Changing the engine with `kvpark bridge engine` stops a running daemon, so the
  next switch starts one with the new settings.
- **Idle.** After `--idle-unload` seconds without work (default 120) the daemon drops its hub
  sides and gives the GPU memory back; after `--idle-exit` seconds (default 600) it exits.
  Status requests do not count as work.
- **Memory.** Before each job a memory guard compares the job's estimated peak with the free
  memory (and the GPU's). It first drops hub sides the job does not need, then moves the job
  to the CPU, and finally refuses it: the switch is then skipped (`engine-memory`) and the
  target prefills natively. A refused job is never rerun outside the guard.
- **Fallback.** Where the daemon cannot run (Windows, a bridge directory whose socket path
  would exceed about 100 bytes, `--mode oneshot`, or an engine script without
  `engine_daemon.py` beside it) or when it cannot start or reports a fault of its own, the
  switch runs the engine as a one-shot process instead, within what is left of the engine
  timeout (`engine_mode` in the report; `daemon_error` says why). A job the daemon itself
  rejects fails the switch like a failed one-shot engine (a rendering it cannot reproduce is
  a skip instead, `template-mismatch`), and so does a job the daemon took and then overran
  the timeout on (the daemon is stopped) or died during: rerunning it would wait as long
  again, or repeat what stopped the daemon (running out of memory, say) without the memory
  guard. In Hermes a failed switch means a full prefill.
- **Features.** The daemon's ping and `kvpark bridge engine-status` list what the engine
  handles beyond older releases (`features`: `template-options`, `media-source`,
  `recurrent-mean`); the bridge decides those switches by them (see
  [Skipped switches](#skipped-switches)).
- **Devices.** `--device` (or the environment variable `KVPARK_ENGINE_DEVICE`) picks
  `gpu`, `cpu` or `auto`. A GPU error that looks like a broken device moves the daemon to the
  CPU until it exits (`gpu_fault` in its status); the job itself is redone on the CPU.
- **Threads.** The CPU path uses one BLAS thread per physical core, at most 16
  (`--threads N`, or `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` or `MKL_NUM_THREADS` in the
  environment, overrides it; one set there is copied to the unset ones). The servers
  are idle while a switch translates; on the 16-core Ryzen AI Max+ 395, 16 threads translated
  13K-token switches 24-30% faster than 8 (2.1 s against 2.8 s Qwen3.8 → Muse, 2.8 s
  against 3.9 s Muse → Qwen3.8).
- **Prewarm.** Staging a switch (`/kvpark switch` in Hermes) starts the daemon in the
  background and loads the pair's hub sides. It also runs the switch's products once on
  zero rows of its shapes, because the first product of each shape on a GPU picks and loads
  its kernels. The first engine job after staging took 0.51-0.52 s (Qwen3.8 → Muse) and
  0.76 s (Muse → Qwen3.8), against 0.45-0.47 and 0.72-0.78 s warm; without the warm-up it
  took 0.69-0.71 and 0.95-0.96 s. (The last window of a job has a row count that depends on
  the conversation, so its shape can still be new.) The prewarmed sides stay loaded only for
  `--idle-unload` seconds (default 120) and the daemon for `--idle-exit` (600), while a
  stage lasts 15 minutes: a switch made later than that loads the pair itself (0.2-8 s in
  the live runs, plus a few seconds to start a daemon that has exited). Raise
  `--idle-unload` if you usually stage well before switching; the sides hold memory
  meanwhile (1.2 GB for Qwen3.8 and Muse on the CPU).

## Hybrid targets: the recurrent state

A hybrid model (Qwen3.5/3.8, Ornith, and the other architectures with gated DeltaNet, Mamba,
short-conv, lightning or KDA layers) keeps, next to its attention cache, a recurrent state that
summarizes the whole conversation and has no per-token form, so it cannot be translated. The
translated cache holds one recurrent cell at the handover point T0 - 1, and the target's native
tail carries it on from there. By default that cell is zero, as for a fresh conversation. A
mean of the model's own native cells does much better:

1. Build the mean once per model from training conversations in the format of
   `experiments/linear_transfer/chat.py` (their train split is used; no download; a GPU prefill
   with a Vulkan build of `kvpark-kv-probe`, about 18 minutes for Qwen3.8-27B):

   ```bash
   experiments/linear_transfer/.venv/bin/python experiments/universal_kv/r0_experiments.py calibrate --model /path/to/model.gguf --name qwen38 --corpus train.jsonl --probe /path/to/kvpark-kv-probe --output ~/kvpark-means/qwen38
   ```

   It saves the model's cells every 1,024 tokens and writes `recurrent-mean.npz` with the groups
   `all`, `chat` and `tools` (78 MB per group for Qwen3.8 in float16).
2. Register it: `kvpark bridge add qwen38 ... --recurrent-mean ~/kvpark-means/qwen38/recurrent-mean.npz`.
   The file is checked against the model's GGUF (architecture, layer count, and for `qwen35`,
   `qwen35moe` and `qwen3next` the recurrent layers and row sizes); models without a recurrent
   state refuse it.

Per switch the engine picks the group: `chat` for a conversation without tools; with tools the
pooled `all`, or `tools` when the model was registered with `--recurrent-tools` (the file's
`tools` group was calibrated on this deployment's own agent prompts and tools). A file without
the chosen group falls back to `all`; one that lacks that too, or was moved or damaged since it
was registered, gives zeros with a note, and the switch still happens. The report's `recurrent`
says what the cell started from (`strategy`, `group`, `note`). The engine daemon keeps a loaded
group until its idle unload (loading takes 0.2 s for Qwen3.8). An engine without the
`recurrent-mean` feature ignores the mean and writes zeros; the report then says so.

Measured on Qwen3.8-27B and Ornith-1.5-9B, 12 held-out conversations of 8.5-16.8K tokens
([RESULTS.md, section 7](../experiments/universal_kv/RESULTS.md)):

- With the target's own attention cache, so only the recurrent cell differs, zeros cost Qwen3.8
  +0.053 / +0.039 / +0.029 nats/token on chats at native tails of about 1.4K / 2.4K / 4.4K
  tokens, the matched mean +0.008 / +0.002 / +0.003; on Ornith zeros cost +0.065 / +0.035 /
  +0.021 and the mean -0.006 / -0.005 / -0.005, level with native. The mean's KL to the native
  prefill was lower than zero's on every document at every tail.
- After a real translation (Muse → Qwen3.8) the mean takes 0.03-0.04 nats/token off chats
  (+0.129 / +0.102 / +0.061 → +0.097 / +0.064 / +0.027).
- On agent traces the pooled mean did no better than zeros on Qwen3.8; a `tools` mean
  calibrated on the traces' own prompt and tools cut KL by more than half. Hermes' agent traffic
  has not been calibrated, so it gets the pooled mean.
- Live, Muse → the production Qwen3.8 registered with its mean: 6 of 6 switches reused exactly
  T0, with the `chat` group for chats and `all` for a tool-calling trace.

The native tail is not changed for you. With the mean, chats keep their quality on a shorter
tail (1,024 tokens with the mean beat about 4,400 with zeros), so `--tail 1024` suits a hybrid
target without a DFlash draft that serves chats. Conversations with tools keep it only with
`--recurrent-tools`: with the pooled mean, Qwen3.8's agent traces scored KL 0.069 at 1,024
against zeros' 0.057 at the default 2,048. A DFlash draft keeps its window rule either way, so
the production Qwen3.8 keeps 2,048.

## Skipped switches

The bridge skips a switch it cannot translate well or at all, and the target prefills the
conversation natively, which is always correct. A skipped switch never touches the source's
cache. Most skips are decided before the target is touched either. Two come later:
`draft-unsupported`, and `short` because of the DFlash window. Both are decided after the
bridge has made the target's layout template. On the first switch into a target server, that
template replaces the target's resident conversation, as the switch itself would have. Three
come from the engine, after the source's slot was saved: `engine-memory`, `template-mismatch`,
and `no-indexer-keys` when the engine refuses what the GGUF header let through. When the engine
cannot render the request's template options and the bridge falls back to the default
rendering, that fallback is planned again after the save too, and can end in
`template-options` or `short`.

Skips that depend on the engine (`media-source`, `template-options`) are decided by what the
engine that runs the job advertises: the daemon's `features` (`kvpark bridge engine-status`,
`engine_daemon.py`'s ping), or the `FEATURES` an engine script declares when jobs run one-shot.
An engine from before a feature keeps the skip.

| `skipped` | When |
| --- | --- |
| `short` | the target would translate fewer than `min_handover` tokens (default 1024) after its native tail, or before the first message with media; a native prefill is about as fast |
| `no-per-token-cache` | source or target is a pure recurrent model (Mamba, Mamba2, RWKV6/7, ARWKV7): its whole history is one fixed-size state, with no per-token cache to translate |
| `dsv4-source` | the source is DeepSeek-V4: it keeps per-token latents only for the last 128 tokens and pools the rest |
| `unsupported-arch` | the engine cannot translate this architecture's cache (it translates the architectures listed under [What it can and cannot do](#what-it-can-and-cannot-do)), or the GGUF cannot be read |
| `no-head-decoders` | the target is DeepSeek V4 and the hub lacks any of its three head decoders (`csa`, `hca`, `lid`; `fit_hub.py --heads`): its pooled caches are synthesized from per-token compressor inputs the hub decodes |
| `no-indexer-keys` | the target is a DSA model (`deepseek32`, `glm-dsa`, `dots3note`, `hy_v4` with an indexer; `qwen4exp`'s block indexer) whose prompt is longer than its indexer `top_k`, and the hub has no indexer ridge for it (`fit_hub.py --indexer-ridge`): its indexer would rank the translated cells by keys the engine cannot write. Decided from the GGUF header and the prompt length before the source is saved; the engine's own refusal gives the same skip |
| `target-no-vision`, `target-no-audio`, `target-no-video` | the conversation has that kind of media and the target's server has no encoder for it (no `--mmproj`) |
| `dsv4-media` | the target is DeepSeek V4 and the conversation has media: media after a translated pooled cache has never been restored, so the target prefills it natively |
| `media-source` | only with an engine older than the `media-source` feature: the source has already processed the conversation's media (it has the encoder and the media sits in a message before its last answer). A current engine reads the source's cache up to its first medium instead |
| `text-parts` | a message gives its text as a list of parts and the target's template renders such lists differently from plain text |
| `draft-unsupported` | the target's server runs a draft model whose state the bridge cannot write (anything but DFlash/DSpark, MTP or none; e.g. a separate full-attention draft model, or an EAGLE3 or plain draft whose state looks like MTP's; for a DeepSeek V4 target any draft, DSpark and MTP included) |
| `target-no-media-marker` | the target's server does not report its media marker (`/props`), so media cannot be placed |
| `template-options` | the request carries `chat_template_kwargs` or `reasoning_effort` that change how the target renders it, and the engine cannot render with them (an engine older than the `template-options` feature, or a server whose own template settings the engine cannot reproduce, see below). The engine then translates the template's default rendering, which the target reuses only up to where the two renderings part; here fewer than `min_handover` tokens would be left (Qwen3.8 writes the reasoning effort into its system prompt). Options that change only the end of the prompt (a classic Qwen3 template's `enable_thinking: false`, which adds an empty think block after the generation prompt) do not shorten the translated part |
| `template-mismatch` | the engine's renderings of the conversation do not match the target server's tokens even without request options: the server renders with settings the engine cannot see (`--chat-template-kwargs`, a `--reasoning-effort` its template reads, `--reasoning off` or `--no-reasoning-preserve` the model was not registered with, see `kvpark bridge add`; a `--chat-template` other than the GGUF's) or a template the engine's Jinja renders differently (decided after the source's slot was saved) |
| `engine-memory` | the engine daemon's memory guard refused the translation: not enough free memory (decided after the source's slot was saved, which leaves the source's cache unchanged) |

The rules for architectures live in one table, `src/kvpark/bridge/capabilities.py`.

## Draft models (DFlash, MTP)

A target's server may draft with a speculative model: DFlash/DFlash2 (`-md` with a `dflash`
draft) or MTP (`--spec-type draft-mtp`, the model's own next-token head). The bridge detects
the setup from the target's own one-token template save: on kvpark's patched server from the
draft state in its `.resume`, on an upstream server from `/slots` (`speculative.types`).
`/props` cannot be used; it always reports `none`.

- **Upstream servers** neither save nor restore draft state: after a restore the draft is
  rebuilt from the native tail.
- **kvpark-patched servers** (`runtime/` here, and the production Qwen3.8 DFlash2 build) write
  `<slot>.resume` on every save and refuse a restore without it ("Missing slot resume file").
  With a draft model the file must hold the draft's state. For a translated cache the bridge
  writes `<slot>.resume` next to the translated slot (owner-only, deleted after the restore,
  like every bridge file) with a draft state synthesized in the layout of the template's own
  save, no speculative state and no checkpoints. The template's `.resume` is cached with the
  template; the native-prefix save is only read by the engine, never restored, so it needs none.
- **DFlash** keeps per-position draft keys and values computed from the target's own hidden
  states, every layer on a sliding window (2048 tokens for the local Qwen3.8 and Muse
  drafts). The bridge keeps the native tail at least `window - 1` tokens long, so at the first
  draft the draft's whole window holds only positions the target prefilled itself. Their
  hidden states still depend on the translated cache through the target's attention, so they
  are close to native, not identical. The restored draft state holds two zero cells at
  positions T0 - 2 and T0 - 1: they are never inside a later window, and they keep the
  draft's last position equal to the target's (the production build counts draft positions
  from the draft's own last position). Set `--draft-window N` when a draft uses another
  window. On the production Qwen3.8 build (its DFlash2 draft counts positions from its own
  last one) the 41 KB aligned dummy restored every time, the server logged no position
  warning or error, and over three conversations of 12-15K tokens DFlash2 accepted 37% of
  its drafted tokens in the 256 tokens after a translated restore against 41% after a native
  prefill (per conversation the difference went both ways; the answers themselves differ).
  The dummy itself costs nothing: with the target's own cache of the same tokens restored the
  same way, DFlash2 accepted 43.4% with the dummy and 43.7% with the server's real draft
  state.
- **MTP** keeps a full-attention cache for its head. The bridge restores it empty (zero cells
  would be attended forever and dilute it): the head then sees only the native tail and what
  follows, so expect lower acceptance than natively for the rest of the conversation. Output
  is unaffected. Measured on the production Qwen3.8 build with its MTP head (three
  conversations, 256 tokens each): 48% of drafted tokens accepted after a translated restore
  against 61% after a native prefill, and 14.7-16.3 against 15.4-20.5 generated tokens/s.
  The empty state accounts for about 5 points of that: the target's own cache restored the
  same way accepted 55% with the empty marker and 60% with the server's real MTP state,
  generating the same text; the rest goes with the translated cache and the text it leads to.
  An EAGLE3 or plain draft model whose state also has one layer is told apart by `/slots`
  (`draft-eagle3`, `draft-simple`) and skipped.
- **DeepSeek V4 targets** restore a translated slot only without a draft: its DSpark and MTP
  drafts keep their own window of target features, which has not been restored around a
  translated slot, so both are skipped (also on upstream servers, told by `/slots`).
- The bridge always erases the target slot before restoring; that also clears the draft's
  cache.

The template is cached per weights file, server build, slot count and speculative setup. A
template from before a server restart with other settings (flash attention, draft model)
makes the restore fail; the bridge then drops it, and the next switch makes a new one. A
cached template whose `.resume` cannot be read (a save cut short) fails the switch and is
dropped the same way.

## Images and audio

A conversation with images (`image_url` parts), audio (`input_audio`) or video
(`input_video`) can be switched to a target that has that modality (`/props` modalities; a
server started with a matching `--mmproj`):

- The bridge renders each media part as the target's media marker exactly as llama-server
  renders the real request (text parts of a media message joined by newlines, none next to a
  marker), so the target's tokens up to the media match the real request's.
- The translated cache stops at the start of the first message with media (`max_t0_message`
  in the engine job); the target prefills that message, the media in it and everything after
  natively. With media only in the newest message nothing is lost; media in an early message
  shrinks the translated part to the text before it (and the switch may then be `short`).
- When the source's own server has encoded the media (a vision source that answered after the
  image), its cache holds the media's cells. The engine reads that cache up to its first medium
  (`source_media` in the engine report): the text before it is aligned and translated as usual,
  and the media cells and everything after them are left out (they lie after T0 anyway). T0
  also stops after the last message found in that text (`source_media.t0_cap`), which matters
  when the request no longer holds the source's media (a client that replaced an earlier image
  by text): the rest is prefilled natively instead of translated from nothing. An engine older
  than this skips such switches (`media-source`). Verified on tiny fixtures and on real
  llama-server slot saves holding an image (a tiny dense model with NORMAL image positions, and
  a tiny Qwen3.5 hybrid with Qwen3.8's projector and M-RoPE positions): the media slot gives
  the same translated cache, byte for byte, as a text-only save of its prefix. Not yet live
  (the only local vision model is Qwen3.8).
- The media's bytes never reach the bridge's files; the target reads them from the real
  request.
- No warm request is sent (it would read the marker as text); Hermes never sends one anyway.
- The native tail is lengthened by the marker's token count per media part, because the
  bridge counts the marker as text where the server puts the media's own cells.

Measured live from Muse (no vision) to the production Qwen3.8 (with its mmproj) on a 13.5K-token
conversation with a generated chart: with the image in the newest message the real chat
request reused all 11,403 translated tokens and prefilled the rest with the image in 9.4 s
(47 s natively), with the same answer as natively; with the image in an earlier message the
translated part stopped there (7,311 tokens), the request reused exactly those, and the answer
was right, although the model's reasoning first doubted that an image was attached. Over three
such conversations the translated prefix changed the outcome each time (that doubt, a
reasoning that did not reach its answer within 1,200 tokens, a formatting request no longer
followed), while the target's own cache restored the same way answered like the native
prefill: the restore places the image correctly, and what follows a translated prefix depends
on its quality. The other way, Qwen3.8 to Muse with the image, is skipped (`target-no-vision`).

In the local setup audio is transcribed to text before it reaches llama-server, so it switches
like any text.

## Request template settings

The real request's own template settings change what the target processes: llama-server passes
`chat_template_kwargs` and `reasoning_effort` to the chat template. Qwen3.8 writes the reasoning
effort into its system prompt (`xhigh`, the default, adds an instruction paragraph; `medium` adds
none; `low` another), `enable_thinking: false` drops it and closes the think block, and Muse
reads the effort as its "Reasoning strength". Such a request renders differently after its first
3 tokens, so a cache translated from the default rendering would be reused for 3 tokens.

The bridge renders the conversation with the request's settings (`/apply-template` with the
same fields) and, when that differs from the default rendering, hands the settings to the
engine, which renders the target record the same way: it builds the template context
llama-server builds for the request (`chat.server_context` in `experiments/linear_transfer`:
the kwargs as template variables, `reasoning_effort` as the `reasoning_effort` kwarg and
`reasoning_strength`, `"none"` as thinking off, a boolean `preserve_reasoning` as
`preserve_thinking` and its negations) and checks that the target's tokens spell its rendering
byte for byte. The server's own settings (`--reasoning-effort`, `--chat-template-kwargs`,
`--reasoning off`, the date a template prints) are not visible over HTTP, so the engine tries
the template's defaults, then llama-server's defaults (with the template flags the model was
registered with, `kvpark bridge add --reasoning-effort ...`), then those with thinking off,
and keeps the first rendering that matches (`rendering` in the engine report). Where none matches, it
says so (`template` error kind; `engine.py` exits with status 4) and the bridge translates the
default rendering instead, up to the tokens it shares with the request's (`template_options` in
the report: the options, `shared_tokens`, `engine` true when the engine rendered with them).
An engine older than the `template-options` feature gets the default rendering in the first
place. Hermes passes the request's settings; `kvpark bridge switch` reads them from the
conversation file.

What Hermes sends (read from `~/.hermes/config.yaml` and its custom-provider code): a top-level
`reasoning_effort: "medium"` for the main model (`agent.reasoning_effort`; `"none"` when
reasoning is turned off), plus `chat_template_kwargs: {"preserve_reasoning": true}` from a
provider entry's `extra_body`, and `chat_template_kwargs: {"enable_thinking": false}` for its
auxiliary compression calls.

Verified:

- Against llama-server's own `/apply-template` (the pinned CPU build, tiny models carrying the
  real Qwen3.8, Muse and Qwen3-4B templates; Qwen3.8 also with the production flags
  `--reasoning-effort xhigh --reasoning-preserve`): a chat and a tool-calling conversation, each
  with no options, `reasoning_effort` medium, low, high and none, `enable_thinking: false`,
  Hermes' pair, and `preserve_reasoning: false`. Every rendering matched byte for byte, with the
  template's defaults except Muse without a system message (its date line: llama-server's
  context). The same on the production Qwen3.8 build (b10815-85d5703a3), and for servers run
  with template flags the engine is told (Qwen3.8 with `--reasoning-effort low`,
  `--no-reasoning-preserve`, `--reasoning off` or `--chat-template-kwargs
  '{"enable_thinking": false}'`; Muse with `--reasoning-effort xhigh`), which match nothing
  without them. `experiments/universal_kv/test_engine.py`, `LlamaServerTemplateTest`.
- Live, Muse-Glimmer-30B → the production Qwen3.8 build (DFlash2 draft, mmproj, the
  production flags; context 65,536), two held-out chats of 12.1K and 13.6K tokens, the real
  chat request sent with the same settings after the switch: with no options, Hermes'
  `reasoning_effort: "medium"`, Hermes' pair (`medium` and `preserve_reasoning`) and
  `enable_thinking: false` (which renders Qwen3.8's prompt byte for byte as Hermes'
  `reasoning_effort: "none"` does), all 8 switches restored T0 and the request reused exactly T0 tokens
  (cache_n 11,579 / 11,541 / 11,541 / 11,543 and 10,028 / 9,990 / 9,990 / 9,992). Before this,
  the three with options were skipped (the renderings share 3 tokens). The switch took 1.4-4.1 s
  (engine 1.1-1.8 s), then the request prefilled its 2,048-token tail in 17.0-21.0 s, against
  80-102 s of native prefill on a machine shared with other GPU work. The Qwen3.8 log has no
  draft-position warning, position error or failed decode.

## DeepSeek V4 targets

`deepseek4` (DeepSeek V4, V4-Pro, V4-Flash) keeps per-token latents only for its last 128 tokens
and pools everything older into compressed rows (CSA with its lightning indexer, HCA). The
engine synthesizes those rows from per-token compressor inputs that the hub decodes with head
decoders fitted for the model (`experiments/universal_kv/fit_hub.py --heads NAME`):

- Target only: a DeepSeek V4 source is skipped (`dsv4-source`), since its cache cannot be read
  back token by token.
- The hub needs all three head kinds for the target (`csa`, `hca`, `lid`), else
  `no-head-decoders`.
- T0 is rounded down to whole 128-token blocks; `restored_tokens` and the reuse follow the
  engine's T0.
- The server must run without a draft model (`draft-unsupported`).
- Conversations with images or audio are prefilled natively (`dsv4-media`).
- No native prefix: `kvpark bridge add --native-prefix` is refused for it, and an older entry's
  flag is ignored (`native_prefix_tokens` 0).
- The translation runs on the CPU whatever `--device` says. Pooling took about 21 s for 32K
  tokens at V4-Flash's geometry, so a long switch takes tens of seconds.
- Verified on a tiny random-weight model only: a switch restores and reuses the translated cells,
  and a state synthesized from the model's own per-token inputs restores to the native logits
  within 1e-9. No real DeepSeek V4 model has been run (V4-Flash at IQ2 is about 91 GB), so the
  translation quality is unknown. Images with DeepSeek V4 are untested, and V4.1 is not in the
  pinned llama.cpp.

## In Hermes

Register the models and the engine as above (the model kvpark manages is the usual source).
Then, inside the conversation you want to move:

```
/kvpark switch muse
```

(`/kvpark switch muse from qwen38` names the model you are leaving.) Staging starts the engine
daemon in the background and loads the pair, so the switch itself runs warm. Then switch
Hermes to that model with `/model` and send your next message. kvpark's request middleware
sees the conversation's request go to the target's server, translates the cache, restores it
there and lets the request through; the target prefills only the native tail. The stage lasts
15 minutes and is used once. `/kvpark switch status` shows the last result (with how the
engine ran: daemon or one-shot, GPU or CPU, and its seconds, and whether the request's
template settings were rendered by the engine), including a skipped switch and its reason (for
example images going to a model without vision, or a rendering the engine cannot reproduce).
Any failure (an engine error) falls back to a normal full prefill; the request itself is never
changed.

## Add a model

A model joins the hub with its own K/V captured on the hub corpus (public SmolTalk2 chats):

1. Render the corpus with the model's template and capture K/V with `kvpark-kv-probe`
   (`experiments/linear_transfer/chat.py`, `collect_v2.py`; ~15 minutes on the GPU for a
   9B model).
2. Refit the hub with the new model added (`experiments/universal_kv/fit_hub.py`, about 2 hours
   on the CPU for four models), or fit only its adapters against an existing hub (`--base`,
   about 15 minutes). A DSA model also needs an indexer ridge (`--indexer-ridge NAME`) for
   prompts longer than its indexer's `top_k`, a DeepSeek V4 target its head decoders
   (`--heads NAME`). Convert the bundle once with `experiments/universal_kv/convert_hub.py`.
3. For a hybrid model, build its recurrent mean and register it with `--recurrent-mean` (see
   [Hybrid targets](#hybrid-targets-the-recurrent-state)).

The probe, state codec and template checks must support the model's architecture first; see
[experiments/linear_transfer/README.md](../experiments/linear_transfer/README.md) for the state
layouts and [experiments/universal_kv/ARCHITECTURE.md](../experiments/universal_kv/ARCHITECTURE.md#what-the-translator-supports-today)
for what each architecture has been validated on.

## Limitations

- **Quality.** Chat translates well: on held-out chats the measured pairs stay within 0.05
  nats/token of a native prefill. Agent conversations with large tool outputs and long
  documents do not: agent traces lost about 0.3 nats/token into Muse, and after a switch into
  Qwen3.8 the model's next-token distribution stayed far from native (KL 0.4-0.5). For such
  conversations a native prefill (`kvpark bridge undo MODEL`, or simply not switching) is the
  safer choice.
- **Sources.** A small model's cache helps a larger model little: of the gap between an
  information-free cache and a native prefill, Qwen3-4B → Qwen3.8 closed 15% and Qwen3-4B → Muse
  nothing (Qwen3.8 → Muse: 56%). Prefer the larger model as the source.
- **Validated models.** Four real models have been measured (Qwen3.8-27B, Ornith-1.5-9B,
  Qwen3-4B, Muse-Glimmer-30B). Every other architecture is verified on random-weight models
  only.
- **The native tail** dominates a switch (8-16 s on the 27-30B models). A shorter tail needs
  better translation; on hybrids, the recurrent mean allows one for chats.
- **Images.** The translated part stops before the first image, so an image early in the
  conversation leaves little to translate, and in 3 of 3 measured conversations the translated
  text in front of an image changed the answer.
- **Drafts.** MTP drafting accepts less after a switch (48% against 61%) for the rest of the
  conversation. Draft models other than DFlash/DSpark and MTP, and any draft on a DeepSeek V4
  target, are skipped.
- **Server settings.** A target server run with chat template flags must be registered with the
  same flags, or switches into it are skipped (`template-mismatch`).
- **Platforms.** The engine daemon runs on Linux and macOS (macOS untested); Windows runs the
  slower one-shot engine. The GPU path is tested on ROCm (Radeon 8060S) only; CUDA and Apple MPS
  are untested.
- **The target's conversation** is replaced by the switch; park it first if you need it.
- **Hermes.** `/kvpark switch` is covered by integration tests with a stand-in engine; a live
  Hermes session has not been run yet.
