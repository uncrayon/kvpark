"""Connect OpenCode to kvpark: a "kvpark" provider in OpenCode's config that points at the proxy,
and a plugin that tags each conversation so the proxy parks and resumes it. See docs/opencode.md.

Only plain-JSON config files are rewritten. Every change keeps the user's other keys, leaves a
timestamped backup next to the file, and is skipped when nothing would change.
"""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

PROVIDER = "kvpark"
PLUGIN = "kvpark.js"
MARKER = "// kvpark-opencode-plugin"
SCHEMA = "https://opencode.ai/config.json"
NPM = "@ai-sdk/openai-compatible"
OUTPUT_MAX = 32000  # OpenCode never asks for more output tokens than this
# OpenCode's system prompt and tools take about 7,300 tokens and a quarter of the window is kept
# for the answer (output_limit), so a smaller window leaves almost no room for the conversation.
MIN_CONTEXT = 16384
# The suffix of the names setup gives its models: how setup and doctor tell them from models the
# user added under provider.kvpark by hand.
MODEL_MARK = " (kvpark)"
# kvpark's default backend arguments with a context that fits OpenCode: its system prompt and
# tool list alone take about 7,300 tokens.
BACKEND_ARGS = ["--ctx-size", "32768", "--cache-ram", "0", "--ctx-checkpoints", "8",
                "--checkpoint-min-step", "128", "--jinja"]


def global_dir():
    """OpenCode's global config directory: $XDG_CONFIG_HOME/opencode, else ~/.config/opencode (all platforms)."""
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config").expanduser() / "opencode"


def config_dir(project=None):
    """Where kvpark writes: the global config directory, or a project's .opencode directory."""
    return Path(project).expanduser().resolve() / ".opencode" if project else global_dir()


def candidates(directory):
    """The config files OpenCode reads in this directory, lowest precedence first."""
    names = ("config.json", "opencode.json", "opencode.jsonc") if directory == global_dir() else ("opencode.json", "opencode.jsonc")
    return [directory / name for name in names]


def read(path):
    """(document, text) for a plain-JSON config file; (None, text) for JSONC with comments or
    trailing commas, which kvpark does not rewrite; ({}, None) for a missing file."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}, None
    if not text.strip():
        return {}, text
    try:
        document = json.loads(text)
    except ValueError:
        return None, text
    if not isinstance(document, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return document, text


def target_file(directory):
    """The file setup edits: an existing plain-JSON opencode.json or opencode.jsonc, else a new
    opencode.json (OpenCode merges it with a commented opencode.jsonc next to it)."""
    for path in (directory / "opencode.json", directory / "opencode.jsonc"):
        if path.exists() and read(path)[0] is not None:
            return path
    path = directory / "opencode.json"
    if path.exists():
        raise ValueError(f"{path} is not plain JSON (it has comments or a syntax error), so kvpark does not "
                         f"rewrite it. Add provider.{PROVIDER} by hand as docs/opencode.md shows.")
    return path


def back_up(path):
    """Copy path to a timestamped .bak next to it (a symlink's target, next to the target)."""
    path = path.resolve()
    stamp = time.strftime('%Y%m%d-%H%M%S')
    backup = path.with_name(f"{path.name}.kvpark-{stamp}.bak")
    counter = 1
    while backup.exists():
        counter += 1
        backup = path.with_name(f"{path.name}.kvpark-{stamp}-{counter}.bak")
    shutil.copy2(path, backup)
    return backup


def replace(path, data, mode):
    """Atomically replace path with data. The temporary file is private until it is complete:
    a config file can hold other providers' API keys."""
    temporary = path.with_name(path.name + ".kvpark-tmp")
    temporary.unlink(missing_ok=True)  # left by an interrupted run
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write(path, document, original):
    """Write `document` as JSON; returns the backup of the previous content, if there was one."""
    if path.is_symlink():
        path = path.resolve()  # a dotfile manager's link stays a link
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    backup = back_up(path) if original is not None else None
    replace(path, (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode(), mode)
    return backup


def merge(base, ours):
    """`base` with `ours` laid over it, key by key; keys only `base` has are kept."""
    result = dict(base)
    for key, value in ours.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def provider_entry(base_url, model_id, label, context, output):
    return {
        "npm": NPM,
        "name": "kvpark",
        # OpenCode leaves {env:…} empty when the variable is unset, and then sends no key.
        "options": {"baseURL": base_url.rstrip("/") + "/v1", "apiKey": "{env:KVPARK_API_KEY}"},
        "models": {model_id: {"name": label, "tool_call": True, "limit": {"context": context, "output": output}}},
    }


def ours_model(spec):
    """Whether setup wrote this model entry (its name ends in " (kvpark)"); renamed, it is the user's."""
    return isinstance(spec, dict) and isinstance(spec.get("name"), str) and spec["name"].endswith(MODEL_MARK)


def with_provider(document, entry, served=()):
    """(document with kvpark's provider, replaced model IDs). A model setup wrote earlier that the
    proxy no longer serves is replaced by the new one, and so is a default model that names it;
    models the user added are kept. $schema is added only to a new file."""
    providers = document.get("provider", {})
    if not isinstance(providers, dict):
        raise ValueError("the config's provider setting is not an object")
    current = providers.get(PROVIDER) if isinstance(providers.get(PROVIDER), dict) else {}
    [model_id] = entry["models"]
    models = current.get("models") if isinstance(current.get("models"), dict) else {}
    replaced = [name for name, spec in models.items()
                if name != model_id and ours_model(spec) and served and name not in served]
    if replaced:
        current = {**current, "models": {name: spec for name, spec in models.items() if name not in replaced}}
    result = {**({"$schema": SCHEMA} if not document else {}), **document,
              "provider": {**providers, PROVIDER: merge(current, entry)}}
    for key in ("model", "small_model"):
        if result.get(key) in [f"{PROVIDER}/{name}" for name in replaced]:
            result[key] = f"{PROVIDER}/{model_id}"
    return result, replaced


def without_provider(document):
    """The document without kvpark's provider and without default models that name it."""
    result = dict(document)
    providers = result.get("provider")
    if isinstance(providers, dict) and PROVIDER in providers:
        providers = {name: value for name, value in providers.items() if name != PROVIDER}
        if providers:
            result["provider"] = providers
        else:
            del result["provider"]
    removed = [key for key in ("model", "small_model") if str(result.get(key, "")).startswith(PROVIDER + "/")]
    for key in removed:
        result.pop(key)
    return result, removed


def plugin_source():
    return Path(__file__).with_name(PLUGIN)


def plugin_path(directory):
    return directory / "plugins" / PLUGIN


def ours(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.readline().startswith(MARKER)
    except (OSError, UnicodeDecodeError):
        return False


def check_plugin(directory):
    path = plugin_path(directory)
    if path.exists() and not ours(path):
        raise ValueError(f"{path} exists and is not kvpark's plugin; move it away and run the setup again")


def install_plugin(directory):
    """Copy the plugin into the directory's plugins/; returns True when the file changed."""
    check_plugin(directory)
    path = plugin_path(directory)
    source = plugin_source().read_bytes()
    if path.exists() and path.read_bytes() == source:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    replace(path, source, 0o644)
    return True


def plugin_version(path):
    """The plugin's first line, which names its version: "// kvpark-opencode-plugin 1"."""
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.readline().strip()
    except (OSError, UnicodeDecodeError):
        return None


def ctx_size(args):
    """The --ctx-size (-c) value in a llama-server argument list, or None."""
    for index, arg in enumerate(args):
        value = arg.split("=", 1)[1] if arg.startswith("--ctx-size=") else (
            args[index + 1] if arg in ("--ctx-size", "-c") and index + 1 < len(args) else None)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                return None
    return None


def too_small(context):
    return (f"OpenCode needs a context window of at least {MIN_CONTEXT:,} tokens: its system prompt and "
            f"tools take about 7,300 and a quarter of the window is kept for the answer. The model has "
            f"{context:,}. Run it with --ctx-size 32768 or more")


def _get(url, *, timeout=5, key_file=None):
    headers = {}
    if key_file:
        headers["Authorization"] = "Bearer " + Path(key_file).read_text().strip()
    elif os.environ.get("KVPARK_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["KVPARK_API_KEY"]
    with urlopen(Request(url, headers=headers), timeout=timeout) as response:
        return json.load(response)


def served_models(base_url):
    """Model IDs the kvpark proxy at base_url serves (it relays the server's /v1/models)."""
    data = _get(base_url.rstrip("/") + "/v1/models")
    return [item["id"] for item in data.get("data", []) if isinstance(item, dict) and isinstance(item.get("id"), str)]


def server_context(upstream_url, backend):
    """The context window the model server runs with, or None when it does not say."""
    key_file = os.environ.get("KVPARK_UPSTREAM_KEY_FILE")
    try:
        if backend == "llama.cpp":
            props = _get(upstream_url.rstrip("/") + "/props", key_file=key_file)
            value = (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx")
        else:
            data = _get(upstream_url.rstrip("/") + "/v1/models", key_file=key_file).get("data") or [{}]
            value = data[0].get("max_model_len")
    except (OSError, URLError, ValueError, KeyError, AttributeError, IndexError):
        return None
    return value if isinstance(value, int) and value > 0 else None


def output_limit(context):
    return min(OUTPUT_MAX, max(1024, context // 4))


def _connection(url, server, model, backend_args):
    """(proxy base URL, served model ID or None, label) for the connection OpenCode will use."""
    if server:
        if not model:
            raise ValueError("--server needs --model /path/to/model.gguf")
        from ..setup import connect_managed
        args = shlex.split(backend_args) if backend_args is not None else BACKEND_ARGS
        size = ctx_size(args)
        if size and size < MIN_CONTEXT:  # 0 means the model's own window, checked after it starts
            raise ValueError(too_small(size) + " in --backend-args.")
        config = connect_managed(server, model, args)
        return config["base_url"], "local", Path(config["model"]).stem
    if url:
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost") or parsed.path.rstrip("/") not in ("", "/v1"):
            raise ValueError("--url must be a local kvpark proxy, such as http://127.0.0.1:8080")
        return f"http://127.0.0.1:{parsed.port or 80}", model, model
    if backend_args is not None:
        raise ValueError("--backend-args applies with --server and --model")
    from ..setup import saved_settings
    from ..hermes_service import Service
    config = saved_settings()
    if not config:
        raise ValueError("Choose the model first: kvpark setup --opencode --server /path/to/llama-server "
                         "--model /path/to/model.gguf (disk park/resume), or kvpark setup for a running "
                         "model server (chat routing only), or --url for a running kvpark proxy such as Hermes'.")
    service = Service(lambda: config, config.update)
    service.ensure()
    managed = config["backend"] == "llama.cpp" and not config["upstream_url"]
    served = "local" if managed else model or config["model"]
    label = Path(config["model"]).stem if managed else config["model"]
    return config["base_url"], served, label


def setup(*, project=None, url=None, server=None, model=None, backend_args=None, context=None):
    """Point OpenCode at kvpark; returns the lines to print."""
    from ..cli import request
    if server and url:
        raise ValueError("use --server and --model to run a model, or --url for a running proxy, not both")
    if project and not Path(project).expanduser().is_dir():
        raise ValueError(f"--project {project} is not a directory")
    directory = config_dir(project)
    path = target_file(directory)  # refuse an unreadable config or a foreign plugin before starting anything
    check_plugin(directory)
    base_url, model_id, label = _connection(url, server, model, backend_args)
    status = request(base_url, "status", timeout=5)
    if not isinstance(status, dict) or status.get("service") != "kvpark":
        raise RuntimeError(f"{base_url} is not a kvpark proxy")
    served = served_models(base_url)
    if model_id is None:
        if len(served) != 1:
            raise ValueError("the proxy serves " + (", ".join(served) or "no models")
                             + "; name the one OpenCode should use with --model")
        model_id = served[0]
    elif served and model_id not in served:
        raise ValueError(f"the proxy does not serve {model_id}; it serves " + ", ".join(served))
    caps = status.get("capabilities") or {}
    upstream = status.get("upstream_url", "")
    detected = server_context(upstream, caps.get("backend", "llama.cpp"))
    if context and detected and context > detected:
        raise ValueError(f"--context {context:,} is larger than the {detected:,} tokens the server runs with; "
                         "OpenCode would send conversations the server cannot hold. Leave --context out.")
    context = context or detected
    if not context:
        raise ValueError(f"{upstream} does not report its context window; pass --context with it (in tokens)")
    if context < MIN_CONTEXT:
        raise ValueError(too_small(context) + (", or pass a larger --context if the server's real window is larger."
                                               if not detected else "; see docs/opencode.md."))
    entry = provider_entry(base_url, model_id, f"{label or model_id}{MODEL_MARK}", context, output_limit(context))
    document, original = read(path)
    updated, replaced = with_provider(document, entry, served)
    changed = updated != document or original is None
    backup = write(path, updated, original) if changed else None
    plugin_changed = install_plugin(directory)
    snapshots = bool(caps.get("disk_snapshots"))
    reported = "" if not detected or context == detected else f" (the server reports {detected:,})"
    lines = [f"OpenCode config: {path} ({'updated' if changed else 'already current'})",
             f"  provider {PROVIDER} → {base_url}/v1, model {PROVIDER}/{model_id}, "
             f"context {context:,} tokens{reported}, output {output_limit(context):,}"]
    if replaced:
        lines.append(f"  replaced {', '.join(replaced)}, which the proxy no longer serves")
    if backup:
        lines.append(f"  backup of the previous file: {backup}")
    lines.append(f"OpenCode plugin: {plugin_path(directory)} ({'installed' if plugin_changed else 'already current'})")
    hidden = hidden_by(filter_dirs(project))
    if hidden:
        lines.append(f"Warning: {hidden}; OpenCode will not list the kvpark provider until you change that.")
    lines.append(f"kvpark proxy: {base_url} → {caps.get('backend', 'llama.cpp')} at {upstream}; "
                 + ("conversations are parked to disk and resumed automatically." if snapshots else
                    "chat routing only: this connection cannot park or resume conversations."))
    lines += [f"Use it: opencode -m {PROVIDER}/{model_id} (or choose it with /models in OpenCode).",
              "Check it: kvpark doctor --opencode" + (f" --project {project}" if project else "")
              + (". After a reboot, run kvpark start before OpenCode." if not url else
                 f". Pass --url {base_url} to kvpark status, save and forget for this proxy.")]
    if not find_opencode():
        lines.append("OpenCode was not found on PATH or in ~/.opencode/bin; install it, then start it as above.")
    return lines


def provider_files(directory):
    """[(path, kvpark provider or None, error)] for the directory's config files that mention kvpark."""
    found = []
    for path in candidates(directory):
        try:
            document, text = read(path)
        except ValueError as exc:
            found.append((path, None, str(exc)))
            continue
        if text is None:
            continue
        if document is None:
            if f'"{PROVIDER}"' in text:
                found.append((path, None, "has comments or trailing commas; kvpark does not rewrite it"))
            continue
        provider = (document.get("provider") or {}).get(PROVIDER) if isinstance(document.get("provider"), dict) else None
        if isinstance(provider, dict):
            found.append((path, provider, None))
    return found


def filter_dirs(project=None):
    """The directories whose config decides which providers OpenCode shows, lowest precedence first."""
    if not project:
        return [global_dir()]
    root = Path(project).expanduser().resolve()
    return [global_dir(), root, root / ".opencode"]


def settings(directories, keys):
    """{key: (file, value)} for top-level keys of the plain-JSON config files in these directories;
    a later file overrides an earlier one, as OpenCode merges them."""
    found = {}
    for directory in directories:
        for path in candidates(directory):
            try:
                document, _ = read(path)
            except ValueError:
                continue
            for key in keys:
                if document and key in document:
                    found[key] = (path, document[key])
    return found


def hidden_by(directories):
    """Why OpenCode hides the kvpark provider (enabled_providers, disabled_providers), or None."""
    found = settings(directories, ("enabled_providers", "disabled_providers"))
    path, disabled = found.get("disabled_providers", (None, []))
    if isinstance(disabled, list) and PROVIDER in disabled:
        return f"{path} lists {PROVIDER} in disabled_providers"
    path, enabled = found.get("enabled_providers", (None, None))
    if isinstance(enabled, list) and PROVIDER not in enabled:
        return f"{path} sets enabled_providers without {PROVIDER}"
    return None


def uninstall(*, project=None, confirm=False):
    """Remove kvpark's provider and plugin from OpenCode's config; a preview unless confirm.
    A file left with nothing but $schema is deleted (it holds no settings); a backup is kept."""
    directory = config_dir(project)
    lines = []
    changes = []
    for path in candidates(directory):
        try:
            document, text = read(path)
        except ValueError as exc:
            lines.append(f"{exc}; kvpark leaves it alone.")
            continue
        if text is None:
            continue
        if document is None:
            if f'"{PROVIDER}"' in text:
                lines.append(f"{path}: has comments, so kvpark leaves it alone; remove provider.{PROVIDER} by hand.")
            continue
        updated, removed = without_provider(document)
        if updated != document:
            empty = not path.is_symlink() and not {key: value for key, value in updated.items() if key != "$schema"}
            changes.append((path, None if empty else updated, text))
            note = f"; also removes {' and '.join(removed)} (it names kvpark)" if removed else ""
            lines.append(f"{path}: " + ("delete (nothing else is in it)" if empty else f"remove provider.{PROVIDER}{note}"))
    plugin = plugin_path(directory)
    remove_plugin = plugin.exists() and ours(plugin)
    if remove_plugin:
        lines.append(f"{plugin}: remove")
    if not changes and not remove_plugin:
        return lines + [f"OpenCode in {directory} has no kvpark provider or plugin that kvpark can remove; nothing changed."]
    if not confirm:
        return ["Preview; nothing changed yet:"] + lines + [
            "Apply with: kvpark uninstall --opencode" + (f" --project {project}" if project else "") + " --confirm",
            "kvpark's own services and saved conversations are unaffected; see docs/uninstall.md for those."]
    done = []
    try:
        for path, updated, text in changes:
            if updated is None:
                backup = back_up(path)
                done.append((path, backup))
                path.unlink()
            else:
                backup = write(path, updated, text)
                done.append((path, backup))
            lines.append(f"  backup: {backup}")
    except BaseException:
        for path, backup in done:  # all files or none
            replace(path.resolve() if path.is_symlink() else path, backup.read_bytes(), backup.stat().st_mode & 0o777)
        raise
    if remove_plugin:
        plugin.unlink()
        try:
            plugin.parent.rmdir()  # only when nothing else is in plugins/
        except OSError:
            pass
    return ["Removed:"] + lines + ["Restart OpenCode to drop the kvpark provider and plugin."]


def context_of(spec):
    limit = spec.get("limit") if isinstance(spec, dict) else None
    return limit.get("context") if isinstance(limit, dict) else None


def find_opencode():
    found = shutil.which("opencode")
    if found:
        return found
    default = Path.home() / ".opencode" / "bin" / "opencode"
    return str(default) if default.is_file() and os.access(default, os.X_OK) else None


def opencode_version(executable):
    try:
        result = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=20,
                                stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = result.stdout.strip().splitlines()
    return lines[-1].strip() if result.returncode == 0 and lines else None


def doctor(*, project=None):
    """(lines, ok): how OpenCode reaches kvpark, without changing anything."""
    from ..cli import request
    ok = True
    lines = []
    executable = find_opencode()
    if executable:
        lines.append(f"OpenCode: {opencode_version(executable) or 'version unknown'} ({executable})")
    else:
        lines.append("OpenCode: not found on PATH or in ~/.opencode/bin")
    directory = config_dir(project)
    found = provider_files(directory)
    usable = [(path, provider) for path, provider, error in found if provider is not None]
    for path, provider, error in found:
        if error:
            lines.append(f"Config: {path} — {error}")
    if not usable:
        lines.append(f"Config: no {PROVIDER} provider in {directory}. Run kvpark setup --opencode"
                     + (f" --project {project}" if project else "") + ".")
        return lines, False
    path, provider = usable[-1]  # OpenCode merges these files; the last one wins
    options = provider.get("options") or {}
    base = str(options.get("baseURL", "")).rstrip("/")
    models = provider.get("models") if isinstance(provider.get("models"), dict) else {}
    lines.append(f"Config: {path} — provider {PROVIDER} → {base or 'no baseURL'}; models: "
                 + (", ".join(name + (f" (context {limit:,})" if isinstance(limit, int) else "")
                              for name, limit in ((name, context_of(spec)) for name, spec in models.items()))
                    or "none"))
    for key, (_, value) in settings(filter_dirs(project), ("model", "small_model")).items():
        if isinstance(value, str) and value.startswith(PROVIDER + "/") and value[len(PROVIDER) + 1:] not in models:
            ok = False
            lines.append(f"Config: {key} is {value}, which provider {PROVIDER} does not define. Run kvpark setup --opencode"
                         + (f" --project {project}" if project else "") + " again, or choose another model.")
    hidden = hidden_by(filter_dirs(project))
    if hidden:
        ok = False
        lines.append(f"Config: {hidden}, so OpenCode does not show the provider.")
    plugin = plugin_path(directory)
    if not plugin.exists():
        ok = False
        lines.append(f"Plugin: missing ({plugin}); conversations cannot be parked. Run kvpark setup --opencode.")
    elif not ours(plugin):
        ok = False
        lines.append(f"Plugin: {plugin} is not kvpark's plugin.")
    elif plugin_version(plugin) != plugin_version(plugin_source()):
        ok = False
        lines.append(f"Plugin: {plugin} is from another kvpark version. Run kvpark setup --opencode to update it.")
    else:
        lines.append(f"Plugin: {plugin} (current)")
    root = base[:-3] if base.endswith("/v1") else base
    try:
        status = request(root, "status", timeout=3)
        if not isinstance(status, dict) or status.get("service") != "kvpark":
            raise RuntimeError("not a kvpark proxy")
    except (OSError, URLError, RuntimeError, ValueError) as exc:
        lines.append(f"Proxy: {root or '?'} does not answer as kvpark ({exc}). Start it with kvpark start, "
                     "or run kvpark setup --opencode again if it moved to another port.")
        return lines, False
    caps = status.get("capabilities") or {}
    upstream = status.get("upstream_url", "")
    lines.append(f"Proxy: kvpark {status.get('version', '?')} at {root} → {caps.get('backend', 'llama.cpp')} at {upstream}; "
                 + ("disk park/resume on" if caps.get("disk_snapshots") else "chat routing only (no park/resume)"))
    if status.get("runtime_error"):
        lines.append(f"Proxy note: {status['runtime_error']}")
    try:
        served = served_models(root)
    except (OSError, URLError, ValueError) as exc:
        served = None
        lines.append(f"Models: the proxy could not list them ({exc})")
    if served is not None:
        missing = [name for name in models if name not in served and ours_model(models[name])]
        added = [name for name in models if name not in served and not ours_model(models[name])]
        if missing:
            ok = False
            lines.append(f"Models: the proxy does not serve {', '.join(missing)}; it serves {', '.join(served) or 'nothing'}. "
                         "Run kvpark setup --opencode again.")
        elif not added:
            lines.append(f"Models: {', '.join(models)} served")
        if added:  # added by hand: OpenCode lists them, and their requests fail
            lines.append(f"Models (warning): the proxy does not serve {', '.join(added)}, which you added; "
                         f"it serves {', '.join(served) or 'nothing'}.")
    completed = [event for event in status.get("recent") or [] if event.get("action") == "completed"]
    if completed and not any(event.get("key") for event in completed):
        lines.append("Requests: none of the proxy's recent requests named a conversation. If OpenCode sent them, "
                     "its plugin is not loaded: restart OpenCode, and check that the plugins directory is readable.")
    context = server_context(upstream, caps.get("backend", "llama.cpp"))
    for name, spec in models.items():
        limit = context_of(spec)
        if context and isinstance(limit, int) and limit > context:
            ok = False
            lines.append(f"Context: {name} is configured for {limit:,} tokens but the server runs {context:,}. "
                         "Run kvpark setup --opencode again.")
    return lines, ok
