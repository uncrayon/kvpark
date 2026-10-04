"""Minimal llama-server client for the bridge: slots, templates, tokens, completions."""

import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ServerError(RuntimeError):
    pass


class LlamaServer:
    def __init__(self, url, *, timeout=1800):
        self.url, self.timeout = url.rstrip("/"), timeout

    def _call(self, method, path, body=None, timeout=None):
        data = None if body is None else json.dumps(body).encode()
        request = Request(self.url + path, data=data, method=method, headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=timeout or self.timeout) as reply:
                payload = json.loads(reply.read() or b"null")
        except HTTPError as error:
            with error:
                detail = error.read()[:300]
            raise ServerError(f"{self.url}{path}: HTTP {error.code}: {detail!r}") from None
        except URLError as error:
            raise ServerError(f"{self.url}{path}: {error.reason}") from None
        return payload, time.perf_counter() - started

    def props(self):
        return self._call("GET", "/props", timeout=10)[0]

    def slots(self):
        return self._call("GET", "/slots", timeout=10)[0]

    def slot_action(self, action, filename=None):
        body = {} if filename is None else {"filename": filename}
        return self._call("POST", f"/slots/0?action={action}", body)

    def apply_template(self, messages, tools=None, *, add_generation_prompt=True, options=None):
        """The prompt the server renders for these messages; options: request fields that change
        the rendering (chat_template_kwargs, reasoning_effort), as a chat request would carry them."""
        body = {**(options or {}), "messages": messages}
        if tools:
            body["tools"] = tools
        if not add_generation_prompt:
            body["add_generation_prompt"] = False
        return self._call("POST", "/apply-template", body)[0]["prompt"]

    def tokenize(self, text, *, add_special=False, parse_special=True):
        reply, _ = self._call("POST", "/tokenize", {"content": text, "add_special": add_special,
                                                     "parse_special": parse_special, "with_pieces": True})
        return [(item["id"], item["piece"]) for item in reply["tokens"]]

    def complete(self, tokens, *, n_predict=1, **extra):
        body = {"prompt": [int(t) for t in tokens], "n_predict": n_predict, "cache_prompt": True, "id_slot": 0,
                "temperature": 0.0, "top_k": 1, "return_tokens": True, **extra}
        return self._call("POST", "/completion", body)

    def chat(self, messages, *, timeout=None, **fields):
        """(reply, seconds) of a /v1/chat/completions request (not streamed) with these fields."""
        return self._call("POST", "/v1/chat/completions", {**fields, "messages": messages}, timeout=timeout)
