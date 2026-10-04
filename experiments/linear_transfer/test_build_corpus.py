"""Corpus builder: row conversion, shortening, splits, special-token checks, exclusions, final-message
cuts, long-context selection (end to end on tiny fixtures with KVPARK_KV_PROBE=/path/binary)."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import build_corpus as bc
import chat


PROBE = os.environ.get("KVPARK_KV_PROBE")


MODELS = {"qwen": Path("/srv/models/qwen3.8-27b-q8/Qwen3.8-27B-Q8_0.gguf"),
          "muse": Path("/srv/models/muse-glimmer-30b-q8/Muse-Glimmer-30B-Q8_0.gguf")}


def row(messages, **kwargs):
    base = {"custom_instructions": "", "enable_thinking": True, "python_tools": [], "xml_tools": []}
    base.update(kwargs)
    return {"messages": [{"role": role, "content": content} for role, content in messages],
            "chat_template_kwargs": base, "source": "test"}


HERMES_TOOLS = ("<tools>\n[{'type': 'function', 'function': {'name': 'set_light', 'description': 'Switch a light.', "
                "'parameters': {'type': 'object', 'properties': {'room': {'type': 'string'}}}}}]\n</tools>")
JSONL_TOOLS = ('{"type": "function", "function": {"name": "web_search", "description": "Search.", '
               '"parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}\n'
               '{"type": "function", "function": {"name": "final_answer", "description": "Answer.", '
               '"parameters": {"type": "object", "properties": {"answer": {"type": "string"}}}}}')


class ConversionTests(unittest.TestCase):
    def test_reasoning_is_split_from_content(self):
        conversation = bc.convert_row("s", 3, row([("user", "  Hola, ¿qué tal?  "),
                                                   ("assistant", "<think>\nThe user greets me.\n</think>\n\nBien, gracias.")]))
        system, user, assistant = conversation["messages"]
        self.assertEqual(system, {"role": "system", "content": bc.DEFAULT_SYSTEM})
        self.assertEqual(user["content"], "Hola, ¿qué tal?")
        self.assertEqual(assistant, {"role": "assistant", "reasoning_content": "The user greets me.",
                                     "content": "Bien, gracias."})
        self.assertEqual(conversation["id"], "s:3")

    def test_custom_instructions_become_the_system_prompt(self):
        conversation = bc.convert_row("s", 0, row([("user", "Hi"), ("assistant", "Hello.")],
                                                  custom_instructions=" You are a pirate. "))
        self.assertEqual(conversation["messages"][0]["content"], "You are a pirate.")

    def test_hermes_python_literal_calls_and_tagged_responses(self):
        messages = [("user", "Lights please."),
                    ("assistant", "<tool_call>\n{'arguments': {'room': 'kitchen'}, 'name': 'set_light'}\n</tool_call>\n"
                                  "<tool_call>\n{'arguments': {'room': 'hall'}, 'name': 'set_light'}\n</tool_call>"),
                    ("tool", '<tool_response>\n{"name": "set_light", "content": "ok"}\n</tool_response>\n'
                             '<tool_response>\n{"name": "set_light", "content": "failed"}\n</tool_response>\n'),
                    ("assistant", "The kitchen light is on; the hall light failed.")]
        conversation = bc.convert_row("h", 0, row(messages, xml_tools=[HERMES_TOOLS], enable_thinking=False))
        self.assertEqual(conversation["tools"][0]["function"]["name"], "set_light")
        call_message = conversation["messages"][2]
        self.assertEqual(call_message["content"], "")
        self.assertEqual([c["function"]["arguments"]["room"] for c in call_message["tool_calls"]],
                         ["kitchen", "hall"])
        tools = conversation["messages"][3:5]
        self.assertEqual([t["tool_call_id"] for t in tools], ["call_2_0", "call_2_1"])
        self.assertIn('"failed"', tools[1]["content"])
        self.assertEqual(conversation["messages"][-1]["role"], "assistant")

    def test_smolagents_jsonl_tools_observations_and_final_answer(self):
        messages = [("user", "New task:\nWho wrote it?"),
                    ("assistant", "<think>Search first.</think>\n<tool_call>{'name': 'web_search', 'arguments': {'query': 'author'}}</tool_call>"),
                    ("tool", "Observation:\nResult: Ada wrote it."),
                    ("assistant", "<think>Done.</think>\n<tool_call>{'name': 'final_answer', 'arguments': {'answer': 'Ada'}}</tool_call>")]
        conversation = bc.convert_row("a", 1, row(messages, xml_tools=[JSONL_TOOLS]))
        self.assertEqual([t["function"]["name"] for t in conversation["tools"]], ["web_search", "final_answer"])
        self.assertEqual(conversation["messages"][2]["tool_calls"][0]["function"]["arguments"], {"query": "author"})
        self.assertEqual(conversation["messages"][3]["content"], "Observation:\nResult: Ada wrote it.")
        self.assertEqual(conversation["messages"][-1],
                         {"role": "assistant", "reasoning_content": "Done.", "content": "Ada"})

    def test_embedded_tool_list_and_plain_instruction_entries(self):
        embedded = ("You are provided with function signatures within <tools></tools> XML tags. "
                    "Here are the available tools:" + HERMES_TOOLS + " Use them wisely.")
        tools, notes = bc.parse_tools([embedded])
        self.assertEqual([tool["function"]["name"] for tool in tools], ["set_light"])
        self.assertEqual(notes, "")
        tools, notes = bc.parse_tools(["You are a helpful assistant, with no access to external functions."])
        self.assertIsNone(tools)
        conversation = bc.convert_row("h", 0, row([("user", "Hi"), ("assistant", "Hello.")],
                                                  xml_tools=["You have no tools today."]))
        self.assertEqual(conversation["messages"][0]["content"], "You have no tools today.")

    def test_trailing_unanswered_call_is_dropped(self):
        messages = [("user", "Lights?"), ("assistant", "Sure, one moment."), ("user", "Kitchen."),
                    ("assistant", "<tool_call>{'name': 'set_light', 'arguments': {'room': 'kitchen'}}</tool_call>")]
        conversation = bc.convert_row("h", 0, row(messages, xml_tools=[HERMES_TOOLS]))
        self.assertEqual(conversation["messages"][-1], {"role": "assistant", "content": "Sure, one moment."})

    def test_rejections(self):
        cases = [
            row([("user", "x"), ("assistant", "<tool_call>{'name': 'set_light', 'arguments': {}}</tool_call>"),
                 ("user", "y")], xml_tools=[HERMES_TOOLS]),
            row([("user", "x"), ("assistant", "<tool_call>{'name': 'set_light', 'arguments': {}}</tool_call>"),
                 ("tool", "<tool_response>a</tool_response><tool_response>b</tool_response>")],
                xml_tools=[HERMES_TOOLS]),
            row([("user", "x"), ("assistant", "<tool_call>{broken</tool_call>")], xml_tools=[HERMES_TOOLS]),
            row([("user", "x"), ("assistant", "   ")]),
            row([("user", "x")]),
            row([("system", "s"), ("user", "x"), ("assistant", "y")]),
            row([("user", "x"), ("assistant", "y")], python_tools=["def f(): pass"]),
            row([("user", "x"), ("assistant", "<tool_call>{'name': 'set_light', 'arguments': {}}</tool_call>"),
                 ("tool", "done"), ("assistant", "ok")]),
        ]
        for number, case in enumerate(cases):
            with self.subTest(case=number), self.assertRaises(bc.Reject):
                bc.convert_row("r", number, case)

    def test_cut_prefers_line_boundaries(self):
        text = "line one\nline two\nline three is long"
        self.assertEqual(bc.cut(text, 20), "line one\nline two")
        self.assertEqual(bc.cut("  short  ", 20), "short")
        self.assertEqual(bc.cut("x" * 50, 10), "x" * 10)
        self.assertEqual(bc.cut("  " + "x" * 50 + " ", 0), "x" * 50)

    def test_zero_limits_keep_tool_outputs_and_reasoning_whole(self):
        messages = [("user", "New task:\nWho wrote it?"),
                    ("assistant", "<think>" + "step " * 40 + "</think>\n<tool_call>{'name': 'web_search', "
                                  "'arguments': {'query': 'author'}}</tool_call>"),
                    ("tool", "Observation:\n" + "line\n" * 50),
                    ("assistant", "<think>Done.</think>\n<tool_call>{'name': 'final_answer', "
                                  "'arguments': {'answer': 'Ada'}}</tool_call>")]
        whole = bc.convert_row("a", 1, row(messages, xml_tools=[JSONL_TOOLS]), tool_limit=0, reasoning_limit=0)
        short = bc.convert_row("a", 1, row(messages, xml_tools=[JSONL_TOOLS]), tool_limit=30, reasoning_limit=30)
        self.assertEqual(whole["messages"][2]["reasoning_content"], ("step " * 40).strip())
        self.assertEqual(whole["messages"][3]["content"], ("Observation:\n" + "line\n" * 50).strip())
        self.assertLessEqual(len(short["messages"][2]["reasoning_content"]), 30)
        self.assertLessEqual(len(short["messages"][3]["content"]), 30)


class ShapingTests(unittest.TestCase):
    def conversation(self, turns):
        messages = []
        for number in range(turns):
            messages += [("user", f"Question {number}?"), ("assistant", f"Answer {number}. " * 20)]
        return bc.convert_row("s", 0, row(messages))

    def test_shorten_drops_trailing_turns_but_keeps_an_answer(self):
        conversation = self.conversation(5)
        size = lambda c: sum(len(m["content"]) for m in c["messages"])
        short = bc.shorten(conversation, size, size(conversation) // 2)
        self.assertLessEqual(size(short), size(conversation) // 2)
        self.assertEqual(short["messages"][-1]["role"], "assistant")
        self.assertEqual(short["messages"], conversation["messages"][:len(short["messages"])])
        with self.assertRaises(bc.Reject):
            bc.shorten(conversation, size, 10)

    def test_special_token_overlap_detection(self):
        tokens = {"byte_end": np.array([3, 8, 12, 20]), "special": np.array([1, 0, 1, 0], dtype=np.uint8)}
        self.assertFalse(bc.special_inside_segments(tokens, [(0, 3, 8), (1, 12, 20)]))
        self.assertTrue(bc.special_inside_segments(tokens, [(0, 3, 10)]))
        self.assertTrue(bc.special_inside_segments(tokens, [(0, 2, 5)]))

    def test_splits_are_stratified_and_deterministic(self):
        items = []
        for subset in ("a", "b"):
            for number in range(40):
                conversation = self.conversation(1)
                conversation["id"], conversation["source"] = f"{subset}:{number}", subset
                items.append({"conversation": conversation, "tokens": {"muse": 100, "qwen": 100}})
        bc.assign_splits(items, {"dev": .1, "test": .15}, 7)
        first = [item["conversation"]["split"] for item in items]
        for subset in ("a", "b"):
            splits = [item["conversation"]["split"] for item in items if item["conversation"]["source"] == subset]
            self.assertEqual(splits.count("dev"), 4)
            self.assertEqual(splits.count("test"), 6)
        again = [dict(item, conversation=dict(item["conversation"])) for item in items]
        bc.assign_splits(again, {"dev": .1, "test": .15}, 7)
        self.assertEqual(first, [item["conversation"]["split"] for item in again])


def conversation_with(users, final=None, source="s", number=0):
    messages = [{"role": "system", "content": "You are helpful."}]
    for user in users:
        messages += [{"role": "user", "content": user}, {"role": "assistant", "content": "Sure."}]
    if final is not None:
        messages[-1] = final
    return {"id": f"{source}:{number}", "source": source, "tools": None, "messages": messages}


class FinalMessageCutTests(unittest.TestCase):
    def test_cut_at_whitespace(self):
        text = "alpha beta\ngamma  delta épsilon"
        self.assertEqual(bc.cut_at_whitespace(text, 100), text)
        self.assertEqual(bc.cut_at_whitespace(text, 12), "alpha beta")
        self.assertEqual(bc.cut_at_whitespace(text, 10), "alpha beta")
        self.assertEqual(bc.cut_at_whitespace(text, 9), "alpha")
        self.assertEqual(bc.cut_at_whitespace(text, 9, keep_bytes=6), "alpha beta")  # first boundary past both
        self.assertEqual(bc.cut_at_whitespace(text, 3), "alpha")
        self.assertEqual(bc.cut_at_whitespace(text, 25), "alpha beta\ngamma  delta")
        self.assertEqual(bc.cut_at_whitespace(text, 40, keep_bytes=10**6), text)
        self.assertEqual(bc.cut_at_whitespace("x" * 50, 10), "x" * 50)
        cjk = "日本語 日本語 日本語"
        self.assertEqual(bc.cut_at_whitespace(cjk, 20), "日本語 日本語")
        for limit in range(1, 40):
            result = bc.cut_at_whitespace(text + " " + cjk, limit)
            self.assertEqual(result, result.strip())
            self.assertTrue((text + " " + cjk).startswith(result))

    def test_reasoning_is_cut_first_and_the_answer_keeps_text(self):
        final = {"role": "assistant", "reasoning_content": " ".join(["think"] * 100), "content": "The answer is 42."}
        conversation = conversation_with(["Question?"], final)
        cut, changes = bc.cut_final_message(conversation, 200)
        message = cut["messages"][-1]
        self.assertEqual(message["content"], "The answer is 42.")
        self.assertEqual(changes, {"reasoning_content": [599, len(message["reasoning_content"])]})
        self.assertLessEqual(len(message["reasoning_content"]) + len(message["content"]), 200)
        self.assertEqual(conversation["messages"][-1], final)  # input untouched
        self.assertEqual(cut["messages"][:-1], conversation["messages"][:-1])
        long_answer = dict(final, content=" ".join(["answer"] * 100))
        cut, changes = bc.cut_final_message(conversation_with(["Question?"], long_answer), 200)
        message = cut["messages"][-1]
        self.assertEqual(set(changes), {"reasoning_content", "content"})
        self.assertGreaterEqual(len(message["reasoning_content"]), 95)  # at least about half the budget
        self.assertTrue(message["content"].startswith("answer"))
        self.assertLessEqual(len(message["reasoning_content"]) + len(message["content"]), 200)

    def test_keep_survives_and_small_messages_are_untouched(self):
        final = {"role": "assistant", "reasoning_content": " ".join(["think"] * 100), "content": "Done."}
        conversation = conversation_with(["Question?"], final)
        cut, _ = bc.cut_final_message(conversation, 200, {"reasoning_content": 400})
        self.assertGreaterEqual(len(cut["messages"][-1]["reasoning_content"]), 400)
        cut, _ = bc.cut_final_message(conversation, 200, {"reasoning_content": 10**6})
        self.assertEqual(cut["messages"][-1]["reasoning_content"], final["reasoning_content"])
        self.assertEqual(bc.cut_final_message(conversation, 10**6), (conversation, {}))
        no_reasoning = conversation_with(["Question?"], {"role": "assistant", "content": "word " * 99 + "end"})
        cut, changes = bc.cut_final_message(no_reasoning, 100)
        self.assertEqual(list(changes), ["content"])
        self.assertLessEqual(len(cut["messages"][-1]["content"]), 100)

    def test_final_tool_calls_are_rejected(self):
        final = {"role": "assistant", "content": "Calling.", "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "f", "arguments": {}}}]}
        conversation = conversation_with(["Question?"], final)
        conversation["tools"] = [{"type": "function", "function": {"name": "f"}}]
        self.assertEqual(bc.final_answer_problem(conversation), "final message has tool calls")
        with self.assertRaises(bc.Reject):
            bc.cut_final_message(conversation, 10)

    def test_final_window_and_keep_bytes(self):
        # text: prefix [0, 10), final segments [10, 20) (reasoning) and [25, 40) (content); 1 token per 5 bytes
        rendered = chat.Rendered(text="x" * 45, segments=[(0, 2, 8), (1, 10, 20), (2, 25, 40)],
                                 segment_keys=["1:content", "2:reasoning", "2:content"], final_message_segments=[1, 2])
        tokens = {"byte_end": np.arange(5, 46, 5)}  # 9 tokens ending at 5, 10, ..., 45
        window = bc.final_window(tokens, rendered, 3)
        # P: tokens ending at or before byte 10 -> 2; scored: [10,15) [15,20) [25,30) [30,35) [35,40)
        self.assertEqual(window, {"n_tokens": 9, "prefix_tokens": 2, "final_tokens": 7, "scored_tokens": 5,
                                  "check_tokens": 6})
        self.assertEqual(bc.final_window(tokens, rendered, 10)["check_tokens"], 9)
        keep = bc.keep_bytes(tokens, rendered, 6, margin=2)  # token 5 ends at byte 30 -> 32
        self.assertEqual(keep, {"reasoning_content": 22, "content": 7})
        self.assertEqual(bc.keep_bytes(tokens, rendered, 3, margin=0), {"reasoning_content": 5})
        with self.assertRaises(bc.Reject):
            bc.final_window(tokens, chat.Rendered(text="", segments=[(0, 2, 8)], segment_keys=["1:content"],
                                                  final_message_segments=[1]), 3)

    def test_scored_text_and_script_shares(self):
        text = "".join(chr(ord("a") + position // 5) for position in range(45))  # token n is five copies of letter n
        rendered = chat.Rendered(text=text, segments=[(0, 2, 8), (1, 10, 20), (2, 25, 40)],
                                 segment_keys=["1:content", "2:reasoning", "2:content"], final_message_segments=[1, 2])
        tokens = {"byte_end": np.arange(5, 46, 5)}
        self.assertEqual(bc.scored_text(tokens, rendered, 3), "cccccdddddfffff")
        self.assertEqual(bc.scored_text(tokens, rendered, 100), "cccccdddddfffffggggghhhhh")
        self.assertEqual(bc.script_shares("abc 日本"), {"non_latin": .4, "cjk": .4})
        shares = bc.script_shares("Zürich λόγος 42")
        self.assertAlmostEqual(shares["non_latin"], 5 / 11)
        self.assertEqual(shares["cjk"], 0)
        self.assertEqual(bc.script_shares("123 -- !"), {"non_latin": 0.0, "cjk": 0.0})
        self.assertEqual(bc.script_shares("한국어 かな"), {"non_latin": 1.0, "cjk": 1.0})
        conversation = conversation_with(["Hi", "日本"], {"role": "assistant", "reasoning_content": "Think.",
                                                           "content": "Done."})
        self.assertEqual(bc.user_text(conversation), "Hi\n日本")
        self.assertEqual(bc.final_text_head(conversation), "Think.\nDone.")
        self.assertEqual(bc.final_text_head(conversation, 3), "Thi")


class SelectionHelperTests(unittest.TestCase):
    def test_exclusions_match_ids_user_sequences_and_substantive_first_turns(self):
        prompt = "Please summarize the following report about river ecology in three sentences."
        excluded = [conversation_with(["Hi", "How are you?"], source="a", number=1),
                    conversation_with([prompt, "Shorter now."], source="b", number=2)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conversations.jsonl"
            path.write_text("".join(json.dumps(c) + "\n" for c in excluded), encoding="utf-8")
            exclusions = bc.read_exclusions([path])
            self.assertEqual(exclusions["files"][str(path)]["conversations"], 2)
            self.assertEqual(exclusions["files"][str(path)]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        reason = lambda c: bc.exclusion_reason(c, exclusions)
        self.assertEqual(reason(conversation_with(["Other"], source="a", number=1)), "excluded id")
        self.assertEqual(reason(conversation_with(["Hi", "How are you?"], source="c", number=9)), "excluded user turns")
        self.assertEqual(reason(conversation_with([prompt], source="c", number=9)), "excluded user turns")
        self.assertIsNone(reason(conversation_with(["Hi", "Tell me a joke."], source="c", number=9)))
        self.assertIsNone(bc.exclusion_reason(conversation_with(["Hi"]), None))

    def test_buckets(self):
        buckets = bc.parse_buckets("4096,8192,16384,29500")
        self.assertEqual(buckets, [(4096, 8192), (8192, 16384), (16384, 29500)])
        self.assertEqual(bc.bucket_of(8192, buckets), (8192, 16384))
        self.assertIsNone(bc.bucket_of(29500, buckets))
        self.assertIsNone(bc.bucket_of(4095, buckets))
        for bad in ("4096", "4096,4096", "0,10", "a,b", "10,5"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                bc.parse_buckets(bad)
        chosen = bc.parse_bucket_subsets('{"8192": "b", "16384": ["b", "a"]}', buckets, ["a", "b"])
        self.assertEqual(chosen, {(4096, 8192): ["a", "b"], (8192, 16384): ["b"], (16384, 29500): ["b", "a"]})
        for bad in ('{"5000": "a"}', '{"4096": "c"}', '{"4096": []}', '{"4096": "a,a"}'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                bc.parse_bucket_subsets(bad, buckets, ["a", "b"])
        self.assertEqual(bc.split_bucket((16384, 29500), 5),
                         [(16384, 19007), (19007, 21630), (21630, 24254), (24254, 26877), (26877, 29500)])
        self.assertEqual(bc.split_bucket((10, 20), 1), [(10, 20)])
        for bad in (0, 11):
            with self.subTest(parts=bad), self.assertRaises(ValueError):
                bc.split_bucket((10, 20), bad)
        bands = [(10, 15), (15, 20), (40, 50)]
        self.assertEqual([bc.nearest_band(value, bands) for value in (3, 10, 14.9, 15, 19.5, 29, 31, 49, 99)],
                         [(10, 15), (10, 15), (10, 15), (15, 20), (15, 20), (15, 20), (40, 50), (40, 50), (40, 50)])

    def test_invalid_arguments(self):
        base = ["--dataset-dir", "/nonexistent", "--probe", "/nonexistent", "--qwen-model", "/nonexistent",
                "--muse-model", "/nonexistent", "--output", "/nonexistent/corpus"]
        cases = [(["--max-doc-tokens", "40000"], "max-doc-tokens"),
                 (["--tool-limit", "-1"], "limit"),
                 (["--prefix-buckets", "10,20"], "docs-per-bucket"),
                 (["--prefix-buckets", "10,20", "--docs-per-bucket", "1"], "--split"),
                 (["--prefix-buckets", "10,20", "--docs-per-bucket", "1", "--split", "test", "--oversample", ".5"],
                  "long-context"),
                 (["--prefix-buckets", "10,20", "--docs-per-bucket", "1", "--split", "test",
                   "--strata-per-bucket", "0"], "long-context"),
                 (["--prefix-buckets", "10,20", "--docs-per-bucket", "1", "--split", "test",
                   "--strata-per-bucket", "11"], "strata"),
                 (["--prefix-buckets", "10,20", "--docs-per-bucket", "1", "--split", "test",
                   "--max-non-latin-share", "1"], "long-context"),
                 (["--subsets", "LongAlign_64k_Qwen3_32B_yarn_131k_think"], "--mix"),
                 (["--mix", '{"a": 1}', "--subsets", "b"], "--mix share"),
                 (["--mix", '{"a": 0}'], "positive"),
                 (["--subsets", "a,a", "--mix", '{"a": 1}'], "distinct")]
        for extra, fragment in cases:
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, fragment):
                bc.main(base + extra)


class FakeTable:
    """The slice/to_pylist/num_rows surface of a pyarrow table that the builder uses."""

    def __init__(self, rows):
        self.rows, self.num_rows = rows, len(rows)

    def slice(self, start, length):
        return FakeTable(self.rows[start:start + length])

    def to_pylist(self):
        return [json.loads(json.dumps(row)) for row in self.rows]


FIXTURE_WORDS = ["alpha", "river", "café", "naïve", "日本語", "données", "Zürich", "λόγος", "stone", "light"]


def words(rng, count):
    return " ".join(FIXTURE_WORDS[int(index)] for index in rng.integers(0, len(FIXTURE_WORDS), count))


def fixture_rows(seed):
    rng = np.random.default_rng(seed)
    long_rows, chat_rows = [], []
    for number in range(40):
        document = "\n".join(words(rng, 12) for _ in range(int(rng.integers(3, 16))))
        reasoning = words(rng, int(rng.integers(20, 90)))
        answer = words(rng, int(rng.integers(5, 40)))
        long_rows.append(row([("user", f"{document}\n\nWhich word comes first?"),
                              ("assistant", f"<think>\n{reasoning}\n</think>\n\n{answer}")]))
    for number in range(60):
        turns = []
        for _ in range(int(rng.integers(1, 5))):
            turns += [("user", words(rng, int(rng.integers(5, 30)))), ("assistant", words(rng, int(rng.integers(10, 60))))]
        chat_rows.append(row(turns))
    return {"longdoc": long_rows, "chatty": chat_rows}


@unittest.skipUnless(PROBE, "set KVPARK_KV_PROBE to run the builder on tiny fixtures")
class FixtureBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_run_v2 import MUSE_BOS, MUSE_CONTROL, MUSE_TEMPLATE, QWEN_CONTROL, QWEN_TEMPLATE
        from tiny_fixture import create_fixture
        cls.temp = tempfile.TemporaryDirectory(prefix="kvpark-corpus-")
        root = cls.root = Path(cls.temp.name)
        create_fixture(root / "qwen.gguf", "qwen35", control_tokens=QWEN_CONTROL, chat_template=QWEN_TEMPLATE)
        create_fixture(root / "muse.gguf", "muse-glimmer", control_tokens=MUSE_CONTROL, chat_template=MUSE_TEMPLATE,
                       bos_token=MUSE_BOS, add_bos=True)
        cls.templates = {name: chat.gguf_chat_context(root / f"{name}.gguf") for name in ("qwen", "muse")}
        cls.tables = fixture_rows(7)
        (root / "data").mkdir()
        for subset in cls.tables:
            (root / "data" / f"{subset}-00000-of-00001.parquet").write_bytes(subset.encode())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def build(self, name, *extra):
        output = self.root / name
        argv = ["--dataset-dir", str(self.root / "data"), "--probe", PROBE, "--qwen-model", str(self.root / "qwen.gguf"),
                "--muse-model", str(self.root / "muse.gguf"), "--output", str(output), *extra]
        tables = lambda paths: FakeTable(self.tables[Path(paths[0]).name.split("-")[0]])
        with mock.patch.object(bc, "read_subset", tables), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            bc.main(argv)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        conversations = [json.loads(line) for line in (output / "conversations.jsonl").read_text("utf-8").splitlines()]
        return manifest, conversations

    def tokenize(self, name, texts):
        records = [{"document_id": f"{number:064x}", "text": text} for number, text in enumerate(texts)]
        with contextlib.redirect_stderr(io.StringIO()):
            result = bc.tokenize_records(PROBE, self.root / f"{name}.gguf", records, workdir=self.root)
        return [result[record["document_id"]] for record in records]

    def test_tokenize_records_drops_documents_over_ctx(self):
        records = [{"document_id": f"{n:064x}", "text": text}
                   for n, text in enumerate(["short one", "x" * 80, "another short", "y" * 70])]
        with contextlib.redirect_stderr(io.StringIO()):
            result = bc.tokenize_records(PROBE, self.root / "muse.gguf", records, ctx=64, workdir=self.root)
        self.assertEqual(set(result), {record["document_id"] for record in records})
        self.assertIsNone(result[f"{1:064x}"])
        self.assertIsNone(result[f"{3:064x}"])
        self.assertEqual(len(result[f"{0:064x}"]["token_id"]), len("short one"))
        self.assertEqual(int(result[f"{2:064x}"]["byte_end"][-1]), len("another short"))

    def test_long_selection_buckets_cuts_and_verifies(self):
        excluded = self.root / "excluded.jsonl"
        blocked = bc.convert_row("longdoc", 3, self.tables["longdoc"][3])
        excluded.write_text(json.dumps(blocked, ensure_ascii=False) + "\n", encoding="utf-8")
        args = ["--subsets", "chatty,longdoc", "--prefix-buckets", "150,400,700,1200", "--docs-per-bucket", "2",
                "--reserve-per-bucket", "1", "--bucket-subsets", '{"150": "chatty,longdoc", "700": "longdoc"}',
                "--split", "test", "--max-doc-tokens", "2000", "--tool-limit", "0", "--reasoning-limit", "0",
                "--final-message-max-bytes", "200", "--scored-window", "128", "--bytes-per-token", "1",
                "--oversample", "3", "--exclude", str(excluded), "--seed", "5"]
        manifest, conversations = self.build("long", *args)
        long_eval = manifest["long_eval"]
        documents = long_eval["documents"]
        self.assertEqual(len(conversations), 9)
        self.assertEqual([d["id"] for d in documents], [c["id"] for c in conversations])
        self.assertNotIn("longdoc:3", [c["id"] for c in conversations])
        self.assertEqual([entry["documents"] for entry in long_eval["by_bucket"]], [{"primary": 2, "reserve": 1}] * 3)
        self.assertEqual({c["split"] for c in conversations}, {"test"})
        self.assertEqual({d["source"] for d in documents if d["bucket"][0] == 700}, {"longdoc"})
        self.assertTrue(any(d["final_cut"] for d in documents))
        self.assertTrue(any(d["final_cut_widened"] for d in documents))  # first cut fell inside the window
        for conversation, document in zip(conversations, documents):
            chat.validate_conversation(conversation)
            rendered = {name: chat.render(conversation, t, bos_token=b) for name, (t, b) in self.templates.items()}
            tokens = {name: self.tokenize(name, [rendered[name].text])[0] for name in rendered}
            window = bc.final_window(tokens["muse"], rendered["muse"], 128)
            self.assertEqual(document["document_id"], chat.document_id(conversation))
            self.assertEqual(document["muse_prefix_tokens"], window["prefix_tokens"])
            self.assertEqual(document["qwen_prefix_tokens"],
                             bc.final_window(tokens["qwen"], rendered["qwen"], 128)["prefix_tokens"])
            self.assertEqual(document["muse_tokens"], len(tokens["muse"]["token_id"]))
            self.assertTrue(document["bucket"][0] <= window["prefix_tokens"] < document["bucket"][1])
            source_row = self.tables[conversation["source"]][int(conversation["id"].split(":")[1])]
            original = bc.convert_row(conversation["source"], 0, source_row, tool_limit=0, reasoning_limit=0)
            final = conversation["messages"][-1]
            if not document["final_cut"]:
                self.assertEqual(final, original["messages"][-1])
                continue
            self.assertEqual(document["verified_tokens"]["muse"], window["check_tokens"])
            self.assertGreaterEqual(window["check_tokens"], window["prefix_tokens"] + 128)
            uncut = chat.render(original, *self.templates["muse"][:1], bos_token=self.templates["muse"][1])
            reference = self.tokenize("muse", [uncut.text])[0]
            check = window["check_tokens"]
            np.testing.assert_array_equal(reference["token_id"][:check], tokens["muse"]["token_id"][:check])
            for field, (old, new) in document["final_cut"].items():
                self.assertEqual(len(original["messages"][-1][field].encode()), old)
                self.assertEqual(len(final[field].encode()), new)
                self.assertTrue(original["messages"][-1][field].startswith(final[field]))
        _, again = self.build("long-again", *args)
        self.assertEqual(again, conversations)

    def test_long_selection_spreads_strata_and_filters_scripts(self):
        limit = .16  # the fixture words mix Latin with Greek and CJK; about half the rows pass
        manifest, conversations = self.build(
            "long-strata", "--subsets", "chatty,longdoc", "--prefix-buckets", "150,400,700,1200",
            "--docs-per-bucket", "2", "--reserve-per-bucket", "1", "--strata-per-bucket", "2",
            "--max-non-latin-share", str(limit), "--split", "test", "--max-doc-tokens", "2000", "--tool-limit", "0",
            "--reasoning-limit", "0", "--final-message-max-bytes", "200", "--scored-window", "128",
            "--bytes-per-token", "1", "--oversample", "3", "--seed", "5")
        long_eval = manifest["long_eval"]
        self.assertEqual((long_eval["strata_per_bucket"], long_eval["max_non_latin_share"]), (2, limit))
        for entry in long_eval["by_bucket"]:
            low, high = entry["bucket"]
            self.assertEqual([band["stratum"] for band in entry["by_stratum"]],
                             [list(band) for band in bc.split_bucket((low, high), 2)])
            self.assertEqual([band["primary"] for band in entry["by_stratum"]], [1, 1])  # one primary per stratum
            self.assertLessEqual(entry["max_script_share"]["scored_non_latin"], limit)
        reasons = " ".join(manifest["rejected"])
        self.assertIn("non-Latin script share over", reasons)
        self.assertIn("scored window non-Latin share over", reasons)
        for conversation, document in zip(conversations, long_eval["documents"]):
            self.assertTrue(document["bucket"][0] <= document["stratum"][0] <= document["muse_prefix_tokens"]
                            < document["stratum"][1] <= document["bucket"][1])
            rendered = chat.render(conversation, self.templates["muse"][0], bos_token=self.templates["muse"][1])
            tokens = self.tokenize("muse", [rendered.text])[0]
            scored = bc.script_shares(bc.scored_text(tokens, rendered, 128))
            user = bc.script_shares(bc.user_text(conversation))
            self.assertEqual(document["script"], {"user_non_latin": round(user["non_latin"], 4),
                                                  "user_cjk": round(user["cjk"], 4),
                                                  "scored_non_latin": round(scored["non_latin"], 4),
                                                  "scored_cjk": round(scored["cjk"], 4)})
            self.assertLessEqual(max(user["non_latin"], scored["non_latin"]), limit)

    def test_long_selection_reports_short_buckets(self):
        with self.assertRaisesRegex(ValueError, "not enough conversations"):
            self.build("long-short", "--subsets", "chatty", "--prefix-buckets", "150,400", "--docs-per-bucket", "500",
                       "--split", "test", "--bytes-per-token", "1", "--seed", "5")

    def test_quota_mode_single_split_with_exclusions(self):
        _, first = self.build("quota", "--mix", '{"chatty": 0.5, "longdoc": 0.5}', "--target-tokens", "3000",
                              "--max-doc-tokens", "1500", "--min-doc-tokens", "50", "--short-tokens", "300", "900",
                              "--split", "train", "--seed", "3")
        self.assertTrue(first)
        self.assertEqual({c["split"] for c in first}, {"train"})
        excluded = self.root / "quota-exclude.jsonl"
        excluded.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in first), encoding="utf-8")
        manifest, second = self.build("quota-2", "--subsets", "chatty", "--mix", '{"chatty": 1}',
                                      "--target-tokens", "1500", "--max-doc-tokens", "1500", "--min-doc-tokens", "50",
                                      "--short-tokens", "300", "900", "--split", "dev", "--exclude", str(excluded),
                                      "--seed", "3")
        self.assertTrue(second)
        self.assertEqual({c["source"] for c in second}, {"chatty"})
        self.assertFalse({c["id"] for c in first} & {c["id"] for c in second})
        self.assertIn("chatty: excluded id", manifest["rejected"])
        self.assertEqual(manifest["excluded_from"][str(excluded)]["conversations"], len(first))


@unittest.skipUnless(all(path.is_file() for path in MODELS.values()), "real GGUF models not present")
class RealTemplateTests(unittest.TestCase):
    def test_converted_tool_conversation_renders_with_both_templates(self):
        messages = [("user", "New task:\nWho wrote it?"),
                    ("assistant", "<think>Search first.</think>\n<tool_call>{'name': 'web_search', 'arguments': {'query': 'author ñ'}}</tool_call>"),
                    ("tool", "Observation:\nResult: Ada wrote it."),
                    ("assistant", "<think>Done.</think>\n<tool_call>{'name': 'final_answer', 'arguments': {'answer': 'Ada'}}</tool_call>")]
        conversation = bc.convert_row("a", 1, row(messages, xml_tools=[JSONL_TOOLS]))
        conversation["split"] = "test"
        for name, path in MODELS.items():
            template, bos = chat.gguf_chat_context(path)  # metadata only
            rendered = chat.render(conversation, template, bos_token=bos)
            data = rendered.text.encode("utf-8")
            texts = dict(chat.segment_texts(conversation))
            for index, start, end in rendered.segments:
                self.assertEqual(data[start:end].decode("utf-8"), texts[rendered.segment_keys[index]])
            self.assertTrue(rendered.final_message_segments)


if __name__ == "__main__":
    unittest.main()
