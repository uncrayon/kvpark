"""pair_eval.py's renderings: the templates' defaults, and as llama-server renders a chat
(--server-render). No model is loaded: a GGUF holding only a tokenizer and a chat template."""

import sys
import tempfile
import unittest
from pathlib import Path

import pair_eval

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "linear_transfer"))
import tiny_fixture  # noqa: E402

# the shape of Qwen3.6's template: earlier turns' reasoning only with preserve_thinking, the last
# turn's always; enable_thinking only in the generation prompt
TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n"
            "{% if m.reasoning_content is defined and m.reasoning_content and "
            "((preserve_thinking is defined and preserve_thinking is true) or loop.last) %}"
            "<think>{{ m.reasoning_content }}</think>{% endif %}{{ m.content }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n"
            "{% if enable_thinking is defined and enable_thinking is false %}<think></think>{% endif %}{% endif %}")
CONVERSATION = {"id": "c1", "source": "eval", "messages": [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "First question?"},
    {"role": "assistant", "content": "First answer.", "reasoning_content": "Earlier reasoning."},
    {"role": "user", "content": "Second question?"},
    {"role": "assistant", "content": "Second answer.", "reasoning_content": "Last reasoning."}]}


def template_gguf(path, template=TEMPLATE):
    gguf = tiny_fixture._gguf_module()
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(["a", "b", "<|endoftext|>", "<|im_end|>"])
    writer.add_token_types([1, 1, 3, 3])
    writer.add_token_merges(["a b"])
    writer.add_bos_token_id(2)
    writer.add_eos_token_id(3)
    writer.add_chat_template(template)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


class RenderTest(unittest.TestCase):
    def test_the_server_keeps_earlier_reasoning_where_the_template_reads_it(self):
        with tempfile.TemporaryDirectory() as directory:
            gguf = template_gguf(Path(directory) / "m.gguf")
            default = pair_eval.render(CONVERSATION, gguf)
            server = pair_eval.render(CONVERSATION, gguf, server=True)
            again = pair_eval.render(CONVERSATION, gguf, server=True)
        self.assertNotIn("Earlier reasoning.", default["text"])
        self.assertIn("<think>Earlier reasoning.</think>First answer.", server["text"])
        self.assertIn("<think>Last reasoning.</think>Second answer.", default["text"])
        self.assertEqual(server["text"], again["text"])                     # the clock is fixed
        self.assertEqual(default["document_id"], server["document_id"])      # the same conversation
        self.assertGreater(len(server["segments"]), len(default["segments"]))  # the reasoning is a segment there

    def test_a_template_without_those_variables_renders_the_same(self):
        plain = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
                 "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
        with tempfile.TemporaryDirectory() as directory:
            gguf = template_gguf(Path(directory) / "m.gguf", plain)
            self.assertEqual(pair_eval.render(CONVERSATION, gguf)["text"],
                             pair_eval.render(CONVERSATION, gguf, server=True)["text"])


if __name__ == "__main__":
    unittest.main()
