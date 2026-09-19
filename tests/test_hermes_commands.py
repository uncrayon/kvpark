import unittest

from kvpark.hermes import command_args


class CommandArgumentsTests(unittest.TestCase):
    def test_rich_text_spaces_separate_actions_and_arguments(self):
        for space in (" ", "\t", "\n", "\u00a0", "\u2007", "\u202f", "\u3000"):
            with self.subTest(space=repr(space)):
                self.assertEqual(command_args(f"{space}uninstall{space}confirm{space}"), ["uninstall", "confirm"])
                self.assertEqual(command_args(f"setup{space}1{space}"), ["setup", "1"])
                self.assertEqual(command_args(f"update{space}"), ["update"])

    def test_quoted_and_escaped_model_names_remain_exact(self):
        model = "model\u00a0name/with spaces#tag"
        self.assertEqual(command_args(f'setup\u00a0--model\u00a0"{model}"'), ["setup", "--model", model])
        self.assertEqual(command_args("setup --model model\\\u00a0name"), ["setup", "--model", "model\u00a0name"])
        self.assertEqual(command_args("setup --model model#tag"), ["setup", "--model", "model#tag"])

    def test_unclosed_quotes_still_fail(self):
        with self.assertRaises(ValueError):
            command_args('uninstall "confirm')


if __name__ == "__main__":
    unittest.main()
