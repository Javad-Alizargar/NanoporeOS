from pathlib import Path
import tempfile
import unittest
from nanoporeos.security import load_secret


class PrivateLoader(unittest.TestCase):
    def test_data_only_and_exact_alias(self):
        with tempfile.TemporaryDirectory() as temp:
            apis=Path(temp)/'apis';apis.mkdir()
            p=apis/'chatgpt.txt';p.write_text('OPENAI_API_KEY=unit_test_value')
            self.assertEqual(load_secret(temp,'openai'),'unit_test_value')
            p.write_text('unit_test_base64==');self.assertEqual(load_secret(temp,'openai'),'unit_test_base64==')
            p.write_text('line1\nline2')
            with self.assertRaises(ValueError):load_secret(temp,'openai')
            with self.assertRaises(ValueError):load_secret(temp,'unidentified_provider')
            self.assertIsNone(load_secret(temp,'github'))


if __name__=='__main__':unittest.main()
