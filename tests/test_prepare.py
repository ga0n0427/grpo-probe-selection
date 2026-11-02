import json
from pathlib import Path
import tempfile
import unittest
from prepare import read_rows,validate_splits


class SplitTests(unittest.TestCase):
    def test_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError,'Overlapping'):
            validate_splits([dict(id='1',image_id='a')],[dict(id='2',image_id='a')],[dict(id='3',image_id='c')])

    def test_disjoint(self):
        validate_splits([dict(id='1',image_id='a')],[dict(id='2',image_id='b')],[dict(id='3',image_id='c')])

    def test_relative_image_path_and_duplicates(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'example.jpg').touch()
            rows=[dict(id='1',image_id='a',image='example.jpg',prompt='Question?',target='answer')]
            f=p/'input.json';f.write_text(json.dumps(rows))
            self.assertEqual(read_rows(f,1)[0]['image'],str((p/'example.jpg').resolve()))
            f.write_text(json.dumps(rows*2))
            with self.assertRaisesRegex(ValueError,'Duplicate'):read_rows(f,2)


if __name__=='__main__':unittest.main()
