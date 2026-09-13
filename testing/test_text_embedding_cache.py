import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from toolkit.dataloader_mixins import TextEmbeddingFileItemDTOMixin


class DummyFileItem(TextEmbeddingFileItemDTOMixin):
    def load_caption(self):
        return None


def make_item(tmp_path, name, caption="fisheye180", content_addressed=True):
    path = tmp_path / name
    path.write_bytes(name.encode("utf-8"))
    item = DummyFileItem()
    item.path = str(path)
    item.caption = caption
    item.text_embedding_space_version = 7
    item.dataset_config = SimpleNamespace(
        cache_text_embeddings_content_addressed=content_addressed,
        dataset_path=str(tmp_path),
        do_i2v=False,
    )
    item.encode_control_in_text_embeddings = False
    item.control_path = None
    item.control_video_paths = []
    item.encode_first_frame_in_text_embeddings = False
    item.is_video = False
    return item


class TextEmbeddingCachePathTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_legacy_cache_paths_remain_per_item(self):
        first = make_item(self.tmp_path, "first.png", content_addressed=False)
        second = make_item(self.tmp_path, "second.png", content_addressed=False)
        self.assertNotEqual(
            first._build_text_embedding_path(), second._build_text_embedding_path()
        )

    def test_content_addressed_text_only_items_share_one_path(self):
        first = make_item(self.tmp_path, "first.png")
        second = make_item(self.tmp_path, "second.png")
        first_path = first._build_text_embedding_path()
        second_path = second._build_text_embedding_path()
        self.assertEqual(first_path, second_path)
        self.assertIn("content_v1", first_path)
        self.assertTrue(first_path.endswith(".safetensors"))

    def test_content_addressed_different_prompts_do_not_share(self):
        first = make_item(self.tmp_path, "first.png", caption="fisheye180")
        second = make_item(self.tmp_path, "second.png", caption="repair stereo")
        self.assertNotEqual(
            first._build_text_embedding_path(), second._build_text_embedding_path()
        )

    def test_content_addressed_visual_references_remain_distinct(self):
        first = make_item(self.tmp_path, "first.png")
        second = make_item(self.tmp_path, "second.png")
        control_a = self.tmp_path / "control_a.png"
        control_b = self.tmp_path / "control_b.png"
        control_a.write_bytes(b"a")
        control_b.write_bytes(b"b")
        for item, control in ((first, control_a), (second, control_b)):
            item.encode_control_in_text_embeddings = True
            item.control_path = str(control)
        self.assertNotEqual(
            first._build_text_embedding_path(), second._build_text_embedding_path()
        )

    def test_text_only_dropout_can_share_despite_different_controls(self):
        first = make_item(self.tmp_path, "first.png")
        second = make_item(self.tmp_path, "second.png")
        first.encode_control_in_text_embeddings = True
        second.encode_control_in_text_embeddings = True
        first.control_path = str(self.tmp_path / "control_a.png")
        second.control_path = str(self.tmp_path / "control_b.png")
        self.assertEqual(
            first._build_text_embedding_path(text_only=True),
            second._build_text_embedding_path(text_only=True),
        )

    def test_self_reference_and_first_frame_keys_include_source_media(self):
        first = make_item(self.tmp_path, "first.mp4")
        second = make_item(self.tmp_path, "second.mp4")
        self.assertNotEqual(
            first._build_text_embedding_path(dopsd_self_ref=True),
            second._build_text_embedding_path(dopsd_self_ref=True),
        )
        for item in (first, second):
            item.encode_first_frame_in_text_embeddings = True
            item.dataset_config.do_i2v = True
            item.is_video = True
        self.assertNotEqual(
            first._build_text_embedding_path(), second._build_text_embedding_path()
        )


if __name__ == "__main__":
    unittest.main()
