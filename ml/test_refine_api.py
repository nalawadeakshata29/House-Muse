import io
import unittest
from unittest.mock import patch
from app import app


class TestRefineImageEndpoint(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_missing_image_returns_400(self):
        resp = self.client.post(
            "/api/refine-image",
            json={"instruction": "add a brass lamp"},
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertIn("No init image", data["error"])

    def test_missing_instruction_returns_400(self):
        resp = self.client.post(
            "/api/refine-image",
            json={"image": "data:image/jpeg;base64,1234"},
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertIn("No instruction", data["error"])

    def test_expired_image_id_returns_404_image_expired(self):
        resp = self.client.post(
            "/api/refine-image",
            json={"image_id": "nonexistent_expired_id", "instruction": "add a plant"},
        )
        self.assertEqual(resp.status_code, 404)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data.get("code"), "image_expired")

    @patch("app.refine_room_image")
    def test_refine_json_success(self, mock_refine):
        mock_refine.return_value = {
            "success": True,
            "image_url": "/api/image/abc12345",
            "image_b64": "fakeb64data",
            "prompt": "contemporary Indian interior design, add a reading chair, keep the room's walls, windows, floor and ceiling unchanged",
            "error": None,
            "status_code": 200,
        }

        resp = self.client.post(
            "/api/refine-image",
            json={
                "image": "data:image/jpeg;base64,1234",
                "instruction": "add a reading chair",
                "style": "contemporary_indian",
                "room_type": "living_room",
                "strength": 0.50,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["instruction"], "add a reading chair")
        self.assertTrue(data["image_url"].startswith("/api/image/"))
        self.assertIsNotNone(data.get("image_id"))
        self.assertEqual(len(data["variants"]), 1)
        self.assertEqual(data["variants"][0]["variant_label"], "Refined View")

        mock_refine.assert_called_once()
        _, kwargs = mock_refine.call_args
        self.assertEqual(kwargs["instruction"], "add a reading chair")
        self.assertIn("add a reading chair", kwargs["positive_prompt"])
        self.assertIn("keep the room's walls, windows, floor and ceiling unchanged", kwargs["positive_prompt"])

    @patch("app.refine_room_image")
    def test_refine_quota_exhausted_returns_429(self, mock_refine):
        mock_refine.return_value = {
            "success": False,
            "image_url": None,
            "image_b64": None,
            "prompt": "test prompt",
            "error": "Hugging Face quota exhausted, try again later",
            "status_code": 429,
        }

        resp = self.client.post(
            "/api/refine-image",
            json={
                "image": "data:image/jpeg;base64,1234",
                "instruction": "add a reading chair",
            },
        )
        self.assertEqual(resp.status_code, 429)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertIn("quota exhausted", data["error"].lower())

    @patch("app.refine_room_image")
    def test_refine_space_asleep_returns_503(self, mock_refine):
        mock_refine.return_value = {
            "success": False,
            "image_url": None,
            "image_b64": None,
            "prompt": "test prompt",
            "error": "Hugging Face Space is currently sleeping or starting up, try again shortly",
            "status_code": 503,
        }

        resp = self.client.post(
            "/api/refine-image",
            json={
                "image": "data:image/jpeg;base64,1234",
                "instruction": "add a lamp",
            },
        )
        self.assertEqual(resp.status_code, 503)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertIn("sleeping", data["error"].lower())

    @patch("app.refine_room_image")
    def test_refine_multipart_success(self, mock_refine):
        mock_refine.return_value = {
            "success": True,
            "image_url": "/api/image/multipart123",
            "image_b64": "multipartb64",
            "prompt": "scandinavian interior, make the bed smaller, keep the room's walls, windows, floor and ceiling unchanged",
            "error": None,
            "status_code": 200,
        }

        image_data = io.BytesIO(b"fake-image-bytes")
        orig_data = io.BytesIO(b"fake-original-bytes")

        resp = self.client.post(
            "/api/refine-image",
            data={
                "image": (image_data, "init.jpg"),
                "original_image": (orig_data, "orig.jpg"),
                "instruction": "make the bed smaller",
                "styles": '["scandinavian"]',
                "room_type": "bedroom",
                "strength": "0.50",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["instruction"], "make the bed smaller")

        mock_refine.assert_called_once()
        _, kwargs = mock_refine.call_args
        self.assertEqual(kwargs["style"], "scandinavian")
        self.assertIn("make the bed smaller", kwargs["positive_prompt"])

    @patch("app.generate_room_image")
    def test_existing_generate_image_endpoint_unaffected(self, mock_generate):
        mock_generate.return_value = {
            "success": True,
            "image_url": "/api/image/gen123",
            "image_b64": "genb64",
            "error": None,
            "status_code": 200,
        }

        resp = self.client.post(
            "/api/generate-image",
            json={
                "room_type": "living_room",
                "styles": ["scandinavian"],
                "n_variants": 1,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(len(data["variants"]), 1)
        self.assertTrue(data["variants"][0]["image_url"].startswith("/api/image/"))

    def test_refine_hash_cache_hit_bypasses_inference(self):
        from modules.image_cache import save_cached_image, save_cached_refine_result
        img_id, _ = save_cached_image("test_proj", b"test_img_bytes")
        settings_dict = {"strength": 0.5, "steps": 25, "max_side": 768, "style": "contemporary_indian"}
        save_cached_refine_result(img_id, "add wooden shelves", settings_dict, {
            "success": True,
            "image_id": "cached_shelf_id",
            "image_url": "/api/image/cached_shelf_id",
            "image_b64": None,
            "prompt": "prompt with shelves",
            "instruction": "add wooden shelves",
            "error": None,
            "status_code": 200,
        })
        save_cached_image("test_proj", b"shelf_bytes", image_id="cached_shelf_id")

        with patch("modules.hf_space_client.call_hf_space_refine") as mock_hf:
            from modules.image_generator import refine_room_image
            res = refine_room_image(
                positive_prompt="base prompt",
                init_image=b"test_img_bytes",
                instruction="add wooden shelves",
                style="contemporary_indian",
                strength=0.5,
                image_id=img_id,
            )
            self.assertTrue(res["success"])
            self.assertEqual(res["image_id"], "cached_shelf_id")
            mock_hf.assert_not_called()


if __name__ == "__main__":
    unittest.main()
