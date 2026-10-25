from unittest.mock import patch
from django.contrib.auth.models import User
from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient

from .models import Project, ProjectRefinement


class RefineImageAPITestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="password123")
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

        self.project = Project.objects.create(
            owner=self.user,
            name="My Scandinavian Living Room",
            room_type="living_room",
            length=14.0,
            width=12.0,
            height=9.0,
            styles=["scandinavian"],
            budget=75000.0,
            generated_image_url="https://images.unsplash.com/photo-1555041469-a586c61ea9bc",
        )

    def test_unauthenticated_request_rejected(self):
        anonymous_client = APIClient()
        response = anonymous_client.post(
            "/api/studio/refine-image/",
            {"project_id": self.project.id, "instruction": "add a reading chair"},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_project_not_found(self):
        response = self.client.post(
            "/api/studio/refine-image/",
            {"project_id": 99999, "instruction": "add a plant"},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_refine_without_prior_image(self):
        project_without_image = Project.objects.create(
            owner=self.user,
            name="Empty Project",
            room_type="bedroom",
            length=10.0,
            width=10.0,
            height=8.0,
            styles=["coastal"],
            budget=30000.0,
            generated_image_url="",
        )
        response = self.client.post(
            "/api/studio/refine-image/",
            {"project_id": project_without_image.id, "instruction": "make bed smaller"},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    @patch("studio.views.refine_room_image")
    def test_successful_refinement_and_history(self, mock_refine):
        mock_refine.return_value = {
            "success": True,
            "image_url": "https://example.com/refined_v1.jpg",
            "image_b64": None,
            "prompt": "scandinavian room, add a reading chair",
        }

        # First refinement
        response1 = self.client.post(
            "/api/studio/refine-image/",
            {"project_id": self.project.id, "instruction": "add a cosy reading chair"},
            format="json",
        )
        self.assertEqual(response1.status_code, status.HTTP_200_OK)
        data1 = response1.json()
        self.assertTrue(data1["success"])
        self.assertEqual(data1["instruction"], "add a cosy reading chair")
        self.assertEqual(len(data1["history"]), 1)
        self.assertEqual(data1["history"][0]["instruction"], "add a cosy reading chair")
        self.assertEqual(data1["history"][0]["resulting_image"], "https://example.com/refined_v1.jpg")

        # Verify project state updated
        self.project.refresh_from_db()
        self.assertEqual(self.project.generated_image_url, "https://example.com/refined_v1.jpg")
        self.assertEqual(self.project.refinements.count(), 1)

        # Second refinement - builds on first refinement
        mock_refine.return_value = {
            "success": True,
            "image_url": "https://example.com/refined_v2.jpg",
            "image_b64": None,
            "prompt": "scandinavian room, add a reading chair, make coffee table oak wood",
        }

        response2 = self.client.post(
            "/api/studio/refine-image/",
            {"project_id": self.project.id, "instruction": "make coffee table oak wood"},
            format="json",
        )
        self.assertEqual(response2.status_code, status.HTTP_200_OK)
        data2 = response2.json()
        self.assertTrue(data2["success"])
        self.assertEqual(len(data2["history"]), 2)
        self.assertEqual(data2["history"][0]["instruction"], "add a cosy reading chair")
        self.assertEqual(data2["history"][1]["instruction"], "make coffee table oak wood")
        self.assertEqual(data2["history"][1]["resulting_image"], "https://example.com/refined_v2.jpg")

        self.project.refresh_from_db()
        self.assertEqual(self.project.generated_image_url, "https://example.com/refined_v2.jpg")
        self.assertEqual(self.project.refinements.count(), 2)

    @patch("studio.views.refine_room_image")
    def test_refinement_with_base64_result(self, mock_refine):
        # 1x1 transparent png in b64
        sample_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        mock_refine.return_value = {
            "success": True,
            "image_url": None,
            "image_b64": sample_b64,
            "prompt": "refined prompt",
        }

        response = self.client.post(
            "/api/studio/refine-image/",
            {"project_id": self.project.id, "instruction": "add hanging brass pendant lights"},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(len(data["history"]), 1)
        refinement = ProjectRefinement.objects.filter(project=self.project).first()
        self.assertIsNotNone(refinement)
        self.assertTrue(bool(refinement.resulting_image))

    @patch("studio.views.refine_room_image")
    @patch("studio.views.generate_room_image")
    def test_generate_image_then_immediately_refine_succeeds_b64(self, mock_gen, mock_refine):
        # Project without prior image
        new_project = Project.objects.create(
            owner=self.user,
            name="New Project B64",
            room_type="living_room",
            length=12.0,
            width=10.0,
            height=9.0,
            styles=["contemporary_indian"],
            budget=50000.0,
            generated_image_url="",
        )
        sample_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        mock_gen.return_value = {
            "success": True,
            "variants": [
                {
                    "variant": 1,
                    "image_url": None,
                    "image_b64": sample_b64,
                    "prompt": "contemporary indian living room",
                }
            ],
        }

        # 1. Generate image
        gen_res = self.client.post(
            "/api/studio/generate-image/",
            {"project_id": new_project.id, "style": "contemporary_indian", "room_type": "living_room"},
            format="json",
        )
        self.assertEqual(gen_res.status_code, status.HTTP_200_OK)
        new_project.refresh_from_db()
        self.assertTrue(bool(new_project.generated_image_url))

        # 2. Immediately refine image without false "no image found" error
        mock_refine.return_value = {
            "success": True,
            "image_url": "https://example.com/refined_result.jpg",
            "image_b64": None,
            "prompt": "refined contemporary indian living room",
        }
        refine_res = self.client.post(
            "/api/studio/refine-image/",
            {"project_id": new_project.id, "instruction": "add brass pendant lamp"},
            format="json",
        )
        self.assertEqual(refine_res.status_code, status.HTTP_200_OK)
        refine_data = refine_res.json()
        self.assertTrue(refine_data["success"])
        self.assertEqual(len(refine_data["history"]), 1)

    @patch("studio.views.refine_room_image")
    @patch("studio.views.generate_room_image")
    def test_generate_image_then_immediately_refine_succeeds_url(self, mock_gen, mock_refine):
        new_project = Project.objects.create(
            owner=self.user,
            name="New Project URL",
            room_type="bedroom",
            length=12.0,
            width=10.0,
            height=9.0,
            styles=["vintage_indian"],
            budget=60000.0,
            generated_image_url="",
        )
        mock_gen.return_value = {
            "success": True,
            "variants": [
                {
                    "variant": 1,
                    "image_url": "https://example.com/generated_initial.jpg",
                    "image_b64": None,
                    "prompt": "vintage indian bedroom",
                }
            ],
        }

        # 1. Generate image
        gen_res = self.client.post(
            "/api/studio/generate-image/",
            {"project_id": new_project.id, "style": "vintage_indian", "room_type": "bedroom"},
            format="json",
        )
        self.assertEqual(gen_res.status_code, status.HTTP_200_OK)
        new_project.refresh_from_db()
        self.assertEqual(new_project.generated_image_url, "https://example.com/generated_initial.jpg")

        # 2. Immediately refine image without false "no image found" error
        mock_refine.return_value = {
            "success": True,
            "image_url": "https://example.com/refined_initial.jpg",
            "image_b64": None,
            "prompt": "refined vintage indian bedroom",
        }
        refine_res = self.client.post(
            "/api/studio/refine-image/",
            {"project_id": new_project.id, "instruction": "change wood to rosewood"},
            format="json",
        )
        self.assertEqual(refine_res.status_code, status.HTTP_200_OK)
        self.assertTrue(refine_res.json()["success"])

    @patch("studio.views.refine_room_image")
    def test_refinement_inference_failure_returns_500_with_underlying_error(self, mock_refine):
        mock_refine.return_value = {
            "success": False,
            "error": "Diffusion model out of VRAM / CUDA OOM",
        }
        response = self.client.post(
            "/api/studio/refine-image/",
            {"project_id": self.project.id, "instruction": "make walls darker"},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("Diffusion model out of VRAM", data["detail"])

    @patch("studio.views.refine_room_image")
    def test_refinement_ml_service_exception_returns_500_with_error(self, mock_refine):
        from studio.services import MLServiceError
        mock_refine.side_effect = MLServiceError("Connection to ML worker at localhost:5000 failed")

        response = self.client.post(
            "/api/studio/refine-image/",
            {"project_id": self.project.id, "instruction": "make walls darker"},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("Connection to ML worker", data["detail"])

    def test_generate_image_project_not_found_returns_404(self):
        response = self.client.post(
            "/api/studio/generate-image/",
            {"project_id": 88888, "style": "scandinavian"},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    @patch("studio.views.generate_room_image")
    def test_generate_image_failure_returns_500_with_error(self, mock_gen):
        mock_gen.return_value = {
            "success": False,
            "error": "ML service GPU timeout",
        }
        response = self.client.post(
            "/api/studio/generate-image/",
            {"project_id": self.project.id, "style": "scandinavian"},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("ML service GPU timeout", data["detail"])

    @patch("studio.views.refine_room_image")
    def test_refinement_lookup_finds_most_recent_image_even_if_empty_refinement_exists(self, mock_refine):
        # Create a project with an initial generated image
        proj = Project.objects.create(
            owner=self.user,
            name="Multi-refine Project",
            room_type="living_room",
            length=10.0,
            width=10.0,
            height=9.0,
            styles=["scandinavian"],
            budget=50000.0,
            generated_image_url="https://example.com/initial.jpg",
        )
        # Create a first refinement with a valid image
        ProjectRefinement.objects.create(
            project=proj,
            instruction="first refinement",
            resulting_image_url="https://example.com/refined_v1.jpg",
        )
        # Create a second refinement record that ended up with no resulting image
        ProjectRefinement.objects.create(
            project=proj,
            instruction="broken refinement",
            resulting_image=None,
            resulting_image_url="",
        )

        mock_refine.return_value = {
            "success": True,
            "image_url": "https://example.com/refined_v3.jpg",
            "image_b64": None,
            "prompt": "refined prompt",
        }

        # Next refinement should find the latest valid image (refined_v1.jpg) and succeed
        res = self.client.post(
            "/api/studio/refine-image/",
            {"project_id": proj.id, "instruction": "third refinement"},
            format="json",
        )
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        # Verify refine_room_image was called with the most recent valid image
        mock_refine.assert_called_once()
        self.assertEqual(mock_refine.call_args[1]["init_image"], "https://example.com/refined_v1.jpg")


class BudgetOverrunAPITestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="budgetuser", password="password123")
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    @patch("studio.views.analyse_room")
    def test_analyse_endpoint_returns_over_budget_flags(self, mock_analyse):
        mock_analyse.return_value = {
            "budget_breakdown": {
                "furniture": 60000.0,
                "wallpaper": 25000.0,
                "curtains": 15000.0,
                "rugs": 15000.0,
                "lamps": 10000.0,
                "wall_art": 10000.0,
                "accent": 10000.0,
            },
            "budget_breakdown_items": {},
            "total_estimated": 145000.0,
            "budget_remaining": -45000.0,
            "over_budget": True,
            "shortfall_amount": 45000.0,
            "budget_total": 100000.0,
            "recommendations": {
                "total_estimated": 145000.0,
                "budget_remaining": -45000.0,
                "over_budget": True,
                "shortfall_amount": 45000.0,
                "budget_total": 100000.0,
            },
        }

        response = self.client.post(
            "/api/studio/analyse/",
            {
                "room_type": "living_room",
                "length": 4.0,
                "width": 3.5,
                "height": 2.8,
                "styles": ["scandinavian"],
                "budget": 100000.0,
                "scope": "interior_design",
            },
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue(data["result"]["over_budget"])
        self.assertEqual(data["result"]["shortfall_amount"], 45000.0)
        self.assertEqual(data["result"]["recommendations"]["over_budget"], True)
        self.assertEqual(data["result"]["recommendations"]["shortfall_amount"], 45000.0)

    @patch("requests.post")
    def test_analyse_room_service_flags_over_budget_and_within_budget(self, mock_post):
        from studio.models import DecorItem, WallOption
        from studio.services import analyse_room

        WallOption.objects.create(
            name="Nordic Wall",
            category="wallpaper",
            price_per_sqft=100.0,
            style_tags=["scandinavian"],
            budget_tier="budget",
        )
        DecorItem.objects.create(
            name="Nordic Lounge Chair",
            category="seating",
            price=25000.0,
            style_tags=["scandinavian"],
            budget_tier="budget",
        )

        mock_resp = mock_post.return_value
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "room_type": "living_room",
            "recommendations": {}
        }

        # Case 1: Real recommendations exceed tiny budget (e.g. 5,000)
        res_over = analyse_room(
            room_type="living_room",
            length=14.0,
            width=12.0,
            height=9.0,
            styles=["scandinavian"],
            budget=5000.0,
            scope="interior_design",
            num_rooms=1,
            harmony="analogous",
        )
        total_est = round(sum(res_over["budget_breakdown"].values()), 2)
        self.assertTrue(total_est > 5000.0)
        self.assertTrue(res_over["over_budget"])
        self.assertEqual(res_over["shortfall_amount"], round(total_est - 5000.0, 2))
        self.assertEqual(res_over["budget_remaining"], round(5000.0 - total_est, 2))

        # Case 2: Real recommendations within large budget (e.g. 5,000,000)
        res_within = analyse_room(
            room_type="living_room",
            length=14.0,
            width=12.0,
            height=9.0,
            styles=["scandinavian"],
            budget=5000000.0,
            scope="interior_design",
            num_rooms=1,
            harmony="analogous",
        )
        total_est_within = round(sum(res_within["budget_breakdown"].values()), 2)
        self.assertTrue(total_est_within <= 5000000.0)
        self.assertFalse(res_within["over_budget"])
        self.assertEqual(res_within["shortfall_amount"], 0.0)
        self.assertGreaterEqual(res_within["budget_remaining"], 0.0)

