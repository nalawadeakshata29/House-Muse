Export Manifest & Change Summary (Based on git diff --stat)
==========================================================
backend/studio/models.py - Project model field adjustments for generated image URLs
backend/studio/serializers.py - Project serializer updates for generated image URL support
backend/studio/services.py - Refine endpoint service logic, ML integration, and project generation workflow
backend/studio/tests.py - Test suite for studio services and refine endpoint functionality
backend/studio/views.py - Refine endpoint view and API request handling
backend/studio/migrations/0003_alter_project_generated_image_url.py - Migration altering Project generated_image_url field
frontend/src/index.css - UI styling including budget overrun banner and studio layout updates
frontend/src/lib/api.ts - Frontend API methods for refine endpoint and image generation requests
frontend/src/pages/Studio.tsx - Studio dashboard updates including refine actions and budget overrun banner
ml/hf_space/app.py - Hugging Face Space app entrypoint supporting YOLO-World detection and inference
ml/modules/caption_engine.py - Room captioning and window-preserving prompt formatting
ml/modules/controlnet_generator.py - ControlNet conditioning and window-preserving prompt support
ml/modules/hf_space_client.py - Remote Hugging Face Space client integration
ml/modules/image_generator.py - Image generator pipeline adjustments for refined styling
ml/modules/image_processor.py - YOLO-World detector integration and room segmentation/preprocessing
ml/test_hf_space.py - Test script for Hugging Face Space integration

Changes made after status review
================================
frontend/src/lib/api.ts - API base URL normalised: VITE_API_BASE works with or without trailing /api (was http://localhost:8000 vs fallback .../api). Ports unchanged.
ml/modules/image_processor.py - YOLO_WORLD_WEIGHTS / COCO_WEIGHTS env vars now honoured; CLIP load failure (missing ViT-B-32.pt) no longer crashes analysis.
config-fixes/APPLY_THESE_CONFIG_CHANGES.md - manual steps for vite strictPort, DB_ENGINE, weights, folder nesting (files not in this export).
