# House Muse — Project Summary & Technical Architecture Report

---

## 1. Project Overview

### What is House Muse?
**House Muse** is an AI-powered interior architecture and design platform tailored for realistic home redesigns. Unlike generic text-to-image diffusion tools that create fantasy rooms detached from physical reality, House Muse anchors itself in the **real geometry, lighting, dimensions, and architectural boundaries** (windows, doorways, ceiling heights, and walls) of the user's actual room. It couples generative room visualization with **real-world catalog matching and budget breakdowns in Indian Rupees (₹)**.

### Who is it for?
- **Homeowners & Tenants:** Looking to renovate or furnish living rooms, bedrooms, kitchens, dining areas, bathrooms, or home offices within a realistic budget.
- **Interior Designers & Decorators:** Seeking rapid design ideation, automated client presentations, and instant cost estimations across distinct curated design styles.
- **Architectural Studios & Real Estate Developers:** Needing fast visual turnaround from unfurnished/empty room scans to styled, market-ready visual concepts.

### Main User Flow
1. **Upload Room Photo & Spatial Dimensions:** The user uploads a photo of their existing room and specifies physical room dimensions (Length × Width × Height in feet).
2. **Choose Scope & Room Type:** The user selects the room category (*Living Room*, *Master Bedroom*, *Culinary Kitchen*, *Dining Room*, *Bath & Spa*, *Library & Study*) and project scope (*Home Decor*, *Interior Design*, or *Full Makeover*).
3. **Select Design Style & Target Budget:** The user picks from 9 curated design styles (such as *Contemporary Indian*, *Vintage Indian*, *Scandinavian Japandi*, *Mid-Century Modern*, *Industrial*, or *Coastal*) and specifies their target budget in ₹.
4. **AI Room Redesign & Analysis:** 
   - Computer vision detects key existing objects and extracts colors and lighting.
   - Spatial rules calculate room volume, ceiling proportions, and wallpaper suitability.
   - Generative diffusion models (SDXL + Depth ControlNet + Style LoRA) synthesize a photorealistic redesign while keeping original walls, windows, and structural openings intact.
5. **Interactive Budget Breakdown & Catalog Swapping:**
   - The user inspects an itemized cost distribution across 7 categories (*Furniture*, *Wallpaper*, *Curtains*, *Rugs*, *Lamps*, *Wall Art*, *Accents*).
   - An interactive SVG donut chart highlights budget allocation and flags any budget overruns/shortfalls.
   - Interactive hotspot markers on the redesigned image allow users to explore and swap items across 3 product tiers (*Essential*, *Atelier*, *Heritage*).
6. **Iterative Refinement via Chat:**
   - The user types plain-English design instructions in the studio chat panel (e.g., *"make the bed smaller, add a brass reading lamp and an armchair"*).
   - Image-to-image (img2img) diffusion updates the design iteratively while preserving the room's geometry and historical edits.

---

## 2. Architecture

House Muse is structured into **three decoupled services**:
1. **Frontend Client (Port 3000):** A modern single-page application (SPA) built with React 19, Vite, and TypeScript. Handles the user interface, before/after image viewers, interactive budget visualizations, and chat-based design refinement.
2. **Backend Gateway & Business Logic (Port 8000):** A Django 5.0 REST Framework service with Knox token authentication. Manages user accounts, project persistence (in SQLite/PostgreSQL), media file storage, product catalog queries, and orchestrates calls to the ML service.
3. **Machine Learning Service (Port 5000):** A Flask-based microservice hosting the vision processing pipeline (YOLOv8, SAM, CLIP, Places365, K-Means) and bridging generative diffusion pipelines (Hugging Face ZeroGPU Space and Stability AI).

### High-Level Request Flow

```
[ User Browser / Client ]
         │
         │  HTTP / REST (Port 3000)
         ▼
┌───────────────────────────────────────────────────────────┐
│                    Frontend (React + Vite)                │
│  - Landing & Gallery (/ )                                 │
│  - Studio Design Flow (/studio)                           │
│  - Project Portfolio (/dashboard)                         │
│  - Authentication (/auth)                                 │
└────────────────────────────┬──────────────────────────────┘
                             │
                             │  Axios REST API with Knox Bearer Token
                             ▼
┌───────────────────────────────────────────────────────────┐
│              Backend Service (Django 5 + DRF)             │
│                       Port 8000                           │
│  - /api/auth/        : Register, Login, Google OAuth      │
│  - /api/studio/      : Projects, Catalog, Swapping        │
│  - /api/studio/analyse/       : Validates & delegates     │
│  - /api/studio/refine-image/  : Manages edit history      │
│  - Database (SQLite / Postgres) & Media Storage           │
└────────────────────────────┬──────────────────────────────┘
                             │
                             │  HTTP POST (JSON & Multipart FormData)
                             ▼
┌───────────────────────────────────────────────────────────┐
│                 ML Service (Flask Server)                 │
│                       Port 5000                           │
│  - /api/analyse       : YOLOv8 + SAM + CLIP + K-Means     │
│  - /api/generate-image: BLIP captioning + Prompt builder  │
│  - /api/refine-image  : Prompt refinement & img2img logic │
└─────────────┬───────────────────────────────┬─────────────┘
              │                               │
              │ gradio_client                 │ HTTP REST
              ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│  Hugging Face ZeroGPU Space   │   │     External Fallback APIs    │
│  (arnav1711/house-muse-...)   │   │  - Stability AI (SDXL API)    │
│  - Depth-Anything Preprocessor│   │  - HF Inference API (BLIP)    │
│  - ControlNet Depth (0.55)    │   └───────────────────────────────┘
│  - 9 Style LoRA Adapters (0.8)│
│  - Endpoints: /redesign,      │
│               /refine         │
└───────────────────────────────┘
```

---

## 3. Tech Stack

### Frontend Service
- **Framework:** React 19 (`react` 19.2.1, `react-dom` 19.2.1)
- **Language:** TypeScript 5.6.3
- **Build Tool:** Vite 7.1.7 (`@vitejs/plugin-react` 5.0.4)
- **Routing:** `wouter` (v3.3.5) — lightweight hash/history routing
- **Styling:** Tailwind CSS v4 (`tailwindcss` 4.1.14 with `@tailwindcss/vite`) combined with structured design tokens in [index.css](file:///C:/Users/Akshata/Desktop/House-Muse-main/house-muse/frontend/src/index.css)
- **UI Components & Icons:** `lucide-react` (0.453.0), `@radix-ui/react-slot`, `clsx`, `class-variance-authority`, `tailwind-merge`
- **Charts & Visualizations:** Native custom SVG `DonutChart` component with dynamic sector offsets, category tooltips, and responsive budget legends
- **HTTP Client:** `axios` (v1.12.0) with custom authorization interceptors and automated 401 session clearing

### Backend Service
- **Framework:** Django 5.0.6
- **REST Layer:** Django REST Framework 3.15.1
- **Authentication:** `django-rest-knox` (v5.0.2) token-based authentication with 14-day rolling sessions, multi-device support, and `google-auth` (v2.29.0) Google Sign-In integration
- **CORS Handling:** `django-cors-headers` (v4.3.1)
- **Databases Supported:**
  - **Default:** SQLite (`db.sqlite3`) for immediate local zero-setup operation
  - **Production-Ready:** PostgreSQL via `psycopg[binary]` (v3.3.4)
- **Environment Management:** `python-dotenv` (v1.0.1)
- **Image Processing:** `Pillow` (v11.0+)
- **Key Apps:**
  - `accounts`: User registration, email/password login, Google OAuth verification, user profiles, password resets.
  - `studio`: Project persistence, room dimensions, recommendation engine wrappers, catalog models (`WallOption`, `DecorItem`), refinement logging (`ProjectRefinement`).

### ML Service
- **Framework:** Flask 2.3.3
- **CORS:** `flask-cors` (v4.0.0)
- **Python Version:** Python 3.10+
- **Deep Learning & Computer Vision:**
  - `ultralytics` (>=8.3.0) — YOLOv8 and YOLO-World inference
  - `opencv-python-headless` (4.8.1.78) — image preprocessing, CLAHE, denoising, color space conversions
  - `torch` & `torchvision` — tensor operations, EfficientNet, CLIP
  - `gradio_client` (>=1.3.0) — ZeroGPU Hugging Face Space client
  - `scikit-learn` — K-Means clustering, cosine similarity
  - `albumentations` (1.3.1), `colormath` (3.0.0), `PyYAML` (6.0.1)

---

## 4. AI and ML Models Used

| Model Name / Weights | Exact Version / Checkpoint | Primary Role | Loading Location | Operational Mode / Fallback | Execution Environment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO-World / YOLOv8** | `yolov8s-worldv2.pt` (primary) / `yolov8m.pt` (COCO fallback) | Detects room objects (furniture, windows, doors, lights) and spatial roles | [image_processor.py](file:///C:/Users/Akshata/Desktop/House-Muse-main/house-muse/ml/modules/image_processor.py) in `_load_yolo()` | If `yolov8s-worldv2.pt` is missing, falls back to local `yolov8m.pt` (COCO 80 classes) | Local CPU / CUDA GPU |
| **SAM (Segment Anything)** | `vit_b` (`sam_vit_b_01ec64.pth`) | Pixel-accurate object mask extraction to eliminate background bleed before color clustering | [image_processor.py](file:///C:/Users/Akshata/Desktop/House-Muse-main/house-muse/ml/modules/image_processor.py) in `_load_sam()` | If `models/sam_vit_b_...pth` missing, falls back to bounding-box crop | Local CPU / CUDA GPU |
| **CLIP** | `ViT-B/32` (512-dim) | Zero-shot style similarity scoring and semantic feature extraction | [image_processor.py](file:///C:/Users/Akshata/Desktop/House-Muse-main/house-muse/ml/modules/image_processor.py) in `_load_clip()` | If weights/internet missing, falls back to `torchvision.models.efficientnet_b4` | Local CPU / CUDA GPU |
| **Places365** | ResNet-50 (`places365`) | Indoor scene and room type classification | [image_processor.py](file:///C:/Users/Akshata/Desktop/House-Muse-main/house-muse/ml/modules/image_processor.py) in `_load_places365()` | If local cache not present, delegates to CLIP zero-shot or YOLO object voting | Local CPU / CUDA GPU |
| **BLIP** | `Salesforce/blip-image-captioning-large` | Visual caption generation describing uploaded room contents | [caption_engine.py](file:///C:/Users/Akshata/Desktop/House-Muse-main/house-muse/ml/modules/caption_engine.py) in `generate_caption()` | If `HF_API_TOKEN` missing or API errors, falls back to structured prompt builder | Hugging Face Serverless Inference API |
| **SDXL Base** | `stabilityai/stable-diffusion-xl-base-1.0` (fp16) | High-fidelity photorealistic room synthesis | [controlnet_generator.py](file:///C:/Users/Akshata/Desktop/House-Muse-main/house-muse/ml/modules/controlnet_generator.py) & [hf_space/app.py](file:///C:/Users/Akshata/Desktop/House-Muse-main/house-muse/ml/hf_space/app.py) | Paired with `madebyollin/sdxl-vae-fp16-fix` and DPM-Solver++ scheduler | Hugging Face ZeroGPU Space |
| **Depth ControlNet** | `diffusers/controlnet-depth-sdxl-1.0` | Geometry-locking condition preserving walls, ceilings, windows, and floors | [hf_space/app.py](file:///C:/Users/Akshata/Desktop/House-Muse-main/house-muse/ml/hf_space/app.py) in `generate_redesign()` | Conditioning scale calibrated to 0.55 (balances geometry lock and new furnishings) | Hugging Face ZeroGPU Space |
| **Depth-Anything** | `LiheYoung/depth-anything-small-hf` | High-contrast structural depth map estimation from raw room photo | [hf_space/app.py](file:///C:/Users/Akshata/Desktop/House-Muse-main/house-muse/ml/hf_space/app.py) in `_get_engine()` | Extracted before ControlNet conditioning | Hugging Face ZeroGPU Space |
| **9 Style LoRAs** | 9 custom `.safetensors` adapters | Applies curated style aesthetics without retraining the base diffusion model | [hf_space/app.py](file:///C:/Users/Akshata/Desktop/House-Muse-main/house-muse/ml/hf_space/app.py) & `models/loras/` | Loaded into VRAM at startup; swapped instantly via `set_adapters()` with scale=0.80 | Hugging Face ZeroGPU Space |
| **Stability AI (Fallback)** | SDXL 1024 v1.0 | External cloud fallback for text-to-image | [image_generator.py](file:///C:/Users/Akshata/Desktop/House-Muse-main/house-muse/ml/modules/image_generator.py) in `_generate_via_stability()` | Requires `STABILITY_API_KEY`; used only for text-only runs when configured | Stability AI REST API |


### Hugging Face Space Deep Dive (`hf_space/`)
The [ml/hf_space/app.py](file:///C:/Users/Akshata/Desktop/House-Muse-main/house-muse/ml/hf_space/app.py) file contains the full deployment code for the remote Hugging Face Space (`arnav1711/house-muse-interior-sdxl`):
- **ZeroGPU Allocation:** Leverages Hugging Face's serverless Nvidia A100 GPU compute via `@spaces.GPU(duration=60)`.
- **Preloaded Weights:** Keeps SDXL Base 1.0, Depth-Anything Small, Depth ControlNet, and all 9 style LoRA adapters resident in memory to ensure sub-15-second generation times.
- **Client Communication:** The main app calls it programmatically from [hf_space_client.py](file:///C:/Users/Akshata/Desktop/House-Muse-main/house-muse/ml/modules/hf_space_client.py) using `gradio_client.Client(space_id, token=hf_token)`:
  - Invokes `/redesign` for initial room re-imagination with `controlnet_scale=0.55` and `lora_scale=0.80`.
  - Invokes `/refine` for conversational modifications with `strength=0.35` (img2img) and optional geometry lock.
- **Quota Management:** Passing the user's `HF_API_TOKEN` charges ZeroGPU compute quota directly to the caller's account, preventing global Space exhaustion.

---

## 5. Image Pipeline Step by Step

```
[ Upload Room Photo ] 
         │
         ▼
[ Step 1: Room Preprocessing & Vision Analysis ]
  • image_processor.py::process_image()
  • CLAHE + Bilateral FastNlMeans Denoising
  • detect_lighting() -> Brightness & Color Temperature (Warm/Cool)
  • detect_objects() -> YOLOv8 / YOLO-World (detects windows, doors, sofa, etc.)
  • extract_object_colors() -> SAM mask -> K-Means (4 clusters)
  • extract_dominant_colors() -> Global K-Means (6 clusters)
  • compute_style_similarities() -> CLIP ViT-B/32 zero-shot style matching
  • infer_room_type() -> Places365 / CLIP / Object Voting
         │
         ▼
[ Step 2: Spatial & Palette Engineering ]
  • space_analyzer.py::analyse_space() -> Area, Volume, Aspect Ratio, Design Rules
  • color_palette.py::generate_palette() -> Harmonies (Complementary, Analogous, etc.)
  • recommendation_engine.py::get_style_palette() -> Style-driven overrides
         │
         ▼
[ Step 3: Prompt Building & Geometry Safeguards ]
  • caption_engine.py::get_generation_prompt()
  • Tries BLIP (Salesforce/blip-image-captioning-large) for image caption
  • Injects room furniture requirements (living room sofa, coffee table, lighting, rugs)
  • Injects preservation clauses: "keep walls, windows, doors, floor in place; do not cover windows"
         │
         ▼
[ Step 4: Diffusion Image Generation ]
  • image_generator.py::generate_room_image()
  • Calls hf_space_client.py::call_hf_space_redesign()
  • Remote HF Space executes Depth-Anything -> Depth Map -> ControlNet SDXL + Style LoRA
  • (Text-only fallback to Stability AI if no photo was uploaded and STABILITY_API_KEY is set)
         │
         ▼
[ Step 5: Catalog Budget Breakdown & Cost Calculation ]
  • services.py::calculate_catalog_budget_breakdown()
  • Queries WallOption & DecorItem tables scored by style tags and budget tier
  • Calculates total estimated cost vs user budget -> computes surplus/shortfall
         │
         ▼
[ Step 6: Conversational Refinement (Chat) ]
  • views.py::RefineImageView -> services.py::refine_room_image()
  • Fetches most recent generated image as init_image
  • Appends user instruction to canonical base style prompt
  • Invokes HF Space /refine (img2img, strength ~0.35)
  • Stores ProjectRefinement in database and returns new image + update history
```

---

## 6. Training and Evaluation

### Dataset & Preparation
The training suite in [ml/training/](file:///C:/Users/Akshata/Desktop/House-Muse-main/house-muse/ml/training) is built to specialize object detectors on indoor furnishings and architectural features:
- **Source Data:** Curated combination of the **IKEA product catalog** (clean product shots on plain backgrounds) and **ADE20K indoor scene annotations**.
- **Dataset Builder (`dataset_builder.py`):** Automatically processes raw images into standard YOLO bounding box format (`class x_center y_center width height`). For white-background catalog shots, it uses OpenCV contour detection (`auto_bbox`) to extract tight bounding boxes.

### Class Taxonomy (19 Classes)
1. `sofa` (maps couch)
2. `chair`
3. `bed`
4. `dining_table`
5. `wardrobe`
6. `bookshelf`
7. `lamp`
8. `rug`
9. `curtain`
10. `plant` (maps plant_pot)
11. `vase`
12. `cushion`
13. `painting`
14. `television`
15. `coffee_table`
16. `side_table`
17. `mirror`
18. `desk`
19. `clock`

### Two-Phase Training Protocol (`train_yolo.py`)
- **Base Architecture:** `yolov8m.pt` (COCO pretrained weights)
- **Phase 1 (Epochs 1–10):** Backbone frozen (`freeze=10`). High initial learning rate (`lr0=0.01`). Forces classification and regression heads to adapt to the 19 indoor classes without corrupting general visual features.
- **Phase 2 (Epochs 11–50):** Full model unfreezing. Optimizer: `AdamW` (`lr0=0.001`, `lrf=0.01`, `weight_decay=0.0005`). Augmentation: Hue/Saturation/Value adjustments (`hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`), Horizontal Flips, Mosaic (`1.0`), and MixUp (`0.1`).

### Evaluation Suite (`evaluate_yolo.py`)
Provides validation against ground-truth validation sets:
- **Metrics Tracked:**
  - **mAP@50:** Mean Average Precision at IoU threshold 0.50 (industry standard benchmark, target > 0.70).
  - **mAP@50-95:** Average mAP across IoU thresholds from 0.50 to 0.95 (target > 0.50).
  - **Precision:** $TP / (TP + FP)$ (measures false detection rates).
  - **Recall:** $TP / (TP + FN)$ (measures missed furniture rate).
  - **F1 Score:** Harmonic mean of Precision and Recall.
- **Outputs Generated:**
  - `confusion_matrix.png`: Normalized per-class confusion matrix highlighting common classification ambiguities (e.g., dining chairs vs. accent chairs).
  - `precision_recall_f1_per_class.png`: Visual triple-bar chart per object class.
  - `map50_per_class.png`: Horizontal bar chart ranking class detection performance.
  - `evaluation_summary.json`: Exported summary and performance grading.

---

## 7. API Endpoints

### Frontend Client Routes (Wouter)
| Route | Page Component | Purpose |
| :--- | :--- | :--- |
| `/` | [Home.tsx](file:///C:/Users/Akshata/Desktop/House-Muse-main/house-muse/frontend/src/pages/Home.tsx) | Landing page, interactive 5-scene architectural hero gallery, branding |
| `/studio` | [Studio.tsx](file:///C:/Users/Akshata/Desktop/House-Muse-main/house-muse/frontend/src/pages/Studio.tsx) | Core redesign studio: photo upload, dimension input, style & budget selection, before/after comparison, SVG budget donut chart, item swapper, refinement chat |
| `/dashboard` | [Dashboard.tsx](file:///C:/Users/Akshata/Desktop/House-Muse-main/house-muse/frontend/src/pages/Dashboard.tsx) | User design workspace: displays saved project portfolio, style statistics, and project launchers |
| `/auth` | [Auth.tsx](file:///C:/Users/Akshata/Desktop/House-Muse-main/house-muse/frontend/src/pages/Auth.tsx) | Account authentication: Email/Password login & registration, Google OAuth One-Tap sign-in |
| `/reset-password/:uid/:token` | [Auth.tsx](file:///C:/Users/Akshata/Desktop/House-Muse-main/house-muse/frontend/src/pages/Auth.tsx) | Password reset confirmation view |

### Backend API Endpoints (Django REST Framework - Port 8000)
| HTTP Method | Endpoint Path | View Class | Authentication | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/auth/register/` | `RegisterView` | Public | Create new user account with email, username, password; returns Knox token |
| `POST` | `/api/auth/login/` | `LoginView` | Public | Authenticate username/password; returns Knox token |
| `POST` | `/api/auth/google/` | `GoogleLoginView` | Public | Verify Google OAuth ID token, find/create user, issue Knox token |
| `POST` | `/api/auth/logout/` | `LogoutView` | Knox Token | Invalidate current device session token |
| `POST` | `/api/auth/logout-all/` | `LogoutAllView` | Knox Token | Invalidate all session tokens across all devices |
| `GET` | `/api/auth/me/` | `MeView` | Knox Token | Fetch current authenticated user profile |
| `POST` | `/api/auth/password-reset/` | `PasswordResetRequestView` | Public | Trigger password reset email / console link |
| `POST` | `/api/auth/password-reset/confirm/` | `PasswordResetConfirmView` | Public | Confirm token and set new password |
| `GET` / `POST` | `/api/studio/projects/` | `ProjectViewSet` | Knox Token | List user's saved redesign projects or create a new project |
| `GET` / `PUT` / `DELETE`| `/api/studio/projects/<id>/` | `ProjectViewSet` | Knox Token | Retrieve, update, or delete a specific saved design project |
| `POST` | `/api/studio/analyse/` | `AnalyseView` | Knox Token | Forward room parameters to ML service, run catalog pricing, save `Project` |
| `POST` | `/api/studio/generate-image/` | `GenerateImageView` | Knox Token | Trigger image generation via ML service and persist output in storage |
| `POST` | `/api/studio/refine-image/` | `RefineImageView` | Knox Token | Submit natural-language refinement, call ML img2img, save `ProjectRefinement` |

### ML Microservice Endpoints (Flask - Port 5000)
| HTTP Method | Endpoint Path | Function Handler | Input Format | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | `health()` | None | Liveness probe returning service operational status |
| `POST` | `/api/analyse` | `analyse()` | Multipart Form / JSON | Runs full vision analysis: YOLO object detection, SAM masks, CLIP features, lighting, K-Means palette, and initial recommendations |
| `POST` | `/api/palette` | `palette_only()` | JSON | Calculates spatial dimensions and color harmonies without running vision detection |
| `POST` | `/api/generate-image` | `generate_image()` | Multipart Form / JSON | BLIP captioning, prompt assembly, and diffusion generation (HF Space or fallbacks) |
| `POST` | `/api/refine-image` | `refine_image()` | Multipart Form / JSON | Img2img refinement: receives init image + user prompt instruction, calls HF Space `/refine` (strength 0.35) |

---

## 8. Configuration

### Environment Variables Matrix (Names Only — No Secrets)

#### Frontend (`frontend/.env.local` or `.env`)
- `VITE_API_BASE`: Target URL of the Django backend (e.g. `http://localhost:8000` or `http://localhost:8000/api`).
- `GOOGLE_OAUTH_CLIENT_ID`: Google Cloud Console OAuth 2.0 Web Client ID for Google Sign-In.

#### Backend (`backend/.env`)
- `DJANGO_SECRET_KEY`: Django cryptographic signing secret key.
- `DJANGO_DEBUG`: Boolean debug flag (`True` / `False`).
- `DJANGO_ALLOWED_HOSTS`: Comma-separated list of permitted host headers (`localhost,127.0.0.1`).
- `DB_ENGINE`: Database engine selector (`sqlite` or `django.db.backends.postgresql`).
- `DB_NAME`: PostgreSQL database name.
- `DB_USER`: PostgreSQL database user.
- `DB_PASSWORD`: PostgreSQL database password.
- `DB_HOST`: PostgreSQL database host (`localhost`).
- `DB_PORT`: PostgreSQL port (`5432`).
- `CORS_ALLOWED_ORIGINS`: Permitted frontend origins (`http://localhost:3000,http://127.0.0.1:3000`).
- `FRONTEND_URL`: URL used to format password reset links (`http://localhost:3000`).
- `GOOGLE_OAUTH_CLIENT_ID`: OAuth client ID matching the frontend configuration.
- `ML_SERVICE_URL`: URL of the Flask ML service (`http://localhost:5000`).
- `EMAIL_BACKEND`: Mail backend (defaults to `console.EmailBackend` for dev).

#### ML Service (`ml/.env`)
- `PORT`: Port for the Flask server (defaults to `5000`).
- `HF_SPACE_ID`: Remote Hugging Face Space repository ID (`arnav1711/house-muse-interior-sdxl`).
- `HF_API_TOKEN`: Hugging Face User Access Token (used for ZeroGPU quota assignment and BLIP inference).
- `REFINE_STRENGTH`: Diffusion denoising strength for refinement passes (defaults to `0.5`).
- `REFINE_STEPS`: Diffusion inference steps for refinement (defaults to `25`).
- `REFINE_MAX_SIDE`: Maximum image dimension in pixels for refinement (defaults to `768`).
- `YOLO_WEIGHTS`: Custom fine-tuned YOLO checkpoint path.
- `YOLO_WORLD_WEIGHTS`: Absolute path to `yolov8s-worldv2.pt`.
- `COCO_WEIGHTS`: Absolute path to COCO fallback weights `yolov8m.pt`.
- `SAM_CHECKPOINT`: Path to Segment Anything `vit_b` checkpoint.

### Ports and CORS Overview
| Service | Local Port | CORS Allowed Inbound Origins |
| :--- | :--- | :--- |
| **Frontend** | `3000` (enforced with `strictPort: true`) | N/A (Client Browser) |
| **Backend** | `8000` | `http://localhost:3000`, `http://127.0.0.1:3000`, `http://localhost:5173` |
| **ML Service**| `5000` | `http://localhost:3000`, `http://localhost:5500`, `http://127.0.0.1:5500` |

---

## 9. Current Status

### What Works Fully
- **Complete Studio Redesign Pipeline:** From uploading room photos and inputting dimensions to receiving AI redesigns and catalog breakdowns.
- **Automated Room & Space Analysis:** YOLOv8 object detection, CLAHE/denoising, ambient lighting assessment, and K-Means color palette extraction.
- **Budget Intelligence & Catalog Matching:** Real INR ₹ calculations from `catalog_seed.csv`, interactive category breakdown, budget overrun/shortfall flags, and 3-tier item alternatives (*Essential*, *Atelier*, *Heritage*).
- **Interactive Before/After Experience:** Side-by-side or slider comparisons with interactive hotspots highlighting specific furniture pieces.
- **User Authentication & Project Persistence:** Registration, Knox session management, Google OAuth One-Tap sign-in, and project saving to database.

### What is Partially Working & Safe Fallbacks
- **YOLO-World vs. COCO Fallback:** If `yolov8s-worldv2.pt` is not downloaded, the system automatically falls back to `yolov8m.pt` (already present in `ml/`). Open-vocabulary detection of windows and doors is slightly reduced in COCO mode, but analysis does not crash.
- **SAM Masking vs. Bounding-Box Crops:** If `sam_vit_b_01ec64.pth` is absent, the system extracts colors from rectangular bounding-box crops rather than pixel masks.
- **CLIP vs. EfficientNet:** If `ViT-B-32.pt` is not cached or OpenAI CLIP cannot be loaded, the system automatically switches to `torchvision.models.efficientnet_b4` for feature extraction.
- **Places365 Cache:** If the PyTorch Hub Places365 model is not cached, room inference cascades to CLIP zero-shot classification, and then to YOLO object voting.
- **Refinement Chatbot:** The refinement API is fully wired and tested in `test_refine_api.py`. When connected to the HF Space with available ZeroGPU quota, it performs iterative img2img edits. When ZeroGPU daily quota is exhausted, it reports the error directly instead of returning generic, mismatched text-only images.

### Known Issues & Operational Gotchas
1. **ZeroGPU Quota Expiration:** Free Hugging Face ZeroGPU accounts have daily burst and hourly quotas. Providing a personal `HF_API_TOKEN` transfers compute allocation to the individual account.
2. **Database Engine Choice:** The default settings run SQLite (`db.sqlite3`). If switching to PostgreSQL (`DB_ENGINE=postgresql`), ensure the Postgres server is active on port 5432 and run `python manage.py migrate`. Tokens in Knox live in the DB, so users must re-authenticate after switching engines.
3. **Strict Port 3000:** Vite is locked to port 3000 via `strictPort: true` so it fails visibly rather than slipping to port 3001 (which would be blocked by Django's CORS settings).

---

## 10. How to Run

Open **three separate Windows PowerShell windows** and run the commands below.

### 1. Start the Machine Learning Service (Port 5000)
```powershell
# In Window 1:
cd C:\Users\Akshata\Desktop\House-Muse-main\house-muse\ml
.\venv\Scripts\Activate.ps1
python app.py
```
*Expected output: `🏠 House Muse API → http://localhost:5000`*

### 2. Start the Django Backend (Port 8000)
```powershell
# In Window 2:
cd C:\Users\Akshata\Desktop\House-Muse-main\house-muse\backend
.\venv\Scripts\Activate.ps1
python manage.py migrate
python manage.py runserver 8000
```
*Expected output: `Starting development server at http://127.0.0.1:8000/`*

### 3. Start the Frontend Client (Port 3000)
```powershell
# In Window 3:
cd C:\Users\Akshata\Desktop\House-Muse-main\house-muse\frontend
npm run dev
```
*Expected output: `VITE v7.1.7 ready in ... ms ➜ Local: http://localhost:3000/`*

Open **`http://localhost:3000`** in any web browser to use House Muse.

---

## 11. Future Improvements

1. **Local GPU Self-Hosting via Docker:** Package the SDXL ControlNet + LoRA pipeline into a containerized TensorRT-LLM / vLLM-Omni environment on local NVIDIA RTX hardware (eliminating Hugging Face ZeroGPU quota limits).
2. **Inpainting for Localized Furniture Edits:** Implement mask-conditioned inpainting (e.g., using SAM masks or bounding box selection in the UI) to replace individual furniture items (e.g., "swap only this coffee table") without altering the rest of the generated room.
3. **Direct E-Commerce Affiliate Integration:** Connect the catalog database directly to real merchant APIs (such as IKEA India, Pepperfry, Urban Ladder, or West Elm) with live pricing, stock availability, and direct cart checkout links.
4. **3D Mesh & Gaussian Splat Export:** Reconstruct full 3D spatial representations from the room photo and generated design using monocular depth estimation, allowing users to take interactive 360° virtual walkthroughs.
5. **Multi-Angle Consistency:** Generate panoramic or multi-camera views (wide angle, corner reading nook, evening view) while maintaining strict object and architectural consistency across all shots.
