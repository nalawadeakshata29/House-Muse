# ML Service Setup & Configuration

The House Muse Machine Learning service is a Flask application running on **port 5000**. It powers spatial room analysis (object detection via YOLOv8, spatial layout, color extraction) and communicates with Hugging Face ZeroGPU Space for SDXL ControlNet room redesign and interactive conversational refinement.

---

## 1. Environment Variables (`ml/.env`)

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `HF_SPACE_ID` | `""` | Hugging Face Space ID hosting the SDXL ControlNet model (e.g. `username/house-muse-interior-sdxl`) |
| `HF_API_TOKEN` | `""` | Hugging Face User Access Token (charges ZeroGPU compute to caller's quota) |
| `REFINE_STRENGTH` | `0.5` | Diffusion denoising strength for refinement (0.0 to 1.0) |
| `REFINE_STEPS` | `25` | Number of inference steps for refinement (20-25 keeps GPU quota usage low) |
| `REFINE_MAX_SIDE` | `768` | Longest image edge in pixels capped before inference to save ZeroGPU compute |
| `STABILITY_API_KEY` | `""` | Optional: Stability AI fallback for text-only generation (never called for refinement) |

---

## 2. Quota Protection Architecture

To keep Hugging Face ZeroGPU quota consumption as low as possible:
1. **CPU Depth Map Generation**: The structural ControlNet depth condition is pre-computed on the Flask side (CPU) and cached per project. No depth preprocessing runs inside the GPU call.
2. **Refine Hash Caching**: Every refinement request is indexed by a SHA-256 hash of `(image_id + instruction + settings)`. Repeat queries return immediately from local cache without touching Hugging Face.
3. **No Automatic Retries**: Quota (429) or sleeping space (503) errors fail immediately with descriptive status codes, avoiding wasted GPU cycles.
4. **Dimension & Step Capping**: Refine calls are strictly capped at 25 steps and 768px long side.
5. **Image Memory**: Successful generations and refinements are stored in `ml/cache/` (retaining up to 10 iterations per project).

---

## 3. Running the Service

```bash
# In house-muse/ml:
python -m venv venv
venv\Scripts\activate   # On Windows
pip install -r requirements.txt

# Start Flask on port 5000:
python app.py
```
