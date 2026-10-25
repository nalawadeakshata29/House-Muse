# Config fixes (files not included in the export, so apply by hand)

All paths are relative to your project's `house-muse/` folder.
Ports are unchanged: frontend 3000, backend 8000, ML 5000.

## 1. frontend/vite.config.ts  -> add `strictPort`
Keeps port 3000 but makes Vite fail loudly instead of silently moving to 3001
(which would break CORS, since 3001 is not in CORS_ALLOWED_ORIGINS).

    server: {
      port: 3000,
      strictPort: true,
    },

## 2. backend/.env  -> make the database choice explicit
backend/.env has PostgreSQL credentials (DB_PORT=5432, DB_NAME, DB_USER) but no
DB_ENGINE, so settings.py falls back to SQLite. Nothing is listening on 5432,
so pick ONE:

 a) Stay on SQLite (works right now, no extra service needed):
        DB_ENGINE=sqlite
    (use whatever value your settings.py compares against; check the
     os.getenv("DB_ENGINE", ...) line in backend/<project>/settings.py)

 b) Use PostgreSQL: start Postgres on 5432, then set the Postgres value of
    DB_ENGINE (e.g. `postgresql`), and run `python manage.py migrate`.
    Existing SQLite data (db.sqlite3) is NOT migrated automatically.
    Note: login tokens (Knox) live in the DB, so users must log in again after
    switching. The frontend now clears stale tokens on 401 automatically.

## 3. frontend/.env.local
`VITE_API_BASE=http://localhost:8000` is now fine as is. api.ts appends `/api`
automatically if it is missing (with or without it both work).

## 4. ML weights (optional, the code now falls back safely)
- `ml/yolov8s-worldv2.pt` missing -> detector falls back to `ml/yolov8m.pt`
  (works, but no YOLO-World open-vocabulary detection, e.g. weaker windows).
  To get YOLO-World, from the `ml/` folder run:
      python -c "from ultralytics import YOLO; YOLO('yolov8s-worldv2.pt')"
  then move the downloaded file into `ml/`. Or set YOLO_WORLD_WEIGHTS=<path>
  in ml/.env (this env var is now actually honoured).
- `~/.cache/clip/ViT-B-32.pt` missing -> CLIP is now skipped gracefully instead
  of crashing /api/analyse. To enable it, run once with internet:
      python -c "import clip; clip.load('ViT-B/32')"
  (requires `pip install git+https://github.com/openai/CLIP.git`).

## 5. Directory nesting
The export's paths are relative to the repo root, i.e. the `house-muse/`
folder. Unzip so that `backend/`, `frontend/`, `ml/` land INSIDE `house-muse/`
(merge/overwrite), not next to it.
