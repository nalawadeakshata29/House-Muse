"""
Small wrapper around calls to the ml/ service (the Flask app in ../ml/app.py).

Keeping this in one place means the views don't need to know about
request formatting, timeouts, etc — and it's the one spot to change if
the ml service's URL or contract ever shifts.
"""
import json
import os
import io
import base64
import requests
from django.conf import settings

import logging

logger = logging.getLogger(__name__)

ML_TIMEOUT_SECONDS = 300


class MLServiceError(Exception):
    def __init__(self, message, status_code=500, error_data=None):
        super().__init__(message)
        self.status_code = status_code
        self.error_data = error_data or {}


def calculate_catalog_budget_breakdown(
    *,
    styles: list,
    total_budget: float,
    wall_area: float = 300.0,
    scope: str = "both",
    room_type: str = "living_room",
    existing_breakdown: dict = None,
) -> tuple[dict, dict]:
    """
    Computes real catalog sums from WallOption and DecorItem tables,
    filtered by style tags and budget tier.

    Returns:
        tuple (budget_breakdown, budget_breakdown_items):
            budget_breakdown: dict of 7 category keys -> float sum of real item prices
            budget_breakdown_items: dict of 7 category keys -> list of item dicts
                [{'name': str, 'price': float, 'thumbnail': str, 'category': str, 'id': int}]
    """
    if isinstance(styles, str):
        try:
            parsed = json.loads(styles)
            styles = parsed if isinstance(parsed, list) else [styles]
        except Exception:
            styles = [styles]
    elif not isinstance(styles, list):
        styles = ["scandinavian"]

    norm_user_styles = [str(s).lower().replace("-", "_").strip() for s in styles if s]
    if not norm_user_styles:
        norm_user_styles = ["scandinavian"]

    try:
        budget_val = float(total_budget)
    except (ValueError, TypeError):
        budget_val = 180000.0

    try:
        wall_area_val = float(wall_area)
        if wall_area_val <= 0:
            wall_area_val = 300.0
    except (ValueError, TypeError):
        wall_area_val = 300.0

    if budget_val < 100000:
        target_tier = "budget"
    elif budget_val <= 350000:
        target_tier = "mid"
    else:
        target_tier = "premium"

    def _score_item(item, is_wall=False):
        score = 0.0
        raw_tags = getattr(item, "style_tags", []) or []
        item_styles = [str(t).lower().replace("-", "_").strip() for t in raw_tags]
        if any(s in item_styles for s in norm_user_styles):
            score += 30.0
        elif any(any(s in t for t in item_styles) or any(t in s for t in item_styles) for s in norm_user_styles):
            score += 15.0

        tier = str(getattr(item, "budget_tier", "")).lower().strip()
        if tier == target_tier or (target_tier == "premium" and tier in ("premium", "luxury")):
            score += 20.0
        elif tier == "mid":
            score += 10.0
        elif target_tier == "mid":
            score += 5.0

        if is_wall:
            price_sqft = float(getattr(item, "price_per_sqft", 0) or 0)
            score -= (price_sqft / 100.0)
        else:
            item_price = float(getattr(item, "price", 0) or 0)
            if item_price > budget_val:
                score -= 50.0

        return score

    breakdown_items = {
        "furniture": [],
        "wallpaper": [],
        "curtains": [],
        "rugs": [],
        "lamps": [],
        "wall_art": [],
        "accent": [],
    }

    try:
        from studio.models import DecorItem, WallOption

        # 1. Wallpaper from WallOption
        wall_qs = WallOption.objects.filter(category="wallpaper")
        if not wall_qs.exists():
            wall_qs = WallOption.objects.all()

        if wall_qs.exists():
            wall_candidates = list(wall_qs)
            wall_candidates.sort(key=lambda w: _score_item(w, is_wall=True), reverse=True)
            chosen_wall = wall_candidates[0]
            wall_price = round(float(chosen_wall.price_per_sqft) * wall_area_val, 2)
            breakdown_items["wallpaper"].append({
                "id": chosen_wall.id,
                "name": chosen_wall.name,
                "price": wall_price,
                "thumbnail": chosen_wall.image_url or "",
                "category": "wallpaper",
            })

        def _pick_decor_item(cat_filter, result_cat, count=1):
            if isinstance(cat_filter, str):
                qs = DecorItem.objects.filter(category=cat_filter)
            else:
                qs = DecorItem.objects.filter(category__in=cat_filter)
            if not qs.exists():
                return []
            candidates = list(qs)
            candidates.sort(key=lambda d: _score_item(d, is_wall=False), reverse=True)
            chosen = []
            for item in candidates[:count]:
                chosen.append({
                    "id": item.id,
                    "name": item.name,
                    "price": round(float(item.price), 2),
                    "thumbnail": item.image_url or "",
                    "category": result_cat,
                })
            return chosen

        # 2. Furniture: seating, table, storage
        breakdown_items["furniture"].extend(_pick_decor_item("seating", "furniture", 1))
        breakdown_items["furniture"].extend(_pick_decor_item("table", "furniture", 1))
        breakdown_items["furniture"].extend(_pick_decor_item("storage", "furniture", 1))

        # 3. Curtains
        breakdown_items["curtains"].extend(_pick_decor_item(["curtain", "curtains"], "curtains", 1))

        # 4. Rugs
        breakdown_items["rugs"].extend(_pick_decor_item(["rug", "rugs"], "rugs", 1))

        # 5. Lamps (lighting)
        breakdown_items["lamps"].extend(_pick_decor_item(["lighting", "lamps", "lamp"], "lamps", 1))

        # 6. Wall Art
        breakdown_items["wall_art"].extend(_pick_decor_item(["wall_art", "art"], "wall_art", 1))

        # 7. Accent: cushion, plant, mirror
        breakdown_items["accent"].extend(_pick_decor_item("cushion", "accent", 1))
        breakdown_items["accent"].extend(_pick_decor_item("plant", "accent", 1))
        breakdown_items["accent"].extend(_pick_decor_item("mirror", "accent", 1))

    except Exception as exc:
        logger.warning("Catalog models query failed in calculate_catalog_budget_breakdown: %s", exc)

    # Ensure all 7 categories have items even if database is empty/unseeded
    primary_style = norm_user_styles[0] if norm_user_styles else "scandinavian"
    style_label = primary_style.replace("_", " ").title()

    fallback_weights = {
        "furniture": 0.40,
        "wallpaper": 0.15,
        "curtains": 0.12,
        "rugs": 0.10,
        "lamps": 0.08,
        "wall_art": 0.08,
        "accent": 0.07,
    }
    if existing_breakdown and isinstance(existing_breakdown, dict) and len(existing_breakdown) == 7:
        target_amounts = {k: float(v) for k, v in existing_breakdown.items()}
    else:
        target_amounts = {k: round(budget_val * w, 2) for k, w in fallback_weights.items()}

    for cat_key in ["furniture", "wallpaper", "curtains", "rugs", "lamps", "wall_art", "accent"]:
        if not breakdown_items[cat_key]:
            amt = target_amounts.get(cat_key, round(budget_val * 0.1, 2))
            if cat_key == "furniture":
                seating_p = round(amt * 0.3807, 2)
                table_p = round(amt * 0.4338, 2)
                storage_p = round(amt - seating_p - table_p, 2)
                breakdown_items[cat_key] = [
                    {"name": f"{style_label} Seating", "price": seating_p, "thumbnail": "", "category": "furniture"},
                    {"name": f"{style_label} Table", "price": table_p, "thumbnail": "", "category": "furniture"},
                    {"name": f"{style_label} Storage", "price": storage_p, "thumbnail": "", "category": "furniture"},
                ]
            elif cat_key == "accent":
                cushion_p = round(amt * 0.35, 2)
                plant_p = round(amt * 0.30, 2)
                mirror_p = round(amt - cushion_p - plant_p, 2)
                breakdown_items[cat_key] = [
                    {"name": f"{style_label} Cushion", "price": cushion_p, "thumbnail": "", "category": "accent"},
                    {"name": f"{style_label} Plant", "price": plant_p, "thumbnail": "", "category": "accent"},
                    {"name": f"{style_label} Mirror", "price": mirror_p, "thumbnail": "", "category": "accent"},
                ]
            else:
                cat_name = cat_key.replace("_", " ").title()
                breakdown_items[cat_key] = [
                    {"name": f"{style_label} {cat_name}", "price": amt, "thumbnail": "", "category": cat_key}
                ]

    budget_breakdown = {}
    for cat_key in ["furniture", "wallpaper", "curtains", "rugs", "lamps", "wall_art", "accent"]:
        budget_breakdown[cat_key] = round(sum(it["price"] for it in breakdown_items[cat_key]), 2)

    return budget_breakdown, breakdown_items


def analyse_room(*, room_type, length, width, height, styles, budget, scope, num_rooms, harmony, image_file=None):
    data = {
        "room_type": room_type,
        "length": length,
        "width": width,
        "height": height,
        "styles": styles if isinstance(styles, str) else json.dumps(styles),
        "budget": budget,
        "scope": scope,
        "num_rooms": num_rooms,
        "harmony": harmony,
    }
    if image_file and hasattr(image_file, "seek"):
        try:
            image_file.seek(0)
        except Exception:
            pass
    files = {"image": image_file} if image_file else None

    try:
        resp = requests.post(
            f"{settings.ML_SERVICE_URL}/api/analyse",
            data=data,
            files=files,
            timeout=ML_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        result = resp.json()
    except requests.RequestException as exc:
        raise MLServiceError(f"ml service unreachable or errored: {exc}") from exc
    finally:
        if image_file and hasattr(image_file, "seek"):
            try:
                image_file.seek(0)
            except Exception:
                pass

    try:
        l = float(length)
        w = float(width)
        h = float(height)
        wall_area = 2 * (l + w) * h
    except (ValueError, TypeError):
        wall_area = 300.0

    raw_styles = styles
    if isinstance(styles, str):
        try:
            raw_styles = json.loads(styles)
        except Exception:
            raw_styles = [styles]
    elif not isinstance(styles, list):
        raw_styles = [styles] if styles else ["scandinavian"]

    existing_bd = None
    if isinstance(result, dict) and "recommendations" in result and isinstance(result["recommendations"], dict):
        existing_bd = result["recommendations"].get("budget_breakdown")

    breakdown, breakdown_items = calculate_catalog_budget_breakdown(
        styles=raw_styles,
        total_budget=float(budget),
        wall_area=float(wall_area),
        scope=str(scope),
        room_type=str(room_type),
        existing_breakdown=existing_bd,
    )

    if isinstance(result, dict):
        result["budget_breakdown"] = breakdown
        result["budget_breakdown_items"] = breakdown_items
        total_est = round(sum(breakdown.values()), 2)
        try:
            budget_val = float(budget)
        except (ValueError, TypeError):
            budget_val = 180000.0

        over_budget = total_est > budget_val
        shortfall_amount = round(total_est - budget_val, 2) if over_budget else 0.0

        result["over_budget"] = over_budget
        result["shortfall_amount"] = shortfall_amount
        result["total_estimated"] = total_est
        result["budget_total"] = budget_val
        result["budget_remaining"] = round(budget_val - total_est, 2)

        if "recommendations" in result and isinstance(result["recommendations"], dict):
            result["recommendations"]["budget_breakdown"] = breakdown
            result["recommendations"]["budget_breakdown_items"] = breakdown_items
            result["recommendations"]["total_estimated"] = total_est
            result["recommendations"]["budget_remaining"] = round(budget_val - total_est, 2)
            result["recommendations"]["over_budget"] = over_budget
            result["recommendations"]["shortfall_amount"] = shortfall_amount
            result["recommendations"]["budget_total"] = budget_val

    return result


def generate_room_image(payload: dict, image_file=None):
    try:
        if image_file:
            data = {}
            for key, val in payload.items():
                if isinstance(val, (dict, list)):
                    data[key] = json.dumps(val)
                elif val is not None:
                    data[key] = str(val)

            file_bytes = b""
            if hasattr(image_file, "open"):
                try:
                    image_file.open("rb")
                except Exception:
                    pass
            if hasattr(image_file, "seek"):
                try:
                    image_file.seek(0)
                except Exception:
                    pass
            if hasattr(image_file, "read"):
                try:
                    file_bytes = image_file.read()
                except Exception:
                    file_bytes = b""
            elif hasattr(image_file, "file") and hasattr(image_file.file, "read"):
                try:
                    file_bytes = image_file.file.read()
                except Exception:
                    file_bytes = b""

            file_name = os.path.basename(getattr(image_file, "name", "room.jpg") or "room.jpg")
            files = {"image": (file_name, file_bytes, "image/jpeg")} if file_bytes else None

            if files:
                resp = requests.post(
                    f"{settings.ML_SERVICE_URL}/api/generate-image",
                    data=data,
                    files=files,
                    timeout=ML_TIMEOUT_SECONDS,
                )
            else:
                resp = requests.post(
                    f"{settings.ML_SERVICE_URL}/api/generate-image",
                    data=data,
                    timeout=ML_TIMEOUT_SECONDS,
                )
        else:
            resp = requests.post(
                f"{settings.ML_SERVICE_URL}/api/generate-image",
                json=payload,
                timeout=ML_TIMEOUT_SECONDS,
            )
        if resp.status_code >= 400:
            try:
                err_data = resp.json()
            except Exception:
                err_data = {"error": resp.text}
            error_msg = err_data.get("error") or err_data.get("detail") or resp.text
            raise MLServiceError(error_msg, status_code=resp.status_code, error_data=err_data)
        return resp.json()
    except MLServiceError:
        raise
    except requests.RequestException as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", 502)
        try:
            err_data = exc.response.json()
        except Exception:
            err_data = {"error": str(exc)}
        raise MLServiceError(str(exc), status_code=status_code, error_data=err_data) from exc
    finally:
        if image_file and hasattr(image_file, "close"):
            try:
                image_file.close()
            except Exception:
                pass


def _get_image_file_tuple(image_source, default_name="image.jpg"):
    """Converts a file, field, path, URL, or base64 into a (filename, BytesIO, mime_type) tuple."""
    if not image_source:
        return None

    if hasattr(image_source, "read") or hasattr(image_source, "open"):
        try:
            if hasattr(image_source, "open"):
                try:
                    image_source.open("rb")
                except Exception:
                    pass
            if hasattr(image_source, "seek"):
                try:
                    image_source.seek(0)
                except Exception:
                    pass
            content = image_source.read()
            if content:
                name = getattr(image_source, "name", default_name)
                return (os.path.basename(name) or default_name, io.BytesIO(content), "image/jpeg")
        except Exception as exc:
            logger.warning("Could not read FieldFile %s: %s", image_source, exc)

    if isinstance(image_source, bytes):
        return (default_name, io.BytesIO(image_source), "image/jpeg")

    if isinstance(image_source, str):
        # 1. Local MEDIA_URL check (e.g. "/media/projects/..." or "http://localhost:5000/media/projects/...")
        media_url = getattr(settings, "MEDIA_URL", "/media/")
        if media_url and media_url in image_source:
            rel_path = image_source.split(media_url, 1)[1].lstrip("/")
            full_path = os.path.join(settings.MEDIA_ROOT, rel_path)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                try:
                    with open(full_path, "rb") as f:
                        return (os.path.basename(full_path), io.BytesIO(f.read()), "image/jpeg")
                except Exception as exc:
                    logger.warning("Failed reading local media file %s: %s", full_path, exc)

        # 2. Local filesystem path check
        if os.path.exists(image_source) and os.path.isfile(image_source):
            try:
                with open(image_source, "rb") as f:
                    return (os.path.basename(image_source), io.BytesIO(f.read()), "image/jpeg")
            except Exception as exc:
                logger.warning("Failed reading local file %s: %s", image_source, exc)

        # 3. Data URI check
        if image_source.startswith("data:image"):
            try:
                _, b64data = image_source.split(",", 1)
                data = base64.b64decode(b64data)
                return (default_name, io.BytesIO(data), "image/jpeg")
            except Exception as exc:
                logger.warning("Failed decoding data:image URI: %s", exc)

        # 4. HTTP / HTTPS URL
        if image_source.startswith("http://") or image_source.startswith("https://"):
            try:
                r = requests.get(image_source, timeout=30)
                if r.status_code == 200:
                    return (default_name, io.BytesIO(r.content), r.headers.get("Content-Type", "image/jpeg"))
            except Exception as exc:
                logger.warning("Failed fetching remote image %s: %s", image_source, exc)

        # 5. Raw base64 string
        try:
            data = base64.b64decode(image_source)
            if len(data) > 100:
                return (default_name, io.BytesIO(data), "image/jpeg")
        except Exception:
            pass

    return None


def refine_room_image(payload: dict, init_image=None, original_image=None):
    """
    POST → forwards to ml/'s /api/refine-image.
    Uses the most recent generated image as the init image for img2img refinement
    with lower strength (~0.3-0.4), preserving room structure from original photo.
    """
    try:
        data = {}
        for key, val in payload.items():
            if isinstance(val, (dict, list)):
                data[key] = json.dumps(val)
            elif val is not None:
                data[key] = str(val)

        files = {}
        init_tuple = _get_image_file_tuple(init_image, "init_image.jpg")
        if init_tuple:
            files["image"] = (init_tuple[0], init_tuple[1].getvalue(), init_tuple[2])
        elif init_image:
            if isinstance(init_image, str):
                data["image"] = init_image
            else:
                raise MLServiceError(f"Could not load or resolve initial image: {init_image}")

        orig_tuple = _get_image_file_tuple(original_image, "original_image.jpg")
        if orig_tuple:
            files["original_image"] = (orig_tuple[0], orig_tuple[1].getvalue(), orig_tuple[2])

        if files:
            resp = requests.post(
                f"{settings.ML_SERVICE_URL}/api/refine-image",
                data=data,
                files=files,
                timeout=ML_TIMEOUT_SECONDS,
            )
        else:
            resp = requests.post(
                f"{settings.ML_SERVICE_URL}/api/refine-image",
                json=payload,
                timeout=ML_TIMEOUT_SECONDS,
            )
        if resp.status_code >= 400:
            try:
                err_data = resp.json()
            except Exception:
                err_data = {"error": resp.text}
            error_msg = err_data.get("error") or err_data.get("detail") or resp.text
            raise MLServiceError(error_msg, status_code=resp.status_code, error_data=err_data)
        return resp.json()
    except MLServiceError:
        raise
    except requests.RequestException as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", 502)
        try:
            err_data = exc.response.json()
        except Exception:
            err_data = {"error": str(exc)}
        raise MLServiceError(str(exc), status_code=status_code, error_data=err_data) from exc

