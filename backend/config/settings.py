
"""
House Muse — Django settings
 
Auth model:
  - django-rest-knox issues the actual API token every authenticated
    request uses (same pattern as Ikshaa).
  - Google Sign-In is just an alternate *entry point*: accounts/views.py
    verifies the Google ID token, finds/creates the User, then hands
    back a normal Knox token. From that point on, Knox runs the session
    exactly like an email/password login would.
"""
 
import os
from datetime import timedelta
from pathlib import Path
 
from dotenv import load_dotenv
 
load_dotenv()
 
BASE_DIR = Path(__file__).resolve().parent.parent
 
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-only-change-me")
DEBUG = os.getenv("DJANGO_DEBUG", "True") == "True"
ALLOWED_HOSTS = os.getenv("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
 
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # third-party
    "rest_framework",
    "knox",
    "corsheaders",
    # local
    "accounts",
    "studio",
]
 
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]
 
ROOT_URLCONF = "config.urls"
 
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]
 
WSGI_APPLICATION = "config.wsgi.application"
 
import sys

if "test" in sys.argv or os.getenv("DB_ENGINE", "django.db.backends.sqlite3") in ("sqlite", "django.db.backends.sqlite3"):
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": os.getenv("DB_ENGINE", "django.db.backends.postgresql"),
            "NAME": os.getenv("DB_NAME", "house_muse_db"),
            "USER": os.getenv("DB_USER", "house_muse_user"),
            "PASSWORD": os.getenv("DB_PASSWORD", ""),
            "HOST": os.getenv("DB_HOST", "localhost"),
            "PORT": os.getenv("DB_PORT", "5432"),
        }
    }
 
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]
 
LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Kolkata"
USE_I18N = True
USE_TZ = True
 
STATIC_URL = "static/"
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"
 
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
 
# ── DRF + Knox ───────────────────────────────────────────────────────────
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "knox.auth.TokenAuthentication",
    ),
    "DEFAULT_PERMISSION_CLASSES": (
        "rest_framework.permissions.IsAuthenticated",
    ),
}
 
REST_KNOX = {
    "TOKEN_TTL": timedelta(hours=24 * 14),   # 14 days, tune as needed
    "AUTH_HEADER_PREFIX": "Bearer",           # same convention as Ikshaa
    "TOKEN_LIMIT_PER_USER": None,             # allow multiple devices
}
 
# ── CORS (React dev server) ─────────────────────────────────────────────
# Vite is configured to run on :3000 in vite.config.ts (not the 5173
# default) — keeping 5173 here too in case that's ever reverted.
CORS_ALLOWED_ORIGINS = os.getenv(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
).split(",")
CORS_ALLOW_CREDENTIALS = True
 
# ── Google Sign-In ───────────────────────────────────────────────────────
GOOGLE_OAUTH_CLIENT_ID = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "")
 
# ── ml/ inference service (Flask, runs separately) ──────────────────────
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:5000")
 
# ── Frontend (used to build password-reset links) ───────────────────────
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
 
# ── Email (dev default: prints to console instead of actually sending) ──
EMAIL_BACKEND = os.getenv(
    "EMAIL_BACKEND", "django.core.mail.backends.console.EmailBackend"
)
 