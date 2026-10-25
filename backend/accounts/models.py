from django.contrib.auth.models import User
from django.db import models


class Profile(models.Model):
    """
    One row per User. Keeps track of *how* someone signed up, so the
    frontend can decide things like "show password reset" vs "this
    account uses Google Sign-In, nothing to reset".
    """

    AUTH_PROVIDER_CHOICES = [
        ("email", "Email & Password"),
        ("google", "Google Sign-In"),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    auth_provider = models.CharField(max_length=20, choices=AUTH_PROVIDER_CHOICES, default="email")
    google_id = models.CharField(max_length=255, blank=True, null=True, unique=True)
    avatar_url = models.URLField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} ({self.auth_provider})"
