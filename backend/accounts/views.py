from django.conf import settings
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import send_mail
from django.db import transaction
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from knox.models import AuthToken
from knox.views import LogoutAllView as KnoxLogoutAllView
from knox.views import LogoutView as KnoxLogoutView
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Profile
from .serializers import LoginSerializer, RegisterSerializer, UserSerializer

try:
    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests
except ImportError:  # google-auth not installed until requirements are set up
    google_id_token = None
    google_requests = None


class RegisterView(APIView):
    """POST username, email, password → creates user + returns a Knox token."""

    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        with transaction.atomic():
            user = serializer.save()
            _, token = AuthToken.objects.create(user)
        return Response(
            {"user": UserSerializer(user).data, "token": token},
            status=status.HTTP_201_CREATED,
        )


class LoginView(APIView):
    """POST username, password → returns a Knox token."""

    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = authenticate(
            username=serializer.validated_data["username"],
            password=serializer.validated_data["password"],
        )
        if not user:
            return Response({"detail": "Invalid credentials."}, status=status.HTTP_401_UNAUTHORIZED)
        _, token = AuthToken.objects.create(user)
        return Response({"user": UserSerializer(user).data, "token": token})


class GoogleLoginView(APIView):
    """
    POST { id_token: "<google id token from the frontend button>" }

    Verifies the token with Google, finds-or-creates the matching User,
    then issues a normal Knox token — same as email/password login from
    this point on.
    """

    permission_classes = [permissions.AllowAny]

    def post(self, request):
        if google_id_token is None:
            return Response(
                {"detail": "google-auth is not installed on the server yet."},
                status=status.HTTP_501_NOT_IMPLEMENTED,
            )

        token = request.data.get("id_token")
        if not token:
            return Response({"detail": "id_token is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            idinfo = google_id_token.verify_oauth2_token(
                token, google_requests.Request(), settings.GOOGLE_OAUTH_CLIENT_ID
            )
        except ValueError:
            return Response({"detail": "Invalid Google token."}, status=status.HTTP_401_UNAUTHORIZED)

        google_uid = idinfo["sub"]
        email = idinfo.get("email", "")
        first_name = idinfo.get("given_name", "")
        last_name = idinfo.get("family_name", "")
        avatar_url = idinfo.get("picture", "")

        profile = Profile.objects.filter(google_id=google_uid).first()
        if profile:
            user = profile.user
        else:
            # No existing Google-linked account — find by email or create fresh
            user, created = User.objects.get_or_create(
                email=email,
                defaults={
                    "username": email or f"google_{google_uid}",
                    "first_name": first_name,
                    "last_name": last_name,
                },
            )
            profile, _ = Profile.objects.get_or_create(user=user)
            profile.auth_provider = "google"
            profile.google_id = google_uid
            profile.avatar_url = avatar_url
            profile.save()

        _, token = AuthToken.objects.create(user)
        return Response({"user": UserSerializer(user).data, "token": token})


class LogoutView(KnoxLogoutView):
    """Logs out the current device's token only."""

    permission_classes = [permissions.IsAuthenticated]


class LogoutAllView(KnoxLogoutAllView):
    """Logs out every device/token for this user."""

    permission_classes = [permissions.IsAuthenticated]


class MeView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        return Response(UserSerializer(request.user).data)


class PasswordResetRequestView(APIView):
    """
    POST { email } → emails a reset link (only meaningful for
    email/password accounts; Google accounts don't have a password here).
    """

    permission_classes = [permissions.AllowAny]

    def post(self, request):
        email = request.data.get("email", "")
        user = User.objects.filter(email=email).first()
        if user and getattr(user, "profile", None) and user.profile.auth_provider == "email":
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            token = default_token_generator.make_token(user)
            reset_link = f"{settings.FRONTEND_URL}/reset-password/{uid}/{token}/"
            send_mail(
                subject="Reset your House Muse password",
                message=f"Reset your password here: {reset_link}",
                from_email=None,
                recipient_list=[email],
                fail_silently=True,
            )
        # Always return 200 — don't reveal whether the email exists
        return Response({"detail": "If that account exists, a reset link has been sent."})


class PasswordResetConfirmView(APIView):
    """POST { uid, token, new_password } → sets the new password."""

    permission_classes = [permissions.AllowAny]

    def post(self, request):
        uid = request.data.get("uid")
        token = request.data.get("token")
        new_password = request.data.get("new_password")

        try:
            user_id = force_str(urlsafe_base64_decode(uid))
            user = User.objects.get(pk=user_id)
        except (User.DoesNotExist, ValueError, TypeError, OverflowError):
            return Response({"detail": "Invalid reset link."}, status=status.HTTP_400_BAD_REQUEST)

        if not default_token_generator.check_token(user, token):
            return Response({"detail": "Invalid or expired reset link."}, status=status.HTTP_400_BAD_REQUEST)

        user.set_password(new_password)
        user.save()
        return Response({"detail": "Password updated. You can log in now."})
