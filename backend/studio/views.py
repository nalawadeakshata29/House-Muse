import base64
import logging
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.utils.timezone import now
from rest_framework import permissions, status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Project, ProjectRefinement
from .serializers import AnalyseRequestSerializer, ProjectSerializer, RefineImageRequestSerializer
from .services import MLServiceError, analyse_room, generate_room_image, refine_room_image

logger = logging.getLogger(__name__)


def _extract_error_detail(exc):
    """Extract a descriptive error message from an exception or its underlying cause."""
    cause = getattr(exc, "__cause__", None)
    if cause and hasattr(cause, "response") and cause.response is not None:
        try:
            data = cause.response.json()
            if isinstance(data, dict):
                return data.get("error") or data.get("detail") or str(data)
        except Exception:
            if cause.response.text:
                return cause.response.text[:200]
    return str(exc)


class ProjectViewSet(viewsets.ModelViewSet):
    """Standard CRUD for a user's saved design projects."""

    serializer_class = ProjectSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Project.objects.filter(owner=self.request.user)

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)


class AnalyseView(APIView):
    """
    POST room dimensions/budget/styles (+ optional image) →
    forwards to ml/'s /api/analyse, optionally saves the result as a
    Project, and returns it to the frontend.
    """

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        serializer = AnalyseRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data
        image_file = request.FILES.get("image")

        try:
            result = analyse_room(
                room_type=payload["room_type"],
                length=payload["length"],
                width=payload["width"],
                height=payload["height"],
                styles=payload["styles"],
                budget=payload["budget"],
                scope=payload["scope"],
                num_rooms=payload["num_rooms"],
                harmony=payload["harmony"],
                image_file=image_file,
            )
        except MLServiceError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_502_BAD_GATEWAY)

        project = None
        if payload.get("save", True):
            project = Project.objects.create(
                owner=request.user,
                name=payload.get("name", ""),
                room_type=payload["room_type"],
                length=payload["length"],
                width=payload["width"],
                height=payload["height"],
                styles=payload["styles"],
                budget=payload["budget"],
                scope=payload["scope"],
                num_rooms=payload["num_rooms"],
                harmony=payload["harmony"],
                source_image=image_file,
                result=result,
            )

        return Response({
            "result": result,
            "project": ProjectSerializer(project).data if project else None,
        })


class GenerateImageView(APIView):
    """POST → forwards to ml/'s /api/generate-image (Hugging Face ZeroGPU Space)."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        project_id = request.data.get("project_id")
        project = None
        if project_id:
            try:
                project = Project.objects.get(id=project_id, owner=request.user)
            except (Project.DoesNotExist, ValueError):
                return Response({"detail": "Project not found."}, status=status.HTTP_404_NOT_FOUND)

        source_image = project.source_image if (project and project.source_image) else None

        try:
            result = generate_room_image(request.data, image_file=source_image)
        except MLServiceError as exc:
            status_code = getattr(exc, "status_code", 500) or 500
            err_body = getattr(exc, "error_data", None) or {"error": str(exc), "detail": str(exc), "success": False}
            return Response(err_body, status=status_code)

        if not result.get("success"):
            error_msg = result.get("error") or "Image generation failed."
            status_code = result.get("status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response(
                {"detail": f"Generation failed: {error_msg}", "error": error_msg, "success": False},
                status=status_code,
            )

        if project and result.get("success"):
            variants = result.get("variants", [])
            image_b64 = result.get("image_b64") or (variants[0].get("image_b64") if variants else None)
            image_url = result.get("image_url") or (variants[0].get("image_url") if variants else None)

            persisted_url = None
            if image_b64:
                try:
                    img_bytes = base64.b64decode(image_b64)
                    file_name = f"projects/generated/gen_{project.id}_{int(now().timestamp())}.jpg"
                    saved_path = default_storage.save(file_name, ContentFile(img_bytes))
                    persisted_url = request.build_absolute_uri(default_storage.url(saved_path))
                except Exception as exc:
                    logger.warning("Failed saving base64 generated image to storage: %s", exc)
                    persisted_url = image_url or f"data:image/jpeg;base64,{image_b64}"
            elif image_url:
                persisted_url = image_url

            if persisted_url:
                project.generated_image_url = persisted_url
                project.save(update_fields=["generated_image_url"])
                if not result.get("image_url"):
                    result["image_url"] = persisted_url
                if variants and not variants[0].get("image_url"):
                    variants[0]["image_url"] = persisted_url

        return Response(result)


class RefineImageView(APIView):
    """
    POST → /api/studio/refine-image/
    Accepts:
        project_id: int
        instruction: str (e.g. "make the bed smaller, add a reading chair")

    Uses the project's stored original room photo + most recent generated image.
    Calls ml/ HF Space generation using the most recent generated image as img2img
    init image with lower strength (~0.3-0.4) and base style + instruction prompt.
    Saves refinement history per project and returns the new image + updated history.
    """

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        serializer = RefineImageRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        project_id = serializer.validated_data["project_id"]
        instruction = serializer.validated_data["instruction"].strip()

        try:
            project = Project.objects.get(id=project_id, owner=request.user)
        except (Project.DoesNotExist, ValueError):
            return Response({"detail": "Project not found."}, status=status.HTTP_404_NOT_FOUND)

        # 1. Locate the most recent generated image (from prior refinements or original generation)
        init_image = None

        # Check prior refinements in reverse chronological order for the most recent valid image
        for ref in project.refinements.order_by("-created_at", "-id"):
            if ref.resulting_image:
                try:
                    if hasattr(ref.resulting_image, "file") and ref.resulting_image.name:
                        init_image = ref.resulting_image
                        break
                except Exception:
                    pass
            if ref.resulting_image_url:
                init_image = ref.resulting_image_url
                break

        # If no prior refinement image was found, check project's original generated image
        if not init_image and project.generated_image_url:
            init_image = project.generated_image_url

        # Check if stored in project's result payload
        if not init_image and isinstance(project.result, dict):
            init_image = (
                project.result.get("generated_image_url")
                or project.result.get("image_url")
                or project.result.get("image_b64")
            )

        # Allow caller to pass init_image in request data as fallback
        if not init_image:
            init_image = (
                serializer.validated_data.get("init_image")
                or request.data.get("init_image")
                or request.data.get("image_url")
                or request.data.get("image")
            )

        if not init_image:
            return Response(
                {"detail": "No previous generated image found for this project. Please generate an image first."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 2. Retrieve original room photo (if any)
        original_image = project.source_image if project.source_image else None

        # 3. Build payload for ml/ service
        styles = project.styles if isinstance(project.styles, list) else [project.styles or "contemporary_indian"]
        primary_style = styles[0] if styles else "contemporary_indian"
        payload = {
            "project_id": project.id,
            "instruction": instruction,
            "style": primary_style,
            "styles": styles,
            "room_type": project.room_type,
            "strength": 0.50,
        }
        image_id = request.data.get("image_id") or request.data.get("current_image_id")
        if image_id:
            payload["image_id"] = image_id
        if request.data.get("init_image"):
            payload["init_image"] = request.data.get("init_image")

        # 4. Call ml/ service to refine
        try:
            result = refine_room_image(payload, init_image=init_image, original_image=original_image)
        except MLServiceError as exc:
            status_code = getattr(exc, "status_code", 500) or 500
            err_body = getattr(exc, "error_data", None) or {"error": str(exc), "detail": str(exc), "success": False}
            return Response(err_body, status=status_code)

        if not result.get("success"):
            error_msg = result.get("error") or "Refinement failed to produce an image."
            status_code = result.get("status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response(
                {"detail": f"Generation failed: {error_msg}", "error": error_msg, "success": False},
                status=status_code,
            )

        variants = result.get("variants", [])
        image_b64 = result.get("image_b64") or (variants[0].get("image_b64") if variants else None)
        image_url = result.get("image_url") or (variants[0].get("image_url") if variants else None)

        if not image_b64 and not image_url:
            return Response(
                {"detail": "Generation failed: No refined image returned by inference service.", "success": False},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # 5. Store refinement in history
        refinement = ProjectRefinement(project=project, instruction=instruction)
        if image_b64:
            try:
                img_bytes = base64.b64decode(image_b64)
                file_name = f"refine_{project.id}_{int(now().timestamp())}.jpg"
                refinement.resulting_image.save(file_name, ContentFile(img_bytes), save=False)
                refinement.resulting_image_url = request.build_absolute_uri(refinement.resulting_image.url)
            except Exception:
                refinement.resulting_image_url = f"data:image/jpeg;base64,{image_b64}"
        elif image_url:
            refinement.resulting_image_url = image_url

        refinement.save()

        # Update project's latest generated image pointer
        resulting_display = refinement.get_resulting_image(request)
        project.generated_image_url = resulting_display
        project.save(update_fields=["generated_image_url"])

        # 6. Retrieve full ordered history
        history = [
            {
                "id": r.id,
                "instruction": r.instruction,
                "resulting_image": r.get_resulting_image(request),
                "created_at": r.created_at.isoformat(),
            }
            for r in project.refinements.all().order_by("created_at")
        ]

        return Response({
            "success": True,
            "image_id": result.get("image_id"),
            "image_url": resulting_display,
            "image_b64": image_b64,
            "resulting_image": resulting_display,
            "instruction": instruction,
            "history": history,
            "project_id": project.id,
        })

