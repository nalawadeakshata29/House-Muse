from rest_framework import serializers

from .models import Project, ProjectRefinement


class ProjectRefinementSerializer(serializers.ModelSerializer):
    resulting_image = serializers.SerializerMethodField()

    class Meta:
        model = ProjectRefinement
        fields = ["id", "instruction", "resulting_image", "created_at"]

    def get_resulting_image(self, obj):
        request = self.context.get("request")
        return obj.get_resulting_image(request)


class ProjectSerializer(serializers.ModelSerializer):
    history = serializers.SerializerMethodField()

    class Meta:
        model = Project
        fields = [
            "id", "name", "room_type", "length", "width", "height",
            "styles", "budget", "scope", "num_rooms", "harmony",
            "source_image", "generated_image_url", "result",
            "history", "created_at", "updated_at",
        ]
        read_only_fields = ["id", "result", "generated_image_url", "history", "created_at", "updated_at"]

    def get_history(self, obj):
        request = self.context.get("request")
        return [
            {
                "id": r.id,
                "instruction": r.instruction,
                "resulting_image": r.get_resulting_image(request),
                "created_at": r.created_at.isoformat(),
            }
            for r in obj.refinements.all().order_by("created_at")
        ]


class RefineImageRequestSerializer(serializers.Serializer):
    project_id = serializers.IntegerField(required=True)
    instruction = serializers.CharField(required=True, allow_blank=False)
    init_image = serializers.CharField(required=False, allow_blank=True, allow_null=True)


class AnalyseRequestSerializer(serializers.Serializer):
    """Validates the inputs before we forward them to the ml/ service."""

    room_type = serializers.CharField(default="living_room")
    length = serializers.FloatField(default=12)
    width = serializers.FloatField(default=10)
    height = serializers.FloatField(default=9)
    styles = serializers.ListField(child=serializers.CharField(), default=["Japandi"])
    budget = serializers.FloatField(default=50000)
    scope = serializers.CharField(default="both")
    num_rooms = serializers.IntegerField(default=1)
    harmony = serializers.CharField(default="analogous")
    save = serializers.BooleanField(default=True)  # whether to persist as a Project
    name = serializers.CharField(required=False, allow_blank=True)
