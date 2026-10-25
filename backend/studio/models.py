from django.contrib.auth.models import User
from django.db import models


class Project(models.Model):
    """
    One saved 'run' of the Studio flow: the inputs the user picked
    (room type, dimensions, budget, styles) plus the JSON result that
    came back from the ml/ service (palette, recommendations, etc).
    """

    ROOM_TYPE_CHOICES = [
        ("living_room", "Living Room"),
        ("bedroom", "Bedroom"),
        ("kitchen", "Kitchen"),
        ("dining_room", "Dining Room"),
        ("bathroom", "Bathroom"),
        ("study_room", "Study Room"),
    ]

    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="projects")
    name = models.CharField(max_length=255, blank=True)

    room_type = models.CharField(max_length=30, choices=ROOM_TYPE_CHOICES, default="living_room")
    length = models.FloatField()
    width = models.FloatField()
    height = models.FloatField()
    styles = models.JSONField(default=list)
    budget = models.FloatField()
    scope = models.CharField(max_length=20, default="both")
    num_rooms = models.PositiveIntegerField(default=1)
    harmony = models.CharField(max_length=30, default="analogous")

    source_image = models.ImageField(upload_to="projects/source/", blank=True, null=True)
    generated_image_url = models.TextField(blank=True, null=True)

    result = models.JSONField(blank=True, null=True)  # raw response from ml/'s /api/analyse

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name or f"{self.get_room_type_display()} — {self.owner.username}"


# ── Catalog: the structured data the recommendation engine queries ─────────
# This is deliberately plain Postgres/Django ORM — a catalog this size
# (hundreds to low tens-of-thousands of rows) is exactly what a relational
# table is for. No need for anything heavier.

STYLE_CHOICES = [
    ("coastal", "Coastal"),
    ("industrial", "Industrial"),
    ("contemporary_indian", "Contemporary Indian"),
    ("vintage_indian", "Vintage Indian"),
    ("scandinavian", "Scandinavian"),
    ("eclectic", "Eclectic"),
    ("shabby_chic_style", "Shabby Chic"),
    ("mid_century_modern", "Mid-Century Modern"),
    ("victorian", "Victorian"),
]

BUDGET_TIER_CHOICES = [
    ("budget", "Budget"),
    ("mid", "Mid-Range"),
    ("premium", "Premium"),
]


class WallOption(models.Model):
    """A paint colour or wallpaper option that can be suggested for a wall."""

    CATEGORY_CHOICES = [
        ("paint", "Paint"),
        ("wallpaper", "Wallpaper"),
    ]

    name = models.CharField(max_length=150)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    hex_color = models.CharField(max_length=7, blank=True)  # e.g. "#C9A84C" — paint only
    pattern = models.CharField(max_length=100, blank=True)  # e.g. "geometric", "botanical" — wallpaper only
    brand = models.CharField(max_length=100, blank=True)
    price_per_sqft = models.FloatField(help_text="₹ per sqft")
    style_tags = models.JSONField(default=list, help_text="Which STYLE_CHOICES this fits")
    budget_tier = models.CharField(max_length=10, choices=BUDGET_TIER_CHOICES, default="mid")
    image_url = models.URLField(blank=True)

    class Meta:
        ordering = ["category", "name"]

    def __str__(self):
        return f"{self.name} ({self.category})"


class DecorItem(models.Model):
    """A furniture/décor product the recommendation engine can suggest."""

    CATEGORY_CHOICES = [
        ("rug", "Rug"),
        ("cushion", "Cushion / Throw Pillow"),
        ("curtain", "Curtain / Drape"),
        ("wall_art", "Wall Art"),
        ("lighting", "Lighting"),
        ("plant", "Plant / Planter"),
        ("mirror", "Mirror"),
        ("storage", "Storage"),
        ("seating", "Seating"),
        ("table", "Table"),
    ]

    name = models.CharField(max_length=200)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    price = models.FloatField(help_text="₹")
    style_tags = models.JSONField(default=list)
    budget_tier = models.CharField(max_length=10, choices=BUDGET_TIER_CHOICES, default="mid")
    image_url = models.URLField(blank=True)
    source_url = models.URLField(blank=True, help_text="Where to actually buy it, if known")

    class Meta:
        ordering = ["category", "price"]

    def __str__(self):
        return f"{self.name} — ₹{self.price:.0f}"


class ProjectRefinement(models.Model):
    """
    One refinement step applied to a Project's generated design.
    Stores the user instruction and resulting image, forming
    an ordered history where each edit builds on the previous one.
    """

    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="refinements")
    instruction = models.TextField()
    resulting_image = models.ImageField(upload_to="projects/refinements/", blank=True, null=True)
    resulting_image_url = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"Refinement {self.id} for Project {self.project_id}: {self.instruction[:30]}"

    def get_resulting_image(self, request=None):
        if self.resulting_image:
            if request:
                return request.build_absolute_uri(self.resulting_image.url)
            return self.resulting_image.url
        return self.resulting_image_url or ""

