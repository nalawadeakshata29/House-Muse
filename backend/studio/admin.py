from django.contrib import admin

from .models import DecorItem, Project, WallOption

admin.site.register(Project)


@admin.register(WallOption)
class WallOptionAdmin(admin.ModelAdmin):
    list_display = ("name", "category", "budget_tier", "price_per_sqft")
    list_filter = ("category", "budget_tier")
    search_fields = ("name", "brand")


@admin.register(DecorItem)
class DecorItemAdmin(admin.ModelAdmin):
    list_display = ("name", "category", "budget_tier", "price")
    list_filter = ("category", "budget_tier")
    search_fields = ("name",)

