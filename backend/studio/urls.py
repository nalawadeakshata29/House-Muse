from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()
router.register("projects", views.ProjectViewSet, basename="project")

urlpatterns = [
    path("analyse/", views.AnalyseView.as_view(), name="analyse"),
    path("generate-image/", views.GenerateImageView.as_view(), name="generate-image"),
    path("refine-image/", views.RefineImageView.as_view(), name="refine-image"),
    path("", include(router.urls)),
]

