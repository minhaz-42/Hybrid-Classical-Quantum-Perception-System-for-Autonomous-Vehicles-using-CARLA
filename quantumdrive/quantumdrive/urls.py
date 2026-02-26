"""
QuantumDrive — Root URL Configuration
======================================
Routes the admin panel plus all core app URLs (pages + API).
"""

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("core.urls")),
]
