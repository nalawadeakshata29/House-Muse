"""
Usage:
    python manage.py load_catalog path/to/catalog_seed.csv

Loads the combined wall-option + décor-item dataset into the database.
Safe to re-run — clears and reloads each time (this is seed/reference
data, not user data, so there's no harm in replacing it wholesale).
"""

import csv

from django.core.management.base import BaseCommand, CommandError

from studio.models import DecorItem, WallOption


class Command(BaseCommand):
    help = "Load the House Muse catalog seed CSV into WallOption and DecorItem tables"

    def add_arguments(self, parser):
        parser.add_argument("csv_path", type=str)

    def handle(self, *args, **options):
        path = options["csv_path"]

        try:
            f = open(path, newline="", encoding="utf-8")
        except FileNotFoundError:
            raise CommandError(f"No such file: {path}")

        wall_created, decor_created = 0, 0

        with f:
            reader = csv.DictReader(f)
            # Group rows by (item_type, name-without-tier-suffix isn't needed —
            # style_tags is one style per row here, so we merge duplicates by name+category)
            wall_rows, decor_rows = {}, {}

            for row in reader:
                style = row["style_tags"]
                key = (row["category"], row["name"])

                if row["item_type"] == "wall_option":
                    if key not in wall_rows:
                        wall_rows[key] = {**row, "style_tags": []}
                    wall_rows[key]["style_tags"].append(style)
                else:
                    if key not in decor_rows:
                        decor_rows[key] = {**row, "style_tags": []}
                    decor_rows[key]["style_tags"].append(style)

        WallOption.objects.all().delete()
        for row in wall_rows.values():
            WallOption.objects.create(
                name=row["name"],
                category=row["category"],
                hex_color=row["hex_color"],
                pattern=row["pattern"],
                brand=row["brand"],
                price_per_sqft=float(row["price"]),
                style_tags=row["style_tags"],
                budget_tier=row["budget_tier"],
                image_url=row["image_url"],
            )
            wall_created += 1

        DecorItem.objects.all().delete()
        for row in decor_rows.values():
            DecorItem.objects.create(
                name=row["name"],
                category=row["category"],
                price=float(row["price"]),
                style_tags=row["style_tags"],
                budget_tier=row["budget_tier"],
                image_url=row["image_url"],
                source_url="",
            )
            decor_created += 1

        self.stdout.write(self.style.SUCCESS(
            f"Loaded {wall_created} wall options and {decor_created} decor items."
        ))
