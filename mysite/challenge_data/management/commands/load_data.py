import csv
from django.core.management.base import BaseCommand
from challenge_data.models import Creator, Idea
from datetime import datetime, timezone


class Command(BaseCommand):
    help = 'Uploads ideas from a TSV file'

    def handle(self, *args, **kwargs):
        file_path = 'C:/Users/pipea/Desktop/interview/data-analytics-challenges/mysite/fixtures/challenge_data.tsv'

        with open(file_path, newline='', encoding='utf-8') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')

            creator_department_map = {}
            idea_objects = []
            creator_objects = []

            for row in reader:
                idea_id = int(row["Idea ID"])
                submitted_date_time = datetime.strptime(row["Submitted Date/Time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                idea_status = row["Idea Status"]
                creator_id = row["Creator ID"]
                creator_department = row["Creator Department"]
                times_shared = int(row["Number of times the Idea has been shared"])
                votes = int(row["Number of Votes"])
                comments = int(row["Number of Comments"])
                views = int(row["Number of Views"])
                keywords = row["Keywords"]

                creator_department_map[creator_id] = creator_department

                idea_objects.append(
                    Idea(
                        id=idea_id,
                        submitted_date_time=submitted_date_time,
                        status=idea_status,
                        creator_id=creator_id,
                        times_shared=times_shared,
                        votes=votes,
                        comments=comments,
                        views=views,
                        keywords=keywords,
                    )
                )

            for creator_id, creator_department in creator_department_map.items():
                creator_objects.append(
                    Creator(
                        id=creator_id,
                        department=creator_department,
                    )
                )

            Creator.objects.bulk_create(creator_objects)
            Idea.objects.bulk_create(idea_objects)
