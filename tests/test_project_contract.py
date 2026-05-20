import csv
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectContractTest(unittest.TestCase):
    def test_required_files_exist(self) -> None:
        required = [
            "docker-compose.yml",
            ".env.example",
            "api/Dockerfile",
            "api/app/main.py",
            "worker/Dockerfile",
            "worker/automl_worker/tasks.py",
            "README.md",
            "submission.json",
        ]
        for relative in required:
            self.assertTrue((ROOT / relative).exists(), relative)

    def test_docker_compose_services_have_healthchecks(self) -> None:
        text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        for service in ["redis", "mlflow", "api", "worker"]:
            pattern = rf"\n  {service}:\n(?P<body>(?:    .*\n)+)"
            match = re.search(pattern, text)
            self.assertIsNotNone(match, service)
            self.assertIn("healthcheck:", match.group("body"))
        self.assertIn("mlflow_data:", text)

    def test_env_example_documents_required_variables(self) -> None:
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        for variable in ["CELERY_BROKER_URL", "MLFLOW_TRACKING_URI", "API_PORT"]:
            self.assertIsNotNone(re.search(rf"^{variable}=", text, re.MULTILINE), variable)

    def test_leaderboard_contract_columns_are_named_in_worker(self) -> None:
        text = (ROOT / "worker/automl_worker/training.py").read_text(encoding="utf-8")
        for column in ["model_id", "model_type", "mean_cv_score", "metric"]:
            self.assertIn(column, text)
        self.assertIn("VotingEnsemble", text)
        self.assertIn("StackingEnsemble", text)

    def test_sample_dataset_is_valid_csv(self) -> None:
        with (ROOT / "examples/iris_sample.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(rows), 30)
        self.assertIn("species", rows[0])

    def test_submission_json_is_valid(self) -> None:
        payload = json.loads((ROOT / "submission.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["api_url"], "http://localhost:8000")

    def test_console_links_to_reports_not_raw_deployment_code(self) -> None:
        html = (ROOT / "api/app/static/index.html").read_text(encoding="utf-8")
        js = (ROOT / "api/app/static/app.js").read_text(encoding="utf-8")
        self.assertIn("Open summary report", html)
        self.assertIn("Open data profile", html)
        self.assertNotIn("deployment-link", html)
        self.assertNotIn("leaderboard-link", html)
        self.assertNotIn("deployment/main.py", js)

    def test_summary_report_includes_download_buttons(self) -> None:
        text = (ROOT / "worker/automl_worker/reporting.py").read_text(encoding="utf-8")
        for artifact in [
            "leaderboard.csv?download=1",
            "best_model.pkl?download=1",
            "deployment/main.py?download=1",
            "deployment/Dockerfile?download=1",
        ]:
            self.assertIn(artifact, text)


if __name__ == "__main__":
    unittest.main()
