#!/usr/bin/env python
"""
Submit a CSV of exam responses to the detection API and display results.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

BACKEND_DIR = Path(__file__).parent.parent.absolute()
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "reports" / "detection"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_SIGNIFICANCE_LEVEL = 0.05
DEFAULT_NUM_MONTE_CARLO = 100
DEFAULT_NUM_THRESHOLD_SAMPLES = 100
DEFAULT_MISSING_VALUE = "*"
DEFAULT_POLL_INTERVAL = 10.0

console = Console(force_terminal=True, legacy_windows=True)
app = typer.Typer()


def read_csv(
    input_path: Path, missing_value: str
) -> tuple[list[str], list[list[str]], int]:
    """Read CSV (candidate_id, answer_string) and return candidate_ids, response_matrix, n_categories.

    Each character in answer_string is one question's response.
    """
    df = pd.read_csv(input_path, dtype=str)
    candidate_ids: list[str] = df["candidate_id"].tolist()
    answer_strings: list[str] = df["answer_string"].tolist()

    # Validate all answer strings are the same length
    lengths = {len(s) for s in answer_strings}
    if len(lengths) != 1:
        console.print(
            f"[red]Inconsistent answer string lengths: {sorted(lengths)}[/red]"
        )
        raise typer.Exit(1)

    response_matrix: list[list[str]] = [list(s) for s in answer_strings]

    unique_values = {
        cell
        for row in response_matrix
        for cell in row
        if cell != missing_value
    }
    n_categories = len(unique_values)
    return candidate_ids, response_matrix, n_categories


def health_check(client: httpx.Client) -> None:
    """Abort if server is unreachable."""
    try:
        resp = client.get("/api/v1/health")
        resp.raise_for_status()
    except httpx.HTTPError as e:
        console.print(f"[red]Server health check failed: {e}[/red]")
        raise typer.Exit(1) from e


def submit_job(client: httpx.Client, payload: dict[str, object]) -> str:
    """POST analysis request, return job_id."""
    resp = client.post("/api/v1/analysis", json=payload)
    if resp.status_code >= 400:
        console.print(
            f"[red]Failed to submit job (HTTP {resp.status_code}):[/red]"
        )
        try:
            body = resp.json()
            detail = body.get("detail", body)
            console.print(f"  {detail}")
        except Exception:
            console.print(f"  {resp.text}")
        raise typer.Exit(1)

    data: dict[str, str] = resp.json()
    return data["job_id"]


def poll_job(
    client: httpx.Client,
    job_id: str,
    poll_interval: float,
) -> dict[str, object]:
    """Poll until job completes or fails. Returns final status response."""
    with console.status("[bold cyan]Waiting for results...") as status:
        while True:
            resp = client.get(f"/api/v1/analysis/{job_id}")
            resp.raise_for_status()
            data: dict[str, object] = resp.json()

            job_status = data["status"]
            if job_status in ("completed", "failed"):
                return data

            # Update spinner with progress info
            progress = data.get("progress")
            if isinstance(progress, dict):
                phase = progress.get("phase", "")
                step = progress.get("current_step", "")
                total = progress.get("total_steps")
                msg = progress.get("message", "")
                step_str = f"{step}/{total}" if total else str(step)
                status.update(
                    f"[bold cyan]{phase}[/bold cyan] step {step_str} - {msg}"
                )

            time.sleep(poll_interval)


def print_suspects_table(suspects: list[dict[str, object]]) -> None:
    """Pretty-print suspects as a rich Table."""
    table = Table(title="Flagged Suspects")
    table.add_column("Candidate ID", style="bold")
    table.add_column("Observed Similarity", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("p-value", justify="right")

    for s in suspects:
        p_val = s.get("p_value")
        p_str = f"{p_val:.4f}" if p_val is not None else "-"
        table.add_row(
            str(s["candidate_id"]),
            str(s["observed_similarity"]),
            str(s["detection_threshold"]),
            p_str,
        )

    console.print(table)


def save_report(
    output_dir: Path,
    job_id: str,
    data: dict[str, object],
) -> Path:
    """Save full response JSON to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{timestamp}_{job_id}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


@app.command()
def main(
    input_path: Path = typer.Argument(
        ...,
        help="Path to CSV file with exam responses",
    ),
    url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Server base URL",
    ),
    threshold: int | None = typer.Option(
        None,
        help="Similarity threshold (enables threshold pipeline)",
    ),
    significance_level: float = typer.Option(
        DEFAULT_SIGNIFICANCE_LEVEL,
        help="Significance level for hypothesis tests",
    ),
    num_monte_carlo: int = typer.Option(
        DEFAULT_NUM_MONTE_CARLO,
        help="Number of Monte Carlo simulations",
    ),
    num_threshold_samples: int = typer.Option(
        DEFAULT_NUM_THRESHOLD_SAMPLES,
        help="Number of threshold calibration samples",
    ),
    random_seed: int | None = typer.Option(
        None,
        help="Random seed for reproducibility",
    ),
    missing_value: str = typer.Option(
        DEFAULT_MISSING_VALUE,
        help="String representing missing responses",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        help="Directory for JSON report output",
    ),
    poll_interval: float = typer.Option(
        DEFAULT_POLL_INTERVAL,
        help="Seconds between status polls",
    ),
) -> None:
    """Submit exam CSV to the detection API and display results."""

    # 1. Validate input
    if not input_path.exists():
        console.print(f"[red]File not found: {input_path}[/red]")
        raise typer.Exit(1)
    if input_path.suffix != ".csv":
        console.print("[red]Only .csv files are supported[/red]")
        raise typer.Exit(1)

    # 2. Read CSV
    candidate_ids, response_matrix, n_categories = read_csv(
        input_path, missing_value
    )
    console.print(
        Panel(
            f"[bold]Analysis Request[/bold]\n\n"
            f"File: [cyan]{input_path}[/cyan]\n"
            f"Candidates: [cyan]{len(candidate_ids)}[/cyan]\n"
            f"Items: [cyan]{len(response_matrix[0]) if response_matrix else 0}[/cyan]\n"
            f"Categories: [cyan]{n_categories}[/cyan]\n"
            f"Server: [cyan]{url}[/cyan]",
            title="Configuration",
        )
    )

    # 3. Health check
    client = httpx.Client(base_url=url, timeout=30.0)
    health_check(client)
    console.print("[green]Server is healthy[/green]")

    # 4. Build payload and submit
    payload: dict[str, object] = {
        "exam_dataset": {
            "candidate_ids": candidate_ids,
            "response_matrix": response_matrix,
            "n_categories": n_categories,
            "missing_values": [missing_value],
        },
        "significance_level": significance_level,
        "num_monte_carlo": num_monte_carlo,
        "num_threshold_samples": num_threshold_samples,
    }
    if threshold is not None:
        payload["threshold"] = threshold
    if random_seed is not None:
        payload["random_seed"] = random_seed

    job_id = submit_job(client, payload)
    console.print(f"Job submitted: [cyan]{job_id}[/cyan]")

    # 5. Poll for results
    data = poll_job(client, job_id, poll_interval)

    # 6. Handle result
    if data["status"] == "failed":
        error = data.get("error", {})
        if isinstance(error, dict):
            console.print(
                f"[red]Job failed: {error.get('message', 'unknown error')}[/red]"
            )
        else:
            console.print(f"[red]Job failed: {error}[/red]")
        raise typer.Exit(1)

    # 7. Display suspects
    result = data.get("result", {})
    if isinstance(result, dict):
        suspects = result.get("suspects", [])
        if isinstance(suspects, list) and suspects:
            print_suspects_table(suspects)
        else:
            console.print("[green]No suspects flagged.[/green]")

    # 8. Save report
    report_path = save_report(output_dir, job_id, data)
    console.print(f"Report saved: [cyan]{report_path}[/cyan]")


if __name__ == "__main__":
    app()
