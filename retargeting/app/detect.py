from pathlib import Path
import typer

detect_app = typer.Typer(help="Commands for detecting critical frames in demonstrations.")

@detect_app.command(help="Detect markers in a video.")
def contacts(demo_path: Path):
    demo_path / "vicon" / "markers.csv"