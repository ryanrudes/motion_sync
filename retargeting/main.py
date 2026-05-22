import typer
from retargeting.app import convert_app, detect_app, fkin_app, model_app, sync_app

app = typer.Typer(help="Retargeting CLI")

app.add_typer(convert_app, name="convert")
app.add_typer(model_app, name="model")
app.add_typer(detect_app, name="detect")
app.add_typer(fkin_app, name="fkin")
app.add_typer(sync_app, name="sync")

def main() -> None:
    """Console script entrypoint."""
    app()


if __name__ == "__main__":
    main()
