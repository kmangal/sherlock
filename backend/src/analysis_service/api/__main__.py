import uvicorn

from analysis_service.api.app import create_app
from analysis_service.api.dependencies import get_settings


def main() -> None:
    settings = get_settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
