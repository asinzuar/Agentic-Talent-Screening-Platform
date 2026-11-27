from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agentic-talent-screener")
except PackageNotFoundError:
    # The Colab experiment imports the source checkout without installing the
    # application's document/API dependencies. Installed builds still use metadata.
    __version__ = "0+source"


def main() -> None:
    print("Hello from agentic-talent-screener!")
