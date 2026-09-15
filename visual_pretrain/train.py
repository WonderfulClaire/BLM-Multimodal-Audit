"""Training entrypoint; see train_manifest for the complete image/region objective."""

from .train_manifest import main, train

__all__ = ["train"]
if __name__ == "__main__":
    main()
