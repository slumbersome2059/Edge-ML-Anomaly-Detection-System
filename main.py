"""Legacy entry point that directs users to the reproducible Makefile workflow."""

from __future__ import annotations


def main() -> None:
    print("Local model training is disabled. Run `make prepare` and `make colab-bundle`, then follow COLAB.md.")


if __name__ == "__main__":
    main()
