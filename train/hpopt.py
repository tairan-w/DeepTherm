"""Hyperparameter search is intentionally external to the deterministic trainer."""


def main():
    raise SystemExit(
        "The incompatible Chemprop/Ray Tune wrapper was removed. Run train.py repeatedly with "
        "explicit --hidden-size, --depth, --attention-heads, --dropout, and learning-rate values."
    )


if __name__ == "__main__":
    main()
