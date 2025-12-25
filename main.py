"""
Discrete Latent Representations - Training Entry Point

This is intentionally minimal. Design your own training loop,
config system, and module structure as you learn.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Train discrete latent models")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    # Your training code here
    print("Ready to learn discrete latent representations!")
    print(f"Config: {args.config}")


if __name__ == "__main__":
    main()
