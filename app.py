# app.py
from src.xpu import XPU


def main():
    processor = XPU()
    processor.benchmark(size=2048)


if __name__ == "__main__":
    main()
