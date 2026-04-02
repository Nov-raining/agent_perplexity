import argparse
import sys

from .agent import build_agent, format_report, format_report_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blurry Image OSINT Tracing Agent Demo")
    parser.add_argument("--image", required=True, help="Path to the blurry image")
    parser.add_argument("--mode", default="mock", choices=["mock", "real"], help="Tool mode")
    parser.add_argument("--output", default="text", choices=["text", "json"], help="Output format")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    agent = build_agent(args.mode)
    output = agent.run(args.image)
    if args.output == "json":
        print(format_report_json(output))
    else:
        print(format_report(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
