from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Face detection step (placeholder)")
	return parser


def main() -> None:
	build_parser().parse_args()
	print(
		"Face detection chưa được implement. "
		"TODO: dùng RetinaFace để detect bbox, crop mở rộng 15-20%, resize 224x224."
	)


if __name__ == "__main__":
	main()