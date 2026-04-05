import gzip
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: print_gzip_lines.py <gzip_path> <num_lines>")

    inpath = sys.argv[1]
    n = int(sys.argv[2])

    with gzip.open(inpath, "rt") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            sys.stdout.write(line)


if __name__ == "__main__":
    main()
