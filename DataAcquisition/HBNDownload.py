import os, argparse, pathlib
import openneuro


def main(args):
	openneuro.download(dataset=args.openneuro_accession_number, target_dir=args.output_dir, max_concurrent_downloads=args.max_concurrent_downloads)

def parse_args():
	# setup arg parser
	parser = argparse.ArgumentParser()

	# INPUT arguments 
	parser.add_argument(
		"--openneuro_accession_number", type=str, help="OpenNeuro accession number", required=True
	)

	parser.add_argument(
		"--max_concurrent_downloads", type=int, help="Maximum number of concurrent downloads", default=5
	)

	parser.add_argument(
		"--output_dir", type=str, help="Output directory"
	)

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()
	main(args)