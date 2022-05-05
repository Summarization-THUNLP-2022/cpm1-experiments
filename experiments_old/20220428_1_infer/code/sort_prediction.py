import argparse


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--file_path", type=str, required=True)
	parser.add_argument("--output_file_path", type=str, required=True)

	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = get_args()
	predictions = []
	with open(args.file_path, 'r') as f, open(args.output_file_path, 'w') as wf:
		for line in f:
			line = line.split('\t')
			predictions.append((line[0], int(line[1])))
		predictions = sorted(predictions, key=lambda x: x[1])
		for prediction in predictions:
			wf.write(prediction[0] + '\n')

