import argparse

from dataset import DatasetFactory


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--file_path", type=str, required=True)
	parser.add_argument("--output_file_path", type=str, required=True)
	parser.add_argument("--dataset", type=str, required=True)

	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = get_args()
	dataset = DatasetFactory.get_dataset(args.dataset)
	dataset.read_data(args.file_path)
	with open(args.output_file_path, 'w') as f:
		for data in dataset._data:
			f.write(data['summary'] + '\n')

