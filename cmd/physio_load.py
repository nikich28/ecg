import argparse
import logging
from utils.ecg_record import download_database


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--database',
                        default='afdb')

    parser.add_argument('-f', '--data_folder',
                        default='records')

    kwargs = vars(parser.parse_args())

    download_database(**kwargs)
