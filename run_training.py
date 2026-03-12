import argparse
import os
import logging

from main import ScorePredictor
import create_dummy_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def main():
    parser = argparse.ArgumentParser(description='Train ScorePredictor or create dummy model')
    parser.add_argument('--data', '-d', help='Path to CSV data file for training')
    parser.add_argument('--text-column', default='Text', help='Name of text column in CSV')
    parser.add_argument('--score-column', default='Score', help='Name of score column in CSV')
    parser.add_argument('--save', '-s', default='model.joblib', help='Path to save trained model')
    parser.add_argument('--rows', '-r', type=int, default=None, help='Number of rows to read from CSV')
    parser.add_argument('--create-dummy', action='store_true', help='Create a small dummy model and save it')
    parser.add_argument('--dummy-path', default='model.joblib', help='Path to save dummy model')

    args = parser.parse_args()

    if args.create_dummy:
        logging.info('Creating dummy model...')
        create_dummy_model.create_and_save_dummy_model(path=args.dummy_path)
        return

    if not args.data:
        logging.error('No data file provided. Use --data <path> or --create-dummy')
        return

    if not os.path.exists(args.data):
        logging.error('Data file not found: %s', args.data)
        return

    predictor = ScorePredictor()
    logging.info('Starting training using data: %s', args.data)
    predictor.train_and_evaluate(
        filepath=args.data,
        text_column=args.text_column,
        score_column=args.score_column,
        model_save_path=args.save,
        num_rows=args.rows
    )
    logging.info('Training finished. Model saved to %s', args.save)


if __name__ == '__main__':
    main()
