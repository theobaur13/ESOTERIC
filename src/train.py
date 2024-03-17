import os
from train.relevancy_classification import create_dataset

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, '..', 'data', 'train')
    db_path = os.path.join(current_dir, '..', 'data', 'data.db')
    create_dataset(db_path, output_dir)

if __name__ == "__main__":
    main()