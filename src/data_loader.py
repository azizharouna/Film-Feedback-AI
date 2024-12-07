import os

def load_data(data_dir):
    """Loads movie review data from the specified directory.

    Args:
        data_dir: The path to the data directory.

    Returns:
        A tuple containing lists of positive and negative reviews.
    """
    pos_reviews = []
    neg_reviews = []

    pos_dir = os.path.join(data_dir, "pos")
    neg_dir = os.path.join(data_dir, "neg")

    for filename in os.listdir(pos_dir):
        with open(os.path.join(pos_dir, filename), "r", encoding="utf-8") as f:
            pos_reviews.append(f.read())

    for filename in os.listdir(neg_dir):
        with open(os.path.join(neg_dir, filename), "r", encoding="utf-8") as f:
            neg_reviews.append(f.read())

    return pos_reviews, neg_reviews

if __name__ == "__main__":
    data_dir = "data/train"  # Example usage
    pos_reviews, neg_reviews = load_data(data_dir)
    print(f"Loaded {len(pos_reviews)} positive reviews and {len(neg_reviews)} negative reviews.")
