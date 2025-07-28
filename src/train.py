from preprocessing import create_data_loaders

def main():
    train_dir = '../data/train'
    test_dir = '../data/test'

    print("Loading data...")
    train_ds, val_ds = create_data_loaders(train_dir, test_dir)
    print("Train and validation data ready.")

    # Placeholder: Model will go here
    # from model import build_model
    # model = build_model()
    # model.fit(train_ds, validation_data=val_ds, epochs=10)

if __name__ == "__main__":
    main()
