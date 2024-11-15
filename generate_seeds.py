

if __name__ == "__main__":
    num_of_seed_pairs = 32
    train_seed_start, test_seed_start = 53705, 50735
    with open("seeds.txt", "w") as f:
        for index in range(num_of_seed_pairs):
            f.write(f"{str(index + train_seed_start)},{str(index + test_seed_start)}\n")
