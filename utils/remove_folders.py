import os

data_path = os.getcwd() + "/data/png_export/"


def remove_empty_folders(path):
    print("\nRemoving the following empty folders ---")
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            if len(os.listdir(os.path.join(root, name))) == 0:
                print(os.path.join(root, name))
                os.rmdir(os.path.join(root, name))


def main():
    remove_empty_folders(data_path)


if __name__ == "__main__":
    main()

