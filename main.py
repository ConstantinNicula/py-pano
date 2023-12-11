import argparse
import image_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_src_folder", help="Path to the folder containing input images")
    parser.add_argument("out_folder", help="Path where result will be stored")
    args = parser.parse_args()

    data, exif = image_loader.load_batch(args.img_src_folder)
    print (exif)

if __name__ == "__main__":
    main()