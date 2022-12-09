from PIL import Image

import boto3
import os

def download_images():
    s3 = boto3.resource('s3')

    my_bucket = s3.Bucket('erikharaldson-airbnb-listing-images')

    for file in my_bucket.objects.all():
        s3_filename = file.key
        s3_filename_split = s3_filename.split('/')

        if len(s3_filename_split) == 3:   # This ignores the objects which are empty folders
            cwd = os.getcwd()
            folder_name = 'images/' + s3_filename_split[1]
            final_directory = os.path.join(cwd, folder_name)

            if not os.path.exists(final_directory):
                os.makedirs(final_directory)

            filepath = os.path.join(final_directory, s3_filename_split[2])
            with open(filepath, 'wb') as data:
                my_bucket.download_fileobj(s3_filename, data)

def resize_images():
    cwd = os.getcwd()
    directory = os.path.join(cwd, 'images')
    walk = os.walk(directory)
    _, subdirectories, _ = next(walk)

    images = []

    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(directory, subdirectory)
        _, _, files = next(os.walk(subdirectory_path))

        for file in files:
            image = Image.open(os.path.join(subdirectory_path, file))
            if image.mode == 'RGB':
                images.append((subdirectory+'/'+file, image, image.size[0], image.size[1]))    # File path, image, image width, image height

    heights = [x[-1] for x in images]
    min_height = min(heights)

    for image in images:
        change_in_height = image[-1] / min_height
        new_image = image[1].resize((round(image[-2]/change_in_height), min_height))
        file_path = image[0].split('/')
        processed_img_dir = os.path.join(cwd, 'data/processed_images/' + file_path[0])
        if not os.path.exists(processed_img_dir):
            os.makedirs(processed_img_dir)
        new_image.save(os.path.join(processed_img_dir, file_path[1]))

def prepare_images():
    download_images()
    resize_images()

if __name__ == "__main__":
    prepare_images()