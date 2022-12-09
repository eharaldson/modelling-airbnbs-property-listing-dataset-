# Modelling airbnbs listings dataset

## Data preparation
### Tabular data
- In the tabular_data file I have created functions to clean the airbnb table data. The functions could be reused for similar tabular data. The tabular data has 988 samples and 20 columns as seen when using `df.info()`:

![alt text](./readme_images/listing_info.png)

- The output from `df.head()` is seen below:

![alt text](./readme_images/listing_head.png)

- Three key steps were completed to clean the tabular data.
    - The first was to remove the rows in which the rating data was missing. The first line in the function is to remove the 'Unamed: 19' column which contains only null values which was created in reading the csv due to an error of an extra comma in the csv file. The function:
    ```
    def remove_rows_with_missing_ratings(df_null):
        df_null = df_null.iloc[:,:-1]
        df = df_null.dropna(subset=['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating'])
        return df.copy()
    ```
    - The second was to fix the description column. The problem with this column was that each element contained a list in the form of a string (i.e. "[...]"). Therefore the literal_eval function from the ast module was used to turn these values into lists. Then a string was created from the list so that the description is a single string. The function:
    ```
    def description_func(x):
        ls = literal_eval(x)
        ls.remove('About this space')
        if '' in ls:
            ls.remove('')
            return ''.join(ls)
        else:
            return ''.join(ls)

    def combine_description_strings(df):
        df = df.copy().dropna(subset={'Description'})
        df['Description'] = df['Description'].apply(description_func)
        return df.copy()
    ```
    - The third step is to set default values for certain columns where there are missing values. The function: 
    ```
    def set_default_feature_value(df):
        df[['guests', 'beds', 'bathrooms', 'bedrooms']] = df[['guests', 'beds', 'bathrooms', 'bedrooms']].fillna(1)
        return df.copy()
    ```

### Images
- The images for the airbnb property samples were prepared in the prepare_image_data.py file. In this file the images were downloaded from the cloud (an s3 bucket from AWS) and placed in a folder. The function used to do this is seen below:
```
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
```
- The images are then processed to all have the same height as the image with the smallest height in the entire dataset while keeping the aspect ratio of each image constant. The function used to do this is seen below:
```
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
```