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

## Regression
### Linear Regression

- In this section of the project I am fitting a linear regression model to the airbnb dataset to predict nightly price. 

- A first baseline model was created as a simple sklearn linear regression model without regularization which produced a root mean squared error of 115.95 on the validation dataset.

- After this a grid search over the hyperparameters: penalty, alpha and max_iter was completed using sklearn's GridSearchCV module. This grid search therefore looked at both Linear regression models with no regularization, L1 regularization, and L2 regularization. The best model produced from this Cross-Validation search was a Ridge regression model with alpha = 0.1 and 100000 iterations. The mean RMSE score on the validation sets was 102.43, this result was better than that without regularization which is to be expected since the regularization increases the generalization of the model.

- The best linear regression model and its score metrics were saved in files in the model/regression/linear_regression folder to be able to compare to future models. 

- The RMSE score however is very high and it gave only and R^2 score of ~0.445 which is not a very good fit. Therefore, I hope to see an improvement when looking at other regression models (non-linear).

### All regression models

- After running a simple linear regression model it is now time to test other regression techniques.

- To do this the following models were looked at:
    - SGDRegressor()
    - AdaBoostRegressor()
    - GradientBoostingRegressor()
    - RandomForestRegressor()
    - DecisionTreeRegressor()
    - LinearSVR()
    - SVR()

- A cross validation over some important hyperparameters was conducted on each model. The models, metrics and best hyperparameters were saved in individual folders.

- Finally a function was created to find the best model from each of the best models of each algorithm with the best hyperparameters.

- The best model and its hyperparameters was a random forest regressor as seen below...

```
RandomForestRegressor(max_depth=100, 
                      max_features='sqrt', 
                      min_samples_leaf=2,
                      n_estimators=20)
```

- For this model the mean validation RMSE (Root Mean Squared Error) was 99.33. This is better than the linear regression model but is still quite low.

- There are more regression models I would like to look at such as: bayesian regression which has not been looked at in the notes and also neural networks (which will ultimately be used later one)

### All classification models

- To test out different classification models, I looked at the numerical data from before and used the 'Category' column of the dataset as the output label. This label consists of different categories of listing such as treehouse and chalet. 

- The classification models that were looked at:
    - LogisticRegression()
    - KNeighborsClassifier()
    - GradientBoostingClassifier()
    - RandomForestClassifier()
    - DecisionTreeClassifier()
    - LinearSVC()
    - SVC()
    - GaussianProcessClassifier()

- A similar process was carried out as before by comparing the best cross validated model but in this case, the mean validation accuracy score was used to compare models.

- The best classification model was found to be:

```
GradientBoostingClassifier(max_depth=1,
                           min_samples_leaf=5)
```

- For this model the mean validation accuracy score is 0.401. This accuracy is better than logistic regression which achieved a mean validation accuracy score of 0.389 but it is still quite low and would not constitute a usable model. The model is clearly better than a random guesser; which since there are 5 classes would tend to achieve an accuracy score of 0.2, however it is still a very low accuracy. Hopefully the neural network used later on will be able to achieve better scores than the classifier models used so far.

- In the future I would like to look more at using a neural network for classification as well as a Naive Bayes classification model.