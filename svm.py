import os
import cv2
import joblib
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
#test
def load_preprocess_data(folder_path, label, size=(128, 128)):
    features, labels = [], []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, size)
            features.append(image.flatten())
            labels.append(label)
    return features, labels

#finds an optimal hyperplane to separate data points of different classes
def svm_grid_search(X_train, y_train):#used to find best hyperplane

    #https://www.analyticsvidhya.com/blog/2021/10/support-vector-machinessvm-a-complete-guide-for-beginners/

    param_grid = {
        'C': [0.1, 1, 10, 100],# to find a margin separating hyperplane. smaller+larger margin, greater+smaller margin
                                #smaller marginde hata payı daha fazla

        'gamma': ['scale', 'auto'],  #gamma controls the width of the gaussian kernel, a parameter for non-linear hyperplanes

        'kernel': ['linear', 'rbf', 'poly'] #rbf gaussian kerneli
    }

    #gamma is a coefficient for the kernel function
    #auto, it uses the value 1/n_features,,value of based on the inverse of the number of features
    #scale adjust the value of gamma more precisely based on the actual data you have,
    # 1/(n_features * X.var())
    #var, refers to how variable the values of each feature are in the data set.

    # 'linear'  use a linear hyperplane.
    # 'rbf' which stands for Radial Basis Function, used for non-linear hyperplane.
    # 'poly' to find the hyperplane in the form of a polynomial


    svm = SVC()


    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def save_model_and_scaler(scaler, svm_model, model_filename="svm_model3.pkl", scaler_filename="scaler3.pkl"):
    joblib.dump(svm_model, model_filename)
    joblib.dump(scaler, scaler_filename)

def main():
    folder_paths = [
        'dataset/Plant_leave_diseases_dataset_with_augmentation/Apple___Apple_scab',
        'dataset/Plant_leave_diseases_dataset_with_augmentation/Apple___Black_rot',
        'dataset/Plant_leave_diseases_dataset_with_augmentation/Apple___Cedar_apple_rust',
        'dataset/Plant_leave_diseases_dataset_with_augmentation/Apple___healthy',

        'dataset/Plant_leave_diseases_dataset_without_augmentation/Apple___Apple_scab',
        'dataset/Plant_leave_diseases_dataset_without_augmentation/Apple___Black_rot',
        'dataset/Plant_leave_diseases_dataset_without_augmentation/Apple___Cedar_apple_rust',
        'dataset/Plant_leave_diseases_dataset_without_augmentation/Apple___healthy',




        'dataset/Plant_leave_diseases_dataset_with_augmentation/Cherry___healthy',
        'dataset/Plant_leave_diseases_dataset_with_augmentation/Cherry___Powdery_mildew',

        'dataset/Plant_leave_diseases_dataset_without_augmentation/Cherry___healthy',
        'dataset/Plant_leave_diseases_dataset_without_augmentation/Cherry___Powdery_mildew',


        'dataset/Plant_leave_diseases_dataset_with_augmentation/Corn___Common_rust',
        'dataset/Plant_leave_diseases_dataset_with_augmentation/Corn___healthy',

        'dataset/Plant_leave_diseases_dataset_without_augmentation/Corn___Common_rust',
        'dataset/Plant_leave_diseases_dataset_without_augmentation/Corn___healthy'

    ]

    all_features, all_labels = [], []
    total_images = 0
    for folder_number, folder_path in enumerate(folder_paths, start=1):
        print(f"Processing folder {folder_number}/{len(folder_paths)}: {folder_path}")

        # Count the number of images in the current class folder
        num_images_in_folder = sum(1 for _ in os.listdir(folder_path) if _.lower().endswith(('.png', '.jpg', '.jpeg')))
        total_images += num_images_in_folder

        # Assign label 1 for 'healthy' class, and 0 otherwise
        features, labels = load_preprocess_data(folder_path, label=1 if "healthy" in folder_path else 0)
        all_features.extend(features)#  Each element is a 1D NumPy array representing the feature vector of an image
        all_labels.extend(labels) # Each element is an integer label (0 or 1) corresponding to the class of an images

    print(f"Total number of images processed: {total_images}")

    # Check if any features were extracted
    if not all_features:
        print("No features extracted. Check the dataset and feature extraction process.")
        return

    # Check for consistent feature vector lengths
    feature_lengths = [len(feature) for feature in all_features]
    unique_lengths = set(feature_lengths)  # Collect the unique lengths of feature vectors
    # This block checks whether all the feature vectors in all_features have the same length.
    # Inconsistent lengths could lead to issues during normalization or training.

    # all feature vectors should have same length
    if len(unique_lengths) != 1:
        print(f"Inconsistent feature vector lengths: {unique_lengths}")
        return

    scaler = StandardScaler()
    # The purpose of standardization is to bring all features to a similar scale

    # fit_transform() computes the mean and standard deviation necessary for scaling and then applies the transformation to the input data

    # fit calculates, transform apply

    # making it easier to compare and interpret their importance in a machine learning model
    # https://www.analyticsvidhya.com/blog/2021/04/difference-between-fit-transform-fit_transform-methods-in-scikit-learn-with-python-code/#:~:text=The%20fit_transform()%20method%20is,()%20and%20transform()%20separately.

    all_features_normalized = scaler.fit_transform(all_features)

    # randomly selecting examples from the minority class, with replacement, and adding them to the training dataset. # https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
    ros = RandomOverSampler(random_state=42)# https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html
    #our data set is unbalance so we are doing over sampling.

    # over sampling to the minority class to have same size as the majority class
    X_res, y_res = ros.fit_resample(all_features_normalized, all_labels)
    # It analyzes the class distribution in all_labels,
    # then randomly selects instances from the minority class and duplicates them until the class distribution is balanced.


    # Split the dataset into training and testing sets using sklearn library
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Perform grid search for hyperparameter tuning
    best_svm = svm_grid_search(X_train, y_train)

    # Train the classifier with the best parameters
    best_svm.fit(X_train, y_train)



    # Save the trained model and scaler
    save_model_and_scaler(scaler, best_svm)

    # Predict on the test data using sklearn library
    # function enables us to predict the labels of the data values on the basis of the trained model.
    # https://www.askpython.com/python/examples/python-predict-function
    y_pred = best_svm.predict(X_test)

    # Evaluate the model provided by sk learn
    # classification report https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    # confusion matrix https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    print("Classification Report:", classification_report(y_test, y_pred))
    print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

    # Classification Report
    # precision: The ratio of correctly predicted positive observations to the total predicted positives.
    # recall: The ratio of correctly predicted positive observations to the all observations in actual class.
    # f1-score: The weighted average of precision and recall. It is a good way to show that a classifier has a good value for both recall and precision.
    # support: The number of actual occurrences of the class in the specified dataset.

    # Confusion matrix
    # [[True Negative   False Positive]
    # [False Negative  True Positive]]

    # True Negative (TN): Instances that were correctly predicted as the negative class.
    # False Positive (FP): Instances that were incorrectly predicted as the positive class.
    # False Negative (FN): Instances that were incorrectly predicted as the negative class.
    # True Positive (TP): Instances that were correctly predicted as the positive class.
if __name__ == "__main__":
    main()
