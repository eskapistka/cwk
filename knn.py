import numpy as np
import cv2
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import argparse

class SimplePreprocessor:
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		# store the target image width, height, and interpolation
		# method used when resizing
		self.width = width
		self.height = height
		self.inter = inter
	def preprocess(self, image):
		# resize the image to a fixed size, ignoring the aspect
		# ratio
		return cv2.resize(image, (self.width, self.height),
			interpolation=self.inter)
	

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors
    	# if the preprocessors are None, initialize them as an
        # empty list
        if self.preprocessors is None:
            self.preprocessors = []
			
    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []
        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
        	# load the image and extract the class label assuming
        	# that our path has the following format:
        	# /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            #cv2.imshow(f'{label}', image)
            #cv2.waitKey(0)
        
            # check to see if our preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to
                # the image
                for p in self.preprocessors:
                    image = p.preprocess(image)
            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)
            # show an update every `verbose` images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))
        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))
    

def main():
    print("Hello World!")
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
        help="path to input dataset")
    ap.add_argument("-k", "--neighbors", type=int, default=1,
        help="# of nearest neighbors for classification")
    ap.add_argument("-j", "--jobs", type=int, default=-1,
        help="# of jobs for k-NN distance (-1 uses all available cores)")
    args = vars(ap.parse_args())

    # grab the list of images that we'll be describing
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(args["dataset"]))
    # initialize the image preprocessor, load the dataset from disk,
    # and reshape the data matrix
    #sp = SimplePreprocessor(32, 32)
    sp = SimplePreprocessor(64, 64)
    sdl = SimpleDatasetLoader(preprocessors=[sp])
    (data, labels) = sdl.load(imagePaths, verbose=100)
    #data = data.reshape((data.shape[0], 3072))
    data = data.reshape((data.shape[0], 12288))
    # show some information on memory consumption of the images
    print("[INFO] features matrix: {:.1f}MB".format(
        data.nbytes / (1024 * 1024.0)))
    
    # encode the labels as integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
        test_size=0.25, random_state=42)
    
    # train and evaluate a k-NN classifier on the raw pixel intensities
    print("[INFO] evaluating k-NN classifier...")
    model = KNeighborsClassifier(n_neighbors=args["neighbors"],
        n_jobs=args["jobs"])
    model.fit(trainX, trainY)

    print(classification_report(testY, model.predict(testX), target_names=le.classes_))
    
    ## test area
    # Load the new image
    earth_image_path = "./test/earth_from_dataset_cropped.png"
    earth_image = cv2.imread(earth_image_path)

    # Preprocess the new image
    sp = SimplePreprocessor(64, 64)
    preprocessed_image = sp.preprocess(earth_image)
    preprocessed_image = preprocessed_image.reshape(1, -1)  # Reshape for prediction

    # Predict the class label
    predicted_label = model.predict(preprocessed_image)

    # Decode the label (if labels were encoded during training)
    predicted_label = le.inverse_transform(predicted_label)

    print("Predicted label:", predicted_label)

    ## test area
    # Load the new image
    moon_image_path = "./test/moon_cropped.png"
    moon_image = cv2.imread(moon_image_path)

    # Preprocess the new image
    sp = SimplePreprocessor(64, 64)
    preprocessed_image = sp.preprocess(moon_image)
    preprocessed_image = preprocessed_image.reshape(1, -1)  # Reshape for prediction

    # Predict the class label
    predicted_label = model.predict(preprocessed_image)

    # Decode the label (if labels were encoded during training)
    predicted_label = le.inverse_transform(predicted_label)

    print("Predicted label:", predicted_label)

if __name__ == "__main__":
    main()