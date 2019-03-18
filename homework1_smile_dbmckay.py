import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
    return np.sum(y==yhat)/float(y.size)

# predictors: set of predictors [[r1,c1,r2,c2], [r1,c1,r2,c2]]
# X: set of images
# y: ground truths for the images
# n: number of images to run on
def measureAccuracyOfPredictors (predictors, X, y, n):
    # using the predictors, we see how accurate it is
    # each feature is I[xr1c1 > xr2c2] where I[] is either 1 for true or 0 for false
    # output is the average prediction for all 5 features
    m = len(predictors)
    yhat = []
    # TODO try and vectorize
    for i in range(0, n):
        smile_predict_cnt = 0
        for j in range(0, len(predictors)):
            r1, c1, r2, c2 = predictors[j]
            smile_predict_cnt += (X[i][r1][c1] > X[i][r2][c2])
        # determine smile or no smile for that image
        yhat.append((smile_predict_cnt/float(m)) > .5)

    return fPC(y[:n], yhat)

# n: number of images to run this on
def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels, n):
    predictors = []
    # number of features
    m = 5
    # length of image (24x24)
    img_size = 24
    for j in range(0, m):
        max_accuracy = 0
        for r1 in range(0, img_size):
            for c1 in range(0, img_size):
                for r2 in range(0, img_size):
                    for c2 in range(0, img_size):
                        if not (r1==r2 and c1==c2):
                            predictors.append([r1, c1, r2, c2])
                            temp = measureAccuracyOfPredictors(predictors, trainingFaces, trainingLabels, n)
                            if temp > max_accuracy:
                                max_accuracy = temp
                                best_predictor = [r1, c1, r2, c2]
                            del predictors[-1]
        predictors.append(best_predictor)
        print('accuracy: ', max_accuracy)



    show = False
    if show:
        # Show an arbitrary test image in grayscale
        im = testingFaces[0,:,:]
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        # Show r1,c1
        rect = patches.Rectangle((c1,r1),1,1,linewidth=2,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle((c2,r2),1,1,linewidth=2,edgecolor='b',facecolor='none')
        ax.add_patch(rect)
        # Display the merged result
        plt.show()

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels, 200)
    # TODO remove
    y = np.array([1,0,0])
    yhat = np.array([1,1,1])
    print(fPC(y,yhat))
