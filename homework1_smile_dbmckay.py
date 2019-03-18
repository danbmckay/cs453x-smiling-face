import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
    return np.sum(y==yhat)/float(y.size)

# predictors: set of predictors [[r1,c1,r2,c2], [r1,c1,r2,c2],....]
# X: set of images
# y: ground truths for the images
# n: number of images to run on
def measureAccuracyOfPredictors (predictors, X, y, n):
    # using the predictors, we see how accurate it is
    # output is the average prediction for all 5 features
    m = len(predictors)
    # guesses based on predictors
    yhat = []
    # arrays for each different feature
    pred_arrays = []
    for j in range(0, len(predictors)):
        r1, c1, r2, c2 = predictors[j]
        pred_arrays.append(X[:n, r1, c1] > X[:n, r2, c2])
    # sum trues for each feature, and divide by number of features and check if its above .5
    yhat = np.sum(pred_arrays, axis=0)/float(m) > .5
    # return accuracy
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
    print('Training accuracy for ', n, ' examples: ', max_accuracy)

    test_acc = measureAccuracyOfPredictors(predictors, testingFaces, testingLabels, len(testingFaces))

    print('Testing accuracy for ', n, ' examples: ', test_acc)

    show = True
    if show:
        # Show an arbitrary test image in grayscale
        im = testingFaces[0,:,:]
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        # differentiate different features
        edgeColors = ['r', 'b', 'g', 'y', 'c']
        for i in range(0,len(predictors)):
            # Show r1,c1
            pred = predictors[i]
            rect = patches.Rectangle((pred[1], pred[0]),1,1,linewidth=2,edgecolor=edgeColors[i],facecolor='none')
            ax.add_patch(rect)
            # show r2, c2
            rect = patches.Rectangle((pred[3], pred[2]), 1, 1, linewidth=2, edgecolor=edgeColors[i], facecolor='none')
            ax.add_patch(rect)
        # Show r2,c2
        # rect = patches.Rectangle((c2,r2),1,1,linewidth=2,edgecolor='b',facecolor='none')
        # ax.add_patch(rect)
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

    # for n in range(400, 2400, 400):
    #     stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels, n)
    stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels, 2000)
