
#  python loop.py --prototxt squeezenet_v1.0.prototxt  --labels label.txt


# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import imutils
import matplotlib.pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
#ap.add_argument("-m", "--model", required=True,
#	help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=True,
	help="path to  labels (i.e., LABLES)")
args = vars(ap.parse_args())
iterations=[]
per=[]
for q in range(2,2100,2):	#adjust the range function based on number of caffemodel file
	# load the class labels from disk
	iterations.append(q)
	rows = open(args["labels"]).read().strip().split("\n")
	classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
	caffe_model='/home/student/Desktop/PC1_OPTIMIZER_NEW/Adagrad/train_iter_'+str(q)+'.caffemodel'  #path to the caffemodel files
	basedir = "/home/student/Desktop/PC1_OPTIMIZER_NEW_Results/all/" #path to the test images
	alls=[]
	elephant=[]
	others=[]
	print("The result for model no",q,'is:')
	for j in range(1,461):		#adjust the range function based on number of test images	

		path=basedir+str(j)+'.jpg'

	# load the input image from disk
		image = cv2.imread(path)
		image = imutils.resize(image, width=400)
	# our CNN requires fixed spatial dimensions for our input image(s)
	# so we need to ensure it is resized to 224x224 pixels while
	# performing mean subtraction (104, 117, 123) to normalize the input;
	# after executing this command our "blob" now has the shape:
	# (1, 3, 224, 224)
		blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

	# load our serialized model from disk
		#print("[INFO] loading model...")
		net = cv2.dnn.readNetFromCaffe(args["prototxt"], caffe_model)

	# set the blob as input to the network and perform a forward-pass to
	# obtain our output classification
		net.setInput(blob)
		#start = time.time()
		preds = net.forward()
		#end = time.time()
		#print("[INFO] classification took {:.5} seconds".format(end - start))

	# sort the indexes of the probabilities in descending order (higher
	# probabilitiy first) and grab the top-5 predictions
		preds = preds.reshape((1, len(classes)))
		idxs = np.argsort(preds[0])[::-1]#[:5]

	# loop over the top-5 predictions and display them
		for (i, idx) in enumerate(idxs):
		# draw the top prediction on the input image
			if i == 0:
				text = "Label: {}, {:.2f}%".format(classes[idx],
					preds[0][idx] * 100)
				alls.append((classes[idx]))
				cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
					0.7, (0, 0, 255), 2)
	for j in range(1,461):  #adjust the range function based on number of test images	
		if j <=225:
			elephant.append((alls[j-1]))
		else:
			others.append((alls[j-1]))
	print(len(elephant))
	print(len(others))
	c1=elephant.count('others')
	c2=others.count('elephant')
	percent=(100-(((c1+c2)/460)*100))     #adjust the value based on number of test images	
	print("The number of others in elephant is ",c1)
	print("The number of elepahnt in others is ",c2)
	print("The percent of accuracy is:",percent)
	
	per.append((percent/100))
	strj=str(q)
	strc1=str(c1)
	strc2=str(c2)
	strpercent=str(percent)
	comp_str=strj+'///'+strc1+'///'+strc2+'///'+strpercent+'$$$'	
	file1=open('adagrad.txt','a')
	file1.write(comp_str)
	file1.close()
print (iterations)
print (per)

plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.title('Accuracy vs iterations for adagrad')
plt.grid(True)
plt.plot(iterations, per, 'g^')
plt.xticks(np.arange(min(iterations), max(iterations)+1, 1.0))
plt.show()
