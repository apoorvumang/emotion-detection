#include <cstdio>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}


int main()
{
	CvCapture* capture = cvCaptureFromCAM(0);
	if(!capture)
	{
		printf("Unable to open camera.\n");
		return 0;
	}
	IplImage* temp = cvQueryFrame(capture);

	int initGrabCount = 0;
	while(!temp && initGrabCount < 20)
	{
		cvWaitKey(5);
		temp = cvQueryFrame(capture);
	}

	if(!temp)
	{
		printf("Unable to get image from camera.\n");
		return 0;
	}
	
	IplImage* img = cvCreateImage(cvSize(temp->width, temp->height), temp->depth, 3);
	IplImage* grayScale = cvCreateImage(cvSize(temp->width, temp->height), temp->depth, 1);
	CascadeClassifier faceClassifier;
	std::vector<Rect> faces;
	if(!faceClassifier.load("haarcascade_frontalface_alt2.xml"))
	{
		printf("Unable to load classifier.\n");
		return 0;
	}



	vector<Mat> images;
    vector<int> labels;







	try {
        read_csv("apoorv-data.csv", images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file. Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size AND we need to reshape incoming faces to this size:
    int im_width = images[0].cols;
    int im_height = images[0].rows;
    // Create a FaceRecognizer and train it on the given images:
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
    model->train(images, labels);
    // That's it for learning the Face Recognition model. You now
    // need to create the classifier for the task of Face Detection.
    // model->save("apoorv-emotion-trained.yml");
    printf("Model Trained.\n");




	while(1)
	{
		IplImage* raw = cvQueryFrame(capture);
		if(raw->nChannels == 1)
			cvCvtColor(raw, img, CV_BayerBG2BGR);
		else
			cvCopy(raw, img);

		cvCvtColor(img, grayScale, CV_BGR2GRAY);
		faces.clear();
		faceClassifier.detectMultiScale(img, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT, Size(100, 100), Size(400,400));
		for (int i = 0; i < faces.size(); ++i)
		{
			cvSetImageROI(grayScale, faces[i]);
			IplImage* facei_resized = cvCreateImage(cvSize(im_width,im_height), grayScale->depth, 1);
			cvResize(grayScale, facei_resized);
			Mat face_i(facei_resized);
			

			int prediction = model->predict(face_i);
			printf("Prediction = %d\n", prediction);
            // And finally write all we've found out to the original image!
            // First of all draw a green rectangle around the detected face:
            // cvRectangle(img, cvPoint(faces[i].x, faces[i].y), cvPoint(faces[i].x+faces[i].width, faces[i].y+faces[i].height), CV_RGB(0, 255,0), 1);
            // // Create the text we will annotate the box with:
            string text[4] = {"Happy", "Neutral", "Sad", "Surprised"};
            // string box_text = format("Prediction = %d", prediction);
            // // Calculate the position for annotated text (make sure we don't
            // // put illegal values in there):
            // int pos_x = std::max(faces[i].tl().x - 10, 0);
            // int pos_y = std::max(faces[i].tl().y - 10, 0);
            // // And now put it into the image:
            Mat temp(img);
            putText(temp, text[prediction], Point(50, 50), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

			cvResetImageROI(grayScale);
			cvReleaseImage(&facei_resized);
			// cvRectangle(img, cvPoint(faces[i].x, faces[i].y), cvPoint(faces[i].x+faces[i].width, faces[i].y+faces[i].height), cvScalar(255,255,255), 2);

		}
		// printf("%zd face(s) are found.\n", faces.size());

		cvShowImage("Main", img);
		char c = cvWaitKey(5)&0xff;
		bool exitFlag = false;
		switch(c)
		{
			case ' ':
			exitFlag = true;
			break;

			default:
			break;
		}

		if(exitFlag)
			break;
	}
	cvReleaseCapture(&capture);
	return 0;
}