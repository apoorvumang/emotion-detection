#include <cstdio>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

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

	int imageSaveCounter = 0;

	while(1)
	{
		IplImage* raw = cvQueryFrame(capture);
		if(raw->nChannels == 1)
			cvCvtColor(raw, img, CV_BayerBG2BGR);
		else
			cvCopy(raw, img);

		cvCvtColor(img, grayScale, CV_BGR2GRAY);
		faces.clear();
		faceClassifier.detectMultiScale(img, faces,1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT, Size(100, 100), Size(400,400));
		for (int i = 0; i < faces.size(); ++i)
		{
			cvRectangle(img, cvPoint(faces[i].x, faces[i].y), cvPoint(faces[i].x+faces[i].width, faces[i].y+faces[i].height), cvScalar(255,255,255), 2);

		}
		printf("%zd face(s) are found.\n", faces.size());

		cvShowImage("Main", img);
		char c = cvWaitKey(0)&0xff;
		bool exitFlag = false;
		switch(c)
		{
			case ' ':
			exitFlag = true;
			break;

			case 's':
			if(faces.size())
			{
				//save faces[0]
				cvSetImageROI(grayScale, faces[0]);
				IplImage* temp = cvCreateImage(cvSize(200,200), grayScale->depth, 1);
				cvResize(grayScale, temp);
				Mat myMat(temp);
				char buffer[100];
				sprintf(buffer, "./data/eyes-open/pos/pos_%d.png", imageSaveCounter++);
				imwrite(buffer, myMat);
				cvResetImageROI(grayScale);
				printf("Saved image %s\n", buffer);
				cvReleaseImage(&temp);
			}
			else
				printf("No face found to save.\n");
			break;

		}
		if(c == ' ')
			break;
	}
	cvReleaseCapture(&capture);
	return 0;
}