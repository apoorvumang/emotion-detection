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
	CascadeClassifier mouthClassifier;
	std::vector<Rect> faces;
	std::vector<Rect> mouths;
	if(!faceClassifier.load("./data/haarcascade_frontalface_alt2.xml"))
	{
		printf("Unable to load face classifier.\n");
		return 0;
	}

	if(!mouthClassifier.load("haarcascade_mcs_mouth.xml"))
	{
		printf("Unable to load mouth classifier.\n");
		return 0;
	}

	int imageSaveCounter = 21;

	while(1)
	{
		IplImage* raw = cvQueryFrame(capture);
		if(raw->nChannels == 1)
			cvCvtColor(raw, img, CV_BayerBG2BGR);
		else
			cvCopy(raw, img);

		cvCvtColor(img, grayScale, CV_BGR2GRAY);
		faces.clear();
		mouths.clear();
		faceClassifier.detectMultiScale(img, faces,1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT, Size(100, 100), Size(400,400));
		
		printf("%zd face(s) are found.\n", faces.size());

		if(faces.size() == 1)
		{
			// cvSetImageROI(img, faces[0]);
			// mouthClassifier.detectMultiScale(img, mouths);
			// cvResetImageROI(img);

			mouths.push_back(Rect(faces[0].width/4, faces[0].height/2, faces[0].width/2, faces[0].height/2));
		}



		for (int i = 0; i < faces.size(); ++i)
		{
			cvRectangle(img, cvPoint(faces[i].x, faces[i].y), cvPoint(faces[i].x+faces[i].width, faces[i].y+faces[i].height), cvScalar(255,255,255), 2);
		}

		for (int i = 0; i < mouths.size(); ++i)
		{
			// double ratio1, ratio2;
			// ratio1 = ((double)(faces[0].width))/((double)mouths[i].width);
			// ratio2 = ((double)(mouths[i].width))/((double)mouths[i].height);
			// printf("%lf and %lf\n", ratio1, ratio2);
			cvRectangle(img, cvPoint(mouths[i].x+faces[0].x, mouths[i].y+faces[0].y), cvPoint(mouths[i].x+mouths[i].width+faces[0].x, mouths[i].y+mouths[i].height+faces[0].y), cvScalar(255,255,255), 2);
		}		

		cvShowImage("Main", img);
		char c = cvWaitKey(0)&0xff;
		bool exitFlag = false;
		switch(c)
		{
			case ' ':
			exitFlag = true;
			break;

			case 's':
			if(mouths.size() == 1)
			{
				//save faces[0]
				cvSetImageROI(grayScale, Rect(faces[0].x+mouths[0].x, faces[0].y+mouths[0].y, mouths[0].width, mouths[0].height));
				IplImage* temp = cvCreateImage(cvSize(100,100), grayScale->depth, 1);
				cvResize(grayScale, temp);
				Mat myMat(temp);
				char buffer[100];
				sprintf(buffer, "./data/mouth-open/notvis/notvis_%d.png", imageSaveCounter++);
				imwrite(buffer, myMat);
				cvResetImageROI(grayScale);
				printf("Saved image %s\n", buffer);
				cvReleaseImage(&temp);
			}
			else
				printf("No mouths or multiple mouths found to save.\n");
			break;

		}
		if(c == ' ')
			break;
	}
	cvReleaseCapture(&capture);
	return 0;
}