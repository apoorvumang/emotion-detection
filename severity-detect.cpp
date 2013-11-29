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
	CascadeClassifier eyeClassifier;
	std::vector<Rect> faces;
	std::vector<Rect> eyes;
	if(!faceClassifier.load("./data/haarcascade_frontalface_alt2.xml"))
	{
		printf("Unable to load face classifier.\n");
		return 0;
	}

	if(!eyeClassifier.load("./data/haarcascade_mcs_eyepair_big.xml"))
	{
		printf("Unable to load eye classifier.\n");
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
		eyes.clear();
		faceClassifier.detectMultiScale(img, faces,1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT, Size(100, 100), Size(400,400));
		
		// printf("%zd face(s) are found.\n", faces.size());

		if(faces.size() == 1)
		{
			cvSetImageROI(img, faces[0]);
			eyeClassifier.detectMultiScale(img, eyes);
			cvResetImageROI(img);
		}

		if(eyes.size() == 1)
		{
			CvPoint p1 = cvPoint(eyes[0].x+faces[0].x, eyes[0].y+faces[0].y-eyes[0].height);
			CvPoint p2 = cvPoint(eyes[0].x+eyes[0].width+faces[0].x, eyes[0].y+faces[0].y);
			if(p1.y < 0)
				p1.y = 0;
			
			std::vector<int> histogram;
			unsigned long long minsumy = 0;
			unsigned long long minsum = 9999999999;
			for (int y = p1.y; y < p2.y; ++y)
			{
				unsigned long long sum = 0;
				for (int x = p1.x; x < p2.x; ++x)
				{
					CvScalar s = cvGet2D(grayScale, y,x);
					sum += (int)(s.val[0]);
				}

				// histogram.push_back(sum);
				if(sum < minsum)
				{
					// printf("Here\n");
					minsum = sum;
					minsumy = y;
				}
			}
			printf("Minsumy = %llu\n", minsumy);

			cvRectangle(img, cvPoint(p1.x, minsumy), cvPoint(p2.x, minsumy), cvScalar(0,0,255), 1);

			// cvRectangle(img, p1, p2, cvScalar(255,0,255), 2);
		}


		for (int i = 0; i < faces.size(); ++i)
		{
			cvRectangle(img, cvPoint(faces[i].x, faces[i].y), cvPoint(faces[i].x+faces[i].width, faces[i].y+faces[i].height), cvScalar(255,255,255), 2);
		}

		for (int i = 0; i < eyes.size(); ++i)
		{
			// double ratio1, ratio2;
			// ratio1 = ((double)(faces[0].width))/((double)eyes[i].width);
			// ratio2 = ((double)(eyes[i].width))/((double)eyes[i].height);
			// printf("%lf and %lf\n", ratio1, ratio2);
			cvRectangle(img, cvPoint(eyes[i].x+faces[0].x, eyes[i].y+faces[0].y), cvPoint(eyes[i].x+eyes[i].width+faces[0].x, eyes[i].y+eyes[i].height+faces[0].y), cvScalar(255,255,255), 2);
		}		

		cvShowImage("Main", img);
		char c = cvWaitKey(5)&0xff;
		bool exitFlag = false;
		switch(c)
		{
			case ' ':
			exitFlag = true;
			break;

			case 's':
			// if(eyes.size() == 1)
			// {
			// 	//save faces[0]
			// 	cvSetImageROI(grayScale, Rect(faces[0].x+eyes[0].x, faces[0].y+eyes[0].y, eyes[0].width, eyes[0].height));
			// 	IplImage* temp = cvCreateImage(cvSize(160,40), grayScale->depth, 1);
			// 	cvResize(grayScale, temp);
			// 	Mat myMat(temp);
			// 	char buffer[100];
			// 	sprintf(buffer, "./data/eyes-wide/neg/neg_%d.png", imageSaveCounter++);
			// 	imwrite(buffer, myMat);
			// 	cvResetImageROI(grayScale);
			// 	printf("Saved image %s\n", buffer);
			// 	cvReleaseImage(&temp);
			// }
			// else
			// 	printf("No eyes or multiple eyes found to save.\n");
			break;

		}
		if(c == ' ')
			break;
	}
	cvReleaseCapture(&capture);
	return 0;
}