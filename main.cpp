#include <cstdio>
#include <fstream>
#include <opencv2/opencv.hpp>

#define MAX_TRIES 10
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
	IplImage* temp  = cvQueryFrame(capture);

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
	std::vector<Rect> mouths;
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



	vector<Mat> images;
    vector<int> labels;
	try {
        read_csv("eyes-open.csv", images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file. Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
   
    int im_eye_width = images[0].cols;
    int im_eye_height = images[0].rows;
    
    Ptr<FaceRecognizer> eyesOpen = createFisherFaceRecognizer();
    eyesOpen->train(images, labels);
   
   images.clear();
   labels.clear();


   try {
        read_csv("mouth-open.csv", images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file. Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
   
    int im_mouth_width = images[0].cols;
    int im_mouth_height = images[0].rows;
    
    Ptr<FaceRecognizer> mouthOpen = createFisherFaceRecognizer();
    mouthOpen->train(images, labels);


    images.clear();
    labels.clear();

    try {
        read_csv("eye-screen.csv", images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file. Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }

    Ptr<FaceRecognizer> eyesScreen = createFisherFaceRecognizer();
    eyesScreen->train(images, labels);

    printf("Model Trained.\n");

    bool var_eyeVisible = false;

    double mainScore = 0;
    int numtries = MAX_TRIES;

	while(numtries-- > 0)
	{
		int frameCounter = 0;
		bool exitFlag = false;

		double eyeOpenCount = 0;
		double eyeScreenCount = 0;
		double eyeVisibleCount = 0;
		double faceVisibleCount = 0;
		double faceDiffX = 0;
		double faceDiffY = 0;
		double lastX = 0, lastY = 0;
		double frownCount = 0;
		double mouthCount = 0;
		double score = 0;

		while(frameCounter < 20)
		{
			frameCounter ++;
			IplImage* raw = cvQueryFrame(capture);
			if(raw->nChannels == 1)
				cvCvtColor(raw, img, CV_BayerBG2BGR);
			else
				cvCopy(raw, img);

			cvCvtColor(img, grayScale, CV_BGR2GRAY);
			faces.clear();
			eyes.clear();
			mouths.clear();

			faceClassifier.detectMultiScale(img, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT, Size(100, 100), Size(400,400));

			if(faces.size() == 1)
			{
				cvSetImageROI(img, faces[0]);
				eyeClassifier.detectMultiScale(img, eyes);
				cvResetImageROI(img);
				mouths.push_back(Rect(faces[0].width/4, faces[0].height/2, faces[0].width/2, faces[0].height/2));
			}

			//detecting eye features

			int eyesOpenPrediction = 0;
			int eyesScreenPrediction = 0;
			if(faces.size() == 1 && eyes.size() == 1)
			{
				cvSetImageROI(grayScale, Rect(faces[0].x+eyes[0].x, faces[0].y+eyes[0].y, eyes[0].width, eyes[0].height));
				IplImage* eye_resized = cvCreateImage(cvSize(im_eye_width,im_eye_height), grayScale->depth, 1);
				cvResize(grayScale, eye_resized);
				Mat eye(eye_resized);
				

				eyesOpenPrediction = eyesOpen->predict(eye);
				eyesScreenPrediction = eyesScreen->predict(eye);
				// printf("Prediction = %d\n", prediction);

				if(eyesOpenPrediction == 1)
				{
					eyeOpenCount++;
				}

				if(eyesScreenPrediction == 1)
					eyeScreenCount++;

				

	          
	            // string text[4] = {"Away from screen", "At Screen"};
	         
	            // Mat temp(img);
	            // putText(temp, text[prediction], Point(50, 50), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

				cvResetImageROI(grayScale);
				cvReleaseImage(&eye_resized);
				// cvRectangle(img, cvPoint(faces[i].x, faces[i].y), cvPoint(faces[i].x+faces[i].width, faces[i].y+faces[i].height), cvScalar(255,255,255), 2);
			}



			string text[4] = {"Closed", "Open"};
         
	            Mat temp(img);
	            putText(temp, text[eyesOpenPrediction], Point(50, 50), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 2.0);

	            string text2[4] = {"Away from Screen", "At Screen"};
         
	            
	            putText(temp, text2[eyesScreenPrediction], Point(50, 100), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 2.0);

			//detecting mouth state
			if(faces.size() == 1)
			{
				cvSetImageROI(grayScale, Rect(faces[0].x+mouths[0].x, faces[0].y+mouths[0].y, mouths[0].width, mouths[0].height));
				IplImage* mouth_resized = cvCreateImage(cvSize(im_mouth_width,im_mouth_height), grayScale->depth, 1);
				cvResize(grayScale, mouth_resized);
				Mat mouth(mouth_resized);
				

				int mouthPrediction = mouthOpen->predict(mouth);

				if(mouthPrediction)
					mouthCount++;
				// printf("Prediction = %d\n", prediction);
	          
	            // string text[4] = {"Closed", "Not Visible", "Open"};
	         
	            // Mat temp(img);
	            // putText(temp, text[prediction], Point(50, 50), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

				cvResetImageROI(grayScale);
				cvReleaseImage(&mouth_resized);
				// cvRectangle(img, cvPoint(faces[i].x, faces[i].y), cvPoint(faces[i].x+faces[i].width, faces[i].y+faces[i].height), cvScalar(255,255,255), 2);
			}

			bool frownFound = false;
			//detecting frown
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
				// printf("Minsumy = %lf\n", ((double)((int)(eyes[0].y+faces[0].y) - minsumy))/(double)eyes[0].height);

				if(((double)((int)(eyes[0].y+faces[0].y) - minsumy))/(double)eyes[0].height < 0.2)
				{
					frownFound = true;
					frownCount++;
				}

				cvRectangle(img, cvPoint(p1.x, minsumy), cvPoint(p2.x, minsumy), cvScalar(0,0,255), 1);

				// cvRectangle(img, p1, p2, cvScalar(255,0,255), 2);
			}

			if(frownFound)
	            putText(temp, "Frown present", Point(50, 150), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 2.0);
	        else
	            putText(temp, "Frown not present", Point(50, 150), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 2.0);


			//painting
			for (int i = 0; i < eyes.size(); ++i)
			{
				// double ratio1, ratio2;
				// ratio1 = ((double)(faces[0].width))/((double)eyes[i].width);
				// ratio2 = ((double)(eyes[i].width))/((double)eyes[i].height);
				// printf("%lf and %lf\n", ratio1, ratio2);
				cvRectangle(img, cvPoint(eyes[i].x+faces[0].x, eyes[i].y+faces[0].y), cvPoint(eyes[i].x+eyes[i].width+faces[0].x, eyes[i].y+eyes[i].height+faces[0].y), cvScalar(255,255,255), 2);
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



			if(eyes.size())
				eyeVisibleCount++;

			if(faces.size())
			{
				faceVisibleCount++;
				if(frameCounter != 1)
				{
					faceDiffX += fabs(faces[0].x - lastX);
					faceDiffY += fabs(faces[0].y - lastY);
				}
				lastX = faces[0].x;
				lastY = faces[0].y;
			}

					// printf("%zd face(s) are found.\n", faces.size());



			cvShowImage("Main", img);
			char c = cvWaitKey(5)&0xff;
			exitFlag = false;
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


		

		if((eyeOpenCount+1)/(eyeVisibleCount+1) < 0.5)
		{
			printf("Blinking\n");
		}
		else
		{
			score++;
			printf("Not blinking\n");
		}

		if((eyeScreenCount+1)/(eyeVisibleCount+1) < 0.5)
		{
			printf("Eyes away from screen\n");
		}
		else
		{
			score++;
			printf("Eyes at screen\n");
		}


		if(faceDiffY > 100 || faceDiffX > 100)
			printf("Head moving\n");
		else
		{
			score++;
			printf("Head not moving\n");
		}

		if(frownCount/faceVisibleCount > 0.5)
		{
			score++;
			printf("Frowning\n");
		}
		else
		{
			printf("Not Frowning\n");
		}

		if((mouthCount/(faceVisibleCount+1)) > 0.2)
			printf("Mouth open/covered\n");
		else
		{
			score++;
			printf("Mouth closed\n");
		}

		printf("\n\tYour score = %lf\n\n", score);

		mainScore += score;

		// printf("%lf %lf\n", faceDiffX, faceDiffY);
		if(exitFlag)
			break;

		// cvWaitKey();
	}

	mainScore = mainScore / (MAX_TRIES-numtries);
	printf("\n\nYour final score is %lf\n", mainScore);
	cvReleaseCapture(&capture);
	return 0;
}