// Object detection Using openCV-4.5.1
// Target is CPU
#include <fstream>
#include <sstream>
#include <iostream>
#include <conio.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

// Initialize the parameters
double confThreshold = 0.5; // Confidence threshold
double nmsThreshold = 0.4;  // Non-maximum suppression threshold

vector<string> classes;
void postprocess(Mat& frame, const vector<Mat>& out);											// Remove bounding boxes with low confidence using non-maxima suppression
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);	// Draw the predicted bounding box
vector<String> getOutputsNames(const Net& net);													// Get the names of the output layers

int main(void)
{

	Mat blob, frame, frame2; // Create surfaces
	int width, height;
	float sx = 0.5, sy = 0.5;

	// Detect OpenCV version
	cout << "OpenCV Version " << CV_VERSION << "\n";

	// Give the configuration and weight files for the model
	//string modelConfiguration	= "yolov4.cfg";
	//string modelWeights			= "yolov4.weights";
	//string modelConfiguration	= "yolov3.cfg";
	//string modelWeights		  = "yolov3.weights";
	string modelConfiguration = "yolov3-tiny.cfg";
	string modelWeights = "yolov3-tiny.weights";
	string classesFile = "coco.names"; // List of 80 objects model is trained on
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	// Load the network
	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	// Set the backend and target to use openCV and CPU
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// Load the webcam or MP4 file
	//VideoCapture capture(0,CAP_DSHOW);
	VideoCapture capture("Traffic1.mp4", CAP_FFMPEG);
	//VideoCapture capture("newyork1970.mp4", CAP_FFMPEG);

	if (!capture.isOpened()) { cerr << "Unable to open video" << endl; _getch(); return 0; }
	capture.read(frame);

	// Video frame loop
	while (true) {
		capture >> frame;
		frame2 = frame;

		width = (int)capture.get(CAP_PROP_FRAME_WIDTH);
		height = (int)capture.get(CAP_PROP_FRAME_HEIGHT);
		if (frame.empty())
			break;

		// Filters to reduce computational power
		//bitwise_not(frame,frame);							// Invert image
		//GaussianBlur(frame, frame, Size(5, 5), 0.8);		// (input image, output image, smoothing window width and height in pixels, sigma value)

		// Create 4D blobs
		blobFromImage(frame, blob, 1 / 255.0, Size(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT)), Scalar(0, 0, 0), true, false);
		net.setInput(blob);

		// Runs the forward pass to get output of the output layers
		vector<Mat> outs;
		net.forward(outs, getOutputsNames(net));

		// Remove the bounding boxes with low confidence
		postprocess(frame, outs);

		// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
		vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;

		// Print info in frame
		string label = format("FPS: %.2f %dx%d", t / 1000.0, width, height);
		putText(frame, label, Point(5, 15), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0));

		imshow("Object Detection", frame);

		// Get the input from keyboard
		int keyboard = waitKey(30);
		if (keyboard == 'q' || keyboard == 27) break;
	} // End of video frame loop

	return 0;
} // End of main

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		/* Scan through all the bounding boxes output from the network and keep only the
		 ones with high confidence scores. Assign the box's class label as the class
		 with the highest score for the box.*/
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}


	// Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
	}


}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	// Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 1);

	// Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	// Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_PLAIN, 1.0, 1, &baseLine);
	top = max(top, labelSize.height);
	putText(frame, label, Point(left, top), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0), 1);
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		// Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		// Get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}
