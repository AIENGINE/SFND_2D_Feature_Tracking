/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <numeric>
#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

int evaluateDetDescAlgorithms(string selectedDetectorType, string selectedDescriptorType, MatchingParameters& descMatchingParameters, ofstream& fileOut, bool visualizationEnable=false)
{
    string dataPath{"../"};

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix{"KITTI/2011_09_26/image_00/data/000000"}; // left camera, color
    string imgFileType{".png"};
    int imgStartIndex{0}; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex{9};   // last file index to load
    int imgFillWidth{4};  // no. of digits which make up the file index (e.g. img-0001.png)


    //NOTE: For the project if buffersize is increased from 2 then calculations below should be adjusted as well
    const int dataBufferSize{2};       // no. of images which are held in memory (ring buffer) at the same time
    array<DataFrame, dataBufferSize> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis;            // visualize results
    uint8_t circularIdx{0};
    uint numberOfKeypointsOnVehicle{0};
    vector<uint> numOfKeypointsPerImage;
    uint avgKeypointNeighborhoodSize{0};
    vector<uint> nunOfKeypointNeighborhoodSizePerImage;
    vector<uint> numOfKeypointMatchesPerImage;
    uint numOfKeypointMatches;
    double keypointDetectionTime;
    double keypointDiscriptorTime;
    vector<double> keypointDetectionTimePerImg;
    vector<double> keypointDiscriptorTimePerImg;
    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        DataFrame frame;
        frame.cameraImg = imgGray;
        circularIdx = circularIdx % dataBufferSize;
        dataBuffer[circularIdx] = frame;


        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        //string detectorType = selectedDetectorType;

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        if (selectedDetectorType == "SHITOMASI")
        {
            detKeypointsShiTomasi(keypoints, imgGray, keypointDetectionTime, visualizationEnable);
        }
        else if (selectedDetectorType == "HARRIS")
        {
            detKeypointsHarris(keypoints, imgGray, keypointDetectionTime, visualizationEnable);
        }
        else
        {
            detKeypointsModern(keypoints, imgGray, selectedDetectorType, keypointDetectionTime, visualizationEnable);
        }
        keypointDetectionTimePerImg.push_back(keypointDetectionTime);
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        vector<cv::KeyPoint> frontVechileKeypoints;
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            for (auto& keypoint: keypoints)
            {
                if (vehicleRect.contains(keypoint.pt))
                {
                    cv::KeyPoint newKeyPoint;
                    newKeyPoint = keypoint;
                    frontVechileKeypoints.push_back(newKeyPoint);
                }
            }
            keypoints = frontVechileKeypoints;
            numberOfKeypointsOnVehicle += keypoints.size();
            numOfKeypointsPerImage.push_back(numberOfKeypointsOnVehicle);
        }

        //neighborhood size of the keypoints and averaging per image
        for (auto & keypoint : keypoints)
        {
            avgKeypointNeighborhoodSize += keypoint.size;
        }
        avgKeypointNeighborhoodSize = avgKeypointNeighborhoodSize / keypoints.size();
        nunOfKeypointNeighborhoodSizePerImage.push_back(avgKeypointNeighborhoodSize);
        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false ;
        if (bLimitKpts)
        {
            const int maxKeypoints = 50;
            keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        dataBuffer[circularIdx].keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        //string descriptorType = "BRISK"; // BRIEF, ORB, FREAK, AKAZE, SIFT
        descKeypoints(dataBuffer[circularIdx].keypoints, dataBuffer[circularIdx].cameraImg, descriptors, selectedDescriptorType, keypointDiscriptorTime);
        keypointDiscriptorTimePerImg.push_back(keypointDiscriptorTime);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        dataBuffer[circularIdx].descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;
        circularIdx++;
//        if (circularIdx > 1)
//        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        if (!dataBuffer[0].cameraImg.empty() && !dataBuffer[1].cameraImg.empty())
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = descMatchingParameters.matcherType;        // MATCH_BF, MATCH_FLANN
            string descriptorType = descMatchingParameters.descriptorType; // DESCRIPTOR_BINARY, DESCRIPTOR_HOG for distance computation selection
            string selectorType = descMatchingParameters.selectorType;       // SELECT_NN, SELECT_KNN

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);

            //// EOF STUDENT ASSIGNMENT
            //TODO: Count the number of matched keypoints for all 10 images

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;
            numOfKeypointMatches = (dataBuffer.end()-1)->kptMatches.size();
            numOfKeypointMatchesPerImage.push_back(numOfKeypointMatches);

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = visualizationEnable;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images
    uint totalAvgNumKeypointsOnVehilce = accumulate(numOfKeypointsPerImage.begin(), numOfKeypointsPerImage.end(), 0.0) / numOfKeypointsPerImage.size();
    uint totalAvgNumKeypointsNeighborboodSize = accumulate(nunOfKeypointNeighborhoodSizePerImage.begin(), nunOfKeypointNeighborhoodSizePerImage.end(), 0.0) / nunOfKeypointNeighborhoodSizePerImage.size();
    fileOut<< "Avg keypoints on the preceding Vehicle for "<<numOfKeypointsPerImage.size() << " images : "<< totalAvgNumKeypointsOnVehilce<<endl;
    fileOut<< "Avg keypoint neighborhood size for "<<nunOfKeypointNeighborhoodSizePerImage.size()<< " images : " << totalAvgNumKeypointsNeighborboodSize<<endl;
    for (uint idx{0}; idx <= numOfKeypointMatchesPerImage.size(); idx++)
    {
        fileOut<< "Keypoints Matches in Image " << idx << " : " << numOfKeypointMatchesPerImage[idx]<<endl;

    }
    fileOut<< "Keypoints Matches Accumulated in all 10 Images : "<< accumulate(numOfKeypointMatchesPerImage.begin(), numOfKeypointMatchesPerImage.end(), 0.0) << endl;

    vector<double> totalDetDescTime(keypointDetectionTimePerImg.size());
    transform(keypointDetectionTimePerImg.begin(), keypointDetectionTimePerImg.end(),
            keypointDiscriptorTimePerImg.begin(), totalDetDescTime.begin(), plus<double>());
//    for (uint imgIdx{0}; imgIdx <= numOfKeypointMatchesPerImage.size(); imgIdx++)
//    {
//        cout<< "Total Det. + Des. time : " << imgIdx << " = " << keypointDetectionTimePerImg[imgIdx] + keypointDiscriptorTimePerImg[imgIdx]<<endl;
//
//    }
    for (uint imgIdx{0}; imgIdx < totalDetDescTime.size(); imgIdx++)
    {
        fileOut<< "Total Det. + Des. time : " << imgIdx << " = " << totalDetDescTime[imgIdx]<<endl;
    }
    double avgTimeElapsed = accumulate(totalDetDescTime.begin(), totalDetDescTime.end(), 0.0) / totalDetDescTime.size();
    fileOut<< "Avg Time Elapsed in " <<totalDetDescTime.size() << " Images : "<< avgTimeElapsed <<endl;
    return 0;

}

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    MatchingParameters descMatchingParameters;
    vector<string> keypointDetectorTypes{"SHITOMASI", "HARRIS", "SIFT", "FAST", "ORB", "BRISK"};
    vector<string> keypointDescriptorTypes{"SIFT", "BRIEF", "BRISK", "FREAK", "ORB"};
    uint retval;
    string outputLogFile{"../src/evaluation_2dfeatures.log"};
    ofstream out(outputLogFile, ios::out);
    bool visualizationFlag{false};
    for (auto& detectorType: keypointDetectorTypes)
    {
        for (auto& descriptorType: keypointDescriptorTypes)
        {
            out<< "DetectorType = "<< detectorType << ", "<< "DescriptorType = "<< descriptorType<<endl;
            cout<< "DetectorType = "<< detectorType << ", "<< "DescriptorType = "<< descriptorType<<endl;
            if(detectorType == "SIFT" and descriptorType == "ORB")
                continue; //SIFT det. and ORB Desc.
            if (descriptorType == "BRISK")
            {
                descMatchingParameters.matcherType = "MATCH_BF";
                descMatchingParameters.descriptorType = "DESCRIPTOR_BINARY";
                descMatchingParameters.selectorType = "SELECT_KNN";
                retval = evaluateDetDescAlgorithms(detectorType, descriptorType, descMatchingParameters, out, visualizationFlag);
                if(retval != 0)
                {
                    cerr<< "Evaluation failed!!!" <<endl;
                    exit(1);
                }
            }
            if(descriptorType != "BRISK")
            {
                descMatchingParameters.matcherType = "MATCH_BF";
                descMatchingParameters.descriptorType = "DESCRIPTOR_HOG";
                descMatchingParameters.selectorType = "SELECT_KNN";
                retval = evaluateDetDescAlgorithms(detectorType, descriptorType, descMatchingParameters, out, visualizationFlag);
            }
            if(retval != 0)
            {
                cerr<< "Evaluation failed!!!" <<endl;
                exit(1);
            }


        }
    }
    out<< "DetectorType = "<< "AKAZE"<< ", "<< "DescriptorType = "<< "AKAZE"<<endl;
    cout<< "DetectorType = "<< "AKAZE"<< ", "<< "DescriptorType = "<< "AKAZE"<<endl;
    retval = evaluateDetDescAlgorithms("AKAZE", "AKAZE", descMatchingParameters, out, visualizationFlag);
    if(retval != 0)
    {
        cerr<< "Evaluation failed!!!" <<endl;
        exit(1);
    }
    out.close();
    return 0;
}
