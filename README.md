# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson. 
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures. 

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.

## Mid-Term Report

* MP.1 Data Buffer Optimization
  * std::array with fixed size and circular indexing approach is implemented to solve the data buffer optimization problem.
  ```
  const int dataBufferSize{2};       // no. of images which are held in memory (ring buffer) at the same time
  array<DataFrame, dataBufferSize> dataBuffer; // list of data frames which are held in memory at the same time
  uint8_t circularIdx{0};
  //....

  DataFrame frame;
  frame.cameraImg = imgGray;
  circularIdx = circularIdx % dataBufferSize;
  dataBuffer[circularIdx] = frame;
  ```
* MP.2 Keypoint Detection
  * string selectedDetectorType is used to select keypoint detection types
  ```
  vector<string> keypointDetectorTypes{"SHITOMASI", "HARRIS", "SIFT", "FAST", "ORB", "BRISK"};
  //..
  int evaluateDetDescAlgorithms(string selectedDetectorType, string selectedDescriptorType, MatchingParameters& descMatchingParameters, ofstream& fileOut, bool visualizationEnable=false)
  {
      //...
  }   

  ```
* MP.3 Keypoint Removal Routine
  ```
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
  ```   
* MP.4 Keypoint Descriptors
  * string selectedDescriptorType is used to select descriptor types
  ```
  vector<string> keypointDescriptorTypes{"SIFT", "BRIEF", "BRISK", "FREAK", "ORB"};
  //...
  int evaluateDetDescAlgorithms(string selectedDetectorType, string selectedDescriptorType, MatchingParameters& descMatchingParameters, ofstream& fileOut, bool visualizationEnable=false)
  {
    //...
  }  
  ```
* MP.5 Descriptor Matching
  * In file dataStructures.h
  ```
  struct MatchingParameters
  {
    std::string matcherType{"MATCH_BF"};        // MAT_BF, MAT_FLANN
    std::string descriptorType{"DESCRIPTOR_BINARY"}; // DES_BINARY, DES_HOG for distance computation selection
    std::string selectorType{"SELECT_KNN"};       // SEL_NN, SEL_KNN
  };
  ```
  * In file MidTermProject_Camera_Student.cpp
  ```
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
  ```
  * In file matching2D_Student.cpp
  ```
  void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
  {
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;


    if (matcherType.compare("MATCH_BF") == 0)
    {
        int normType = descriptorType.compare("DESCRIPTOR_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType == "MATCH_FLANN")
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::FlannBasedMatcher::create();
        cout << "FLANN matching"<<endl;
    }

    // perform matching task
    if (selectorType == "SELECT_NN")
    { // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType == "SELECT_KNN")
    { // k nearest neighbors (k=2)

        vector<vector<cv::DMatch>> knnMatches;
        const float ratioThreshold = 0.80;

        matcher->knnMatch(descSource, descRef, knnMatches, 2);

        for (auto& knnMatch: knnMatches)
        {

            if ((knnMatch[0].distance / knnMatch[1].distance) < ratioThreshold)
                matches.push_back(knnMatch[0]);
        }
    }
  }
  ```
* MP.6 Descriptor Distance Ratio
  * Please see reference "In file matching2D_Student.cpp" MP.5 Descriptor Matching 
* MP.7 Performance Evaluation 1
  * Given : Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented. 
  * Solution : Avg.(taken over 10 images) Keypoints on the preceding vehicle
  
    | Detector/Descriptor  | SIFT  |
    |---|---|---|---|---|---|---|
    | SHI-TOMASI  | 658  | 
    | HARRIS  | 34 |
    | SIFT  | 747 |
    | FAST  | 649 |
    | ORB  | 604 | 
    | BRISK  | 1531 |
    | AKAZE  | 899 |
  * Solution : Average (taken over 10 images) Keypoint size
    | Keypoint Detector  | Average Size  |
    |---|---|
    | SHI-TOMASI  | 4 |
    | HARRIS  | 10 |
    | SIFT  | 4 |
    | FAST  | 7 |
    | ORB  | 55 |
    | BRISK  | 21 |
    | AKAZE  | 6 |
    
* MP.8 Performance Evaluation 2
  * Given : Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.
  * Solution : Please see the src/evaluation_2dfeatures.log for per Image count of matched keypoints. Here, summation of matched keypoints over 10 images are tabulated.
      | Detector/Descriptor  | SIFT  | BRIEF  | BRISK  | FREAK  | ORB  | AKAZE  |
    |---|---|---|---|---|---|---|
    | SHI-TOMASI  | 915 | 806 | 768  | 574  | 764  | N/A  |
    | HARRIS  | 55 | 52 | 52 | 52 | 52 | N/A  |
    | SIFT  | 800 | 598 | 590 | 485 | OutOfMemoryError | N/A  |
    | FAST  | 844 | 705 | 720 | 524 | 691 | N/A  |
    | ORB  | 767 | 453 | 747 | 361 | 516 | N/A  |
    | BRISK  | 1671 | 1319 | 1580 | 1116 | 929 | N/A  |
    | AKAZE  | N/A | N/A   | N/A  | N/A  | N/A  | 1170 | 
* MP.9 Performance Evaluation 3
  * Given : Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.
  * Solution : Average Elapsed time (ms) taken by Detector/Desriptor combo
    | Detector/Descriptor  | SIFT  | BRIEF  | BRISK  | FREAK  | ORB  | AKAZE  |
    |---|---|---|---|---|---|---|
    | SHI-TOMASI  | 26.55 | 13.468 | 13.8187 | 36.7416 | 10.8217 | N/A  |
    | HARRIS  | 18.2995 | 8.27656  | 7.99608 | 32.7012  | 10.3373  | N/A  |
    | SIFT  | 120.185 | 78.1293 | 83.7981 | 108.046 | OutOfMemoryError | N/A  |
    | FAST  | 11.5878 | 0.936305 | 1.59498 | 25.7951 | 3.03115 | N/A  |
    | ORB  | 32.1468 | 6.20513 | 6.52838 | 30.1003 | 14.4892 | N/A  |
    | BRISK  | 49.1982 | 32.6832 | 33.1593 | 56.8014 | 39.358 | N/A  |
    | AKAZE  | N/A  | N/A  | N/A  | N/A  | N/A  | 73.8249  | 
  * Solution : TOP 3 Detector/ Descriptor based on Elapsed time(ms)
    | Detector/Descriptor   | #Keypoints Matched  | Elapsed Time  |
    |---|---|---|
    | FAST+BRIEF | 705  | 0.936305  |
    | FAST+BRISK  | 720  | 1.59498  |
    | FAST+ORB  | 691  | 3.03115  |
     