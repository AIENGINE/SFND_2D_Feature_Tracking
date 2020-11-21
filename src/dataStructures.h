#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <string>
#include <opencv2/core.hpp>


struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
};

struct MatchingParameters
{
    std::string matcherType{"MATCH_BF"};        // MAT_BF, MAT_FLANN
    std::string descriptorType{"DESCRIPTOR_BINARY"}; // DES_BINARY, DES_HOG for distance computation selection
    std::string selectorType{"SELECT_KNN"};       // SEL_NN, SEL_KNN
};
#endif /* dataStructures_h */
