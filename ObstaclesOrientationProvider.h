#pragma once
#include "Tools/Module/Module.h"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h> 
#include <sys/mman.h>
#include "Representations/Perception/ObstaclesPercepts/ObstaclesOrientation.h"
#include "Representations/Perception/ObstaclesPercepts/ObstaclesImagePercept.h"
#include "Representations/Infrastructure/CameraImage.h"
#include "Representations/Infrastructure/CameraInfo.h"
#include "Representations/Perception/ImagePreprocessing/ECImage.h"


MODULE(ObstaclesOrientationProvider, 
{,
    REQUIRES(CameraInfo),
    REQUIRES(ObstaclesImagePercept),
    REQUIRES(CameraImage),
    REQUIRES(ECImage),
    PROVIDES(ObstaclesOrientation),
    });
class ObstaclesOrientationProvider : public ObstaclesOrientationProviderBase 
{
    cv::CascadeClassifier cascade;
    
    void update(ObstaclesOrientation& obstaclesorientation) override;
    void  ECtoCV(cv::Mat &mat, const Image<PixelTypes::GrayscaledPixel> &gray);
public:
        ObstaclesOrientationProvider();
};