#include "Tools/Debugging/Stopwatch.h"
#include "ObstaclesOrientationProvider.h"
#include "Representations/spqr_representations/OurDefinitions.h"
#include <algorithm>
#include <iostream>
#include "Tools/Debugging/Stopwatch.h"




using namespace cv; 

ObstaclesOrientationProvider::ObstaclesOrientationProvider()
{
    std::string filename="/home/gabrbrr/RoboCup/spqrnao2023/Config/lbp_orientation.xml";
    if (!cascade.load(filename)) {
        std::cerr << "---------------------(!)Error loading CASCADE" << std::endl;
    }

}


void ObstaclesOrientationProvider::update(ObstaclesOrientation& obstaclesorientation){
    if(theObstaclesImagePercept.obstacles.size()>0){
        Image<PixelTypes::GrayscaledPixel> gray=theECImage.grayscaled;
        cv::Mat image(gray.height, gray.width, CV_8U);
        ECtoCV(image,gray);
        int top, right,left,bottom;
        for(ObstaclesImagePercept::Obstacle obstacle : theObstaclesImagePercept.obstacles){
            if(!obstacle.bottomFound || obstacle.fallen || obstacle.probability<0.9) continue;
            top=std::max(obstacle.top,0); right=std::min(std::max(obstacle.right,0),image.cols); left= std::max(obstacle.left,0); bottom=std::min(std::max(obstacle.bottom,0),image.rows);
            int newtop, newleft, newwidth,newheight,m;
            m=0;newtop=top+(int)(bottom-top)*2/3; newleft=std::max(left-10,0); newwidth=std::min(right-left+10,image.cols-left); newheight=std::min(((int)(bottom-top)/3+10),image.rows-newtop);

            Rect recRoi(newleft,newtop,newwidth,newheight);
            Mat roi = image(recRoi);
            std::vector<Rect> feet;
            STOPWATCH("ObstaclesOrientationProvider::detect") cascade.detectMultiScale(roi,feet,1.1,8,30,Size(15,15),Size(image.cols-1,image.rows-1));
            // for (int i=0;i<feet.size();i++){
            //     std::string name =  std::to_string(theCameraImage.timestamp)+"_"+std::to_string(m) + "_" + std::to_string(i) + "_" + ".bmp";
            //     if (cv::imwrite( name, roi(feet[i]) )) {
            //     std::cerr << std::to_string(theCameraImage.timestamp).c_str() << " salvata" << std::endl;
            //     } 
			// 													else {
            //     std::cerr << "error saving " << std::to_string(theCameraImage.timestamp).c_str() << std::endl;
            //     }
            // }
            // m++;
        
        // std::string name = std::to_string(theCameraImage.timestamp) + "_log" + ".bmp";
        // if(  cv::imwrite(name, roi, { CV_IMWRITE_PNG_COMPRESSION,9 } ) ) {
        //     std::cerr << std::to_string(theCameraImage.timestamp).c_str() << " salvata" << std::endl;
        // } else {
        //     std::cerr << "error saving " << std::to_string(theCameraImage.timestamp).c_str() << std::endl;
        // }
        
        
    }
}}


void  ObstaclesOrientationProvider::ECtoCV(cv::Mat &mat, const Image<PixelTypes::GrayscaledPixel> &gray){
    

    for (int y = 0; y < gray.height; ++y) {
        for (int x = 0; x < gray.width; ++x) {
            uint8_t pixelValue = gray(x,y);
            mat.at<uint8_t>(y, x) = pixelValue; 
        }
    }
}

MAKE_MODULE(ObstaclesOrientationProvider,perception);
   

