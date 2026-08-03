/***********************************************************************
Copyright (c) 2022, www.guyuehome.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
***********************************************************************/

#ifndef ORIGINBOT_TELOP_H
#define ORIGINBOT_TELOP_H

#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include <boost/thread/thread.hpp>
#include <dirent.h>
#include <iostream>
#include <sys/ioctl.h>
#include <sys/poll.h>
#include <sys/stat.h>
#include <termios.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "ImageWidthReaderCpp.hpp"


#define MAX_SPEED_LINEARE_X (0.5)
#define MAX_SPEED_ANGULAR_Z (0.5)
struct SaveTask
{
    cv::Mat img;
    cv::Mat imgLeft;
    std::string filename;
    std::string leftName;
};
class OriginbotTeleop : public rclcpp::Node
{
private:
    float _speed_linear_x;
    float _speed_angular_z;
    geometry_msgs::msg::Twist cmdvel_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_cmd;
    struct termios initial_settings, new_settings;
    int kfd = 0;
    int saveCount;
    float speed, turn;
    int continueBackTLeftNum;
    int continueBackTRightNum;
    int continueBackTurnNum;
    int continueMoveNum;//打滑
    int continueTurnNum;
    bool m_savePic;
    bool m_usbCam;

    std::queue<SaveTask> saveQueue;
    std::mutex saveMutex;
    std::condition_variable saveCond;
    std::thread saveThread;
    std::atomic<bool> exitThread{false};
private:
    cv::Point2f src_points[4];
    cv::Point2f dst_points[4];
    cv::Mat rotation, img_warp,m_imgLeft;
    cv::Point lastTurnP;
    bool m_sendMove;
    bool m_stopFg;
    void rotationMatInit();
    void adjustPic(cv::Mat &src);
    void saveFramePic(std::string name,unsigned long int frame,int err,errInfo& errObj);
    void waitMs(int ms);
    void rotate_image_folder(const std::string& base_path);
    void timeoutLog(int ms,int index);
    void init();
    void saveImageThread();
public:
    bool savePic;
    OriginbotTeleop(std::string nodeName);
    void camTest();
    void turnRobot(bool isTurnLeft);
    void backTurnRobot(bool isTurnLeft);
    void stopRobot();
    void moveRobot();
    void backMove();
    ~OriginbotTeleop();
    void showMenu();


    cv::Point pic2robotAxis(cv::Point p1, double ratiox, double ratioy, int wid, int hei);
    cv::Ptr<cv::ml::SVM> m_svm;
    cv::Ptr<cv::ml::SVM> m_svm_green;

};

#endif
