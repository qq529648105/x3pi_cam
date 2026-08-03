#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <sp_vio.h>
#include <sp_sys.h>
#include <sp_display.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "originbot_teleop.h"
using namespace cv;
int main(int argc,char** argv)
{
    rclcpp::init(argc,argv);

    auto node = std::make_shared<OriginbotTeleop>("originbot_teleop");
    //rclcpp::spin(std::make_shared<OriginbotTeleop>("originbot_teleop"));
    //rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}
