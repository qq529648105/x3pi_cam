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
#include <chrono>
#include <ctime>
#include <sstream>
#include <string>
#include "ImageWidthReaderCpp.hpp"
#include "originbot_teleop.h"
using namespace cv;
using namespace cv::ml;
OriginbotTeleop::OriginbotTeleop(std::string nodeName) : Node(nodeName)
{

    m_stopFg=true;
    savePic=false;
    speed=0;
    turn=0;
    m_sendMove=true;
    RCLCPP_INFO(this->get_logger(), "速度:%d",speed);
    _speed_linear_x = MAX_SPEED_LINEARE_X;
    _speed_angular_z = MAX_SPEED_ANGULAR_Z;
    pub_cmd = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 1);
    lastTurnP.x=10000;
    lastTurnP.y=10000;
    saveCount=0;

    m_svm=SVM::load("/userdata/robotly/src/robotly/robotly_x3piCam/lawn.xml");
//    cmdvel_.linear.x = speed * _speed_linear_x;
//    cmdvel_.angular.z = turn * _speed_angular_z;
//    pub_cmd->publish(cmdvel_);



    rotationMatInit();

    camTest();

//    cmdvel_.linear.x = -speed * _speed_linear_x;
//    cmdvel_.angular.z = turn * _speed_angular_z;
//    pub_cmd->publish(cmdvel_);

}
OriginbotTeleop::~OriginbotTeleop()
{
   // tcsetattr(0, TCSANOW, &new_settings);
}



void OriginbotTeleop::saveFramePic(std::string name,unsigned long int frame)
{
    if(saveCount<20000)
    {
        std::time_t now = std::time(nullptr);
        std::tm tm;
        localtime_r(&now, &tm); // 将时间转换为本地时间，线程安全
        char buffer[80];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H%M%S_", &tm);

//    auto now = std::chrono::system_clock::now();
//    auto now_time_t = std::chrono::system_clock::to_time_t(now);

//    // 使用std::tm的堆栈分配（避免指针问题）
//    std::tm tm_struct = {};
//    localtime_r(&now_time_t, &tm_struct);

//    // 格式化时间
//    std::ostringstream oss;
//    oss << std::put_time(&tm_struct, "%Y-%m-%d_%H%M%S_");

//    // 添加毫秒（避免文件名重复）
//    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
//                now.time_since_epoch()) % 1000;
//    oss << std::setfill('0') << std::setw(3) << milliseconds.count() << "_";

        ++saveCount;

//    std::string retStr= oss.str();
    std::string fileName=name+std::string(buffer)+std::to_string(frame)+".png";
    imwrite(fileName,img_warp);


    }

}

std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }

    return tokens;
}
void OriginbotTeleop::rotationMatInit()
{
    src_points[0] = Point2f(180.0, 0);//lt
    src_points[1] = Point2f(442.0, 0);//rt
    src_points[2] = Point2f(0, 480.0);//lb
    src_points[3] = Point2f(640.0, 480.0);//rb

    dst_points[0] = Point2f(0.0, 0.0);
    dst_points[1] = Point2f(640.0, 0.0);
    dst_points[2] = Point2f(0.0, 480.0);
    dst_points[3] = Point2f(640.0, 480.0);
    rotation = getPerspectiveTransform(src_points, dst_points);
}
Point OriginbotTeleop::pic2robotAxis(Point p1, double ratiox,double ratioy,int wid,int hei)
{
    Point robotP;
    robotP.x=(p1.x-wid/2)*ratiox;
    robotP.y=(hei-p1.y)*ratioy;
    return robotP;
}
void OriginbotTeleop::adjustPic(Mat &src)
{
     warpPerspective(src, img_warp, rotation, src.size());
}

void OriginbotTeleop::teleopKeyboardLoop()
{
  
    int num=0;  
    while (++num<100)
    {       
            cmdvel_.linear.x = speed * _speed_linear_x;
            cmdvel_.angular.z = turn * _speed_angular_z;
            pub_cmd->publish(cmdvel_);
        
    }
}


double get_stride(int width, int bit)
{
    double temp = (width * bit / 8.0 / 16.0); // Determining whether alignment is possible
    double fractpart;
    double intpart;
   
    // get fractpart
    fractpart = modf(temp, &intpart);

    if (fractpart > 0)
    {
        // Rounding
        return ceil(temp) * 16;
    }

    return temp * 16;
}

struct arguments
{
    int width;
    int height;
    int bit;
    int count;
};

cv::Mat NV12ToBGR(const unsigned char* data, int width, int height) {
    // 创建一个YUV矩阵，注意NV12是YUV420格式，所以整个数据大小为 width * height * 3/2
    cv::Mat yuv(height * 3/2, width, CV_8UC1, (void*)data);
    cv::Mat bgr;
    // 使用OpenCV转换颜色空间，从YUV420（NV12）到BGR
    cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);
    return bgr;
}
void OriginbotTeleop::camTest()
{
     unsigned long int myCount=0;

//    Mat m1=imread("a.png");
//    ImageWidthReader reader;
//    int a=0;
//    Point pos;
//    std::vector<Point> v1=reader.getBlock(m1,pos);
//    if(v1.size())
//    {
//        bool isLeft=false;
//        if(pos.x<m1.cols/2)
//        {
//            isLeft=true;
//        }
//       RCLCPP_INFO(this->get_logger(),"返回:size:%d 障碍物位置左:%d 最近坐标:x:%d y:%d",v1.size(),isLeft,pos.x,pos.y);
//    }



    int ret = 0;
    struct arguments args;
    memset(&args, 0, sizeof(args));

    args.width=640;
    args.height=480;
    args.bit=16;
    args.count=1;

    int widths[] = {args.width};
    int heights[] = {args.height};

    int yuv_size = FRAME_BUFFER_SIZE(args.width, args.height);
    sp_sensors_parameters parms;
    parms.fps = -1;
    parms.raw_height = args.height;
    parms.raw_width = args.width;


    int yuv_count = 0;

    char yuv_filename[50];

    unsigned char *yuv_data=NULL;
    // init camera
    void *camera = sp_init_vio_module();
    // open camera
    // ret = sp_open_camera(camera, 0, -1, 1, &widths[0], &heights[0]);
    ret = sp_open_camera_v2(camera, 0, -1, 1, &parms, widths, heights);
    sleep(2); // wait for isp stability

    // malloc buffer
    // raw_data = (unsigned char*)malloc(raw_size * sizeof(char));
     yuv_data = (unsigned char*)malloc(yuv_size * sizeof(char));


     if (ret != 0)
     {
         printf("[Error] sp_open_camera failed!\n");
         free(yuv_data);

         sp_vio_close(camera);
         sp_release_vio_module(camera);
         return;

     }
     //frame_data= (unsigned char*)malloc(yuv_size * sizeof(char));

     int continueTurn=0;
     int moveNum=0;
     int backTurnNum=0;
     int stopNum=0;
     int cccc1;
     ImageWidthReader reader;//=new ImageWidthReader();
     auto startT = std::chrono::high_resolution_clock::now();
     //while(myCount<15)
     while(1)
     {
        ++myCount;
//        if(saveCount>20000)
//        {
//            system("rm -f saveImg/*.png");
//        }
        auto curT = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(curT-startT);
        int frameTime=duration.count();
        printf("frame:%ld %dms \n", myCount,frameTime);

        startT = std::chrono::high_resolution_clock::now();
        sp_vio_get_yuv(camera, (char*)yuv_data, args.width, args.height, 2000);
        sprintf(yuv_filename, "yuv_%d.yuv", yuv_count++);
        // FILE *yuv_file = fopen(yuv_filename, "wb");
        // fwrite(yuv_data, sizeof(char), yuv_size, yuv_file);
        // fflush(yuv_file);

        //sp_vio_get_frame(camera, (char*)frame_data, args.width, args.height, 2000);

        //sp_vio_get_raw(camera, (char*)raw_data, args.width, args.height, 2000);

        Mat matFrame=NV12ToBGR(yuv_data,args.width,args.height);

        adjustPic(matFrame);
        if(myCount<10)
        {

            continue;
        }

//        if(myCount==13)
//            img_warp=imread("stone.png");
//        else
//            img_warp=imread("lawn.png");




        double area;
        Point pos;
        std::vector<Point> v1;
        double allAreaRatio;
        auto getBlockT = std::chrono::high_resolution_clock::now();


        v1=reader.getBlock(img_warp,m_svm,pos,allAreaRatio,cccc1);

        auto endBlockT = std::chrono::high_resolution_clock::now();

        auto duraBlock = std::chrono::duration_cast<std::chrono::milliseconds>(endBlockT-getBlockT);

        int blockTime=duraBlock.count();

        printf("\n检测时间:%dms %d\n",blockTime,cccc1);
        Rect rect(0,0,1,1);

        //printf("getBlock\n");
        Point robotP(1000,1000);


        if(v1.size()&&pos.y>140)
        {

            robotP=pic2robotAxis(pos,0.40322,1.5118,640,480);
            area=contourArea(v1);
            rect=boundingRect(v1);



            if(allAreaRatio>0.7||(robotP.y<10&&abs(robotP.x)<100))
            {
                ++stopNum;
                printf("物体 x:%d y:%d allAreaRatio:%.2f\n",robotP.x,robotP.y,allAreaRatio);
                if(moveNum>10)//&&backTurnNum<5)
                {

                    saveFramePic("/userdata/robotly/src/robotly/robotly_x3piCam/saveImg/backT_",myCount);

                    ++backTurnNum;
                    backTurnRobot(lastTurnP.x>0?true:false);
                    stopRobot();


                }
//                if(stopNum%40==0&&continueTurn>10)
//                {
//                    printf("停止:%d",stopNum);
//                    backTurnNum=0;
//                }
            }
            else
            {
                stopNum=0;
                backTurnNum=0;

                if(area>150)
                {
                    saveFramePic("/userdata/robotly/src/robotly/robotly_x3piCam/saveImg/turn_",myCount);


                    lastTurnP=robotP;
                    ++continueTurn;
                    turnRobot(robotP.x>0?true:false);
                }


            }

        }
        else
        {
            stopNum=0;
            backTurnNum=0;
            continueTurn=0;
            ++moveNum;
            if(moveNum>10000000)
                moveNum=11;
            moveRobot();
            if(moveNum%100==0)
            {
                saveFramePic("/userdata/robotly/src/robotly/robotly_x3piCam/saveImg/auto_",myCount);

            }
        }

        if(robotP.x==1000)
        {

            printf("返回:无障碍，前进 frame:%ld %dms\n",myCount,frameTime);
        }
        else
        {
            printf("返回:rect:%d %d %d %d 障碍物:%d %d pos.y:%d frame:%ld %dms\n",rect.x,rect.y,rect.width,rect.height,robotP.x,robotP.y,pos.y, myCount,frameTime);

        }

//        if(savePic)
//        {
//            imwrite("yuv.png",img_warp);
//            Mat rgb=img_warp.clone();
//            for(unsigned int i=0;i<v1.size();i++)
//            {
//                circle(rgb,v1[i],1,Scalar(0,0,255),1);
//            }
//            circle(rgb,pos,1,Scalar(255,0,0),5);
//            imwrite("yuv_2.png",rgb);
//            break;
//        }


        //imwrite("show.png",img_warp);

        // cv::Mat f2(args.height,args.width,CV_8UC3,raw_data);
        // imwrite("raw.png",f2);

        // cv::Mat f3(args.height,args.width,CV_8UC3,frame_data);
        // imwrite("frame.png",f3);

        //printf("save f2.png \n");
        //sprintf(raw_filename, "raw_%d.raw", raw_count++);
        // FILE *raw_file = fopen(raw_filename, "wb");
        // fwrite(raw_data, sizeof(char), raw_size, raw_file);
        // fflush(raw_file);

    }
   // printf("aaa\n");
    // sleep(3);
     if(yuv_data)
       free(yuv_data);
    // printf("bbb\n");

     RCLCPP_INFO(this->get_logger(),"你好节点");


}
void OriginbotTeleop::stopRobot()
{
    m_stopFg=true;
    speed=0;
    turn=0;
    cmdvel_.linear.x = speed * _speed_linear_x;
    cmdvel_.angular.z = turn * _speed_angular_z;
    pub_cmd->publish(cmdvel_);
    printf("stop robot\n");
    sleep(1);
}


void OriginbotTeleop::turnRobot(bool isTurnLeft)
{

    //60 ＝30转/min
    m_stopFg=true;
    speed=60;
    turn=100;
    cmdvel_.linear.x = speed * _speed_linear_x;
    cmdvel_.angular.z = turn * _speed_angular_z;
    if(isTurnLeft==false)
    {
        cmdvel_.angular.z*=-1;
    }
    if(m_sendMove)
        pub_cmd->publish(cmdvel_);
    if(isTurnLeft)
    {
         printf("左转 robot\n");
    }
    else
    {
         printf("右转 robot\n");
    }

    sleep(2);
}

void OriginbotTeleop::moveRobot()
{

    speed=60;//3;
    turn=0;
    cmdvel_.linear.x = speed * _speed_linear_x;
    cmdvel_.angular.z = turn * _speed_angular_z;
    if(m_sendMove)
        pub_cmd->publish(cmdvel_);

    if(m_stopFg)
    {
        sleep(1);
    }

    m_stopFg=false;

}

void OriginbotTeleop::backTurnRobot(bool isTurnLeft)
{
    stopRobot();
    speed=-40;
    turn=0;
    cmdvel_.linear.x = speed * _speed_linear_x;
    cmdvel_.angular.z = turn * _speed_angular_z;
    if(isTurnLeft==false)
    {
        cmdvel_.angular.z*=-1;
    }
    if(m_sendMove)
        pub_cmd->publish(cmdvel_);
    if(isTurnLeft)
    {
         printf("后退 robot\n");
    }
    else
    {
         printf("后退 robot\n");
    }

    sleep(2);

    stopRobot();

    speed=-20;
    turn=236;
    cmdvel_.linear.x = speed * _speed_linear_x;
    cmdvel_.angular.z = turn * _speed_angular_z;
    if(isTurnLeft==false)
    {
        cmdvel_.angular.z*=-1;
    }
    if(m_sendMove)
        pub_cmd->publish(cmdvel_);
    if(isTurnLeft)
    {
         printf("原地倒向左转 robot\n");
    }
    else
    {
         printf("原地倒向右转 robot\n");
    }
    sleep(2);
    stopRobot();
}
