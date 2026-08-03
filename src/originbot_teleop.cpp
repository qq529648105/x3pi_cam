#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <sp_vio.h>
#include <sp_sys.h>
#include <sp_display.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <ctime>
#include <sstream>
#include <fstream>
#include <algorithm>
#include "originbot_teleop.h"
using namespace cv;
using namespace cv::ml;

OriginbotTeleop::OriginbotTeleop(std::string nodeName) : Node(nodeName)
{

    m_savePic=false;
    m_usbCam=false;
    m_stopFg=true;
    savePic=false;
    speed=0;
    turn=0;

    //RCLCPP_INFO(this->get_logger(), "速度:%d",speed);
    _speed_linear_x = MAX_SPEED_LINEARE_X;
    _speed_angular_z = MAX_SPEED_ANGULAR_Z;
    pub_cmd = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 1);
    lastTurnP.x=10000;
    lastTurnP.y=10000;
    saveCount=0;
    continueBackTLeftNum=0;
    continueBackTRightNum=0;
    continueBackTurnNum=0;
    continueMoveNum=0;
    continueTurnNum=0;

    saveThread = std::thread(&OriginbotTeleop::saveImageThread, this);

    init();
    rotate_image_folder("/userdata/robotly/src/robotly/robotly_x3piCam");
    m_svm=SVM::load("/userdata/robotly/src/robotly/robotly_x3piCam/lawn.xml");
    m_svm_green=SVM::load("/userdata/robotly/src/robotly/robotly_x3piCam/lawn_green.xml");
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
    exitThread = true;
    saveCond.notify_all();

    if (saveThread.joinable())
        saveThread.join();
   // tcsetattr(0, TCSANOW, &new_settings);
}
void OriginbotTeleop::init()
{
    std::ifstream configFile("/userdata/robotly/src/robotly/robotly_x3piCam/config.txt");  // 当前目录下的配置文件
    if (!configFile.is_open()) {
        std::cerr << "错误：无法打开配置文件 config.txt" << std::endl;
        return ;
    }


    std::string line;
    while (std::getline(configFile, line)) {
        // 跳过空行或注释行（可选）
        if (line.empty()) continue;

        if (line == "saveImg=1") {
            m_savePic = true;
            //break;   // 找到后立即退出，因为只需要知道是否存在
        }
        else if(line=="usbCam=1")
        {
            m_usbCam=true;
        }


    }
    configFile.close();

}


void OriginbotTeleop::saveFramePic(std::string name,unsigned long int frame,int err,errInfo& errObj)
{
    if(m_savePic&&saveCount<8000)
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
        std::string fileName=name+std::string(buffer)+std::to_string(err)+"_"+std::to_string(errObj.b_greenLawn)+"_"+std::to_string(frame)+".png";

        auto t0 = std::chrono::steady_clock::now();


        SaveTask task;
        task.filename = fileName;
        task.img = img_warp.clone();      // 必须 clone

        {
            std::lock_guard<std::mutex> lock(saveMutex);
            saveQueue.push(std::move(task));
        }

        saveCond.notify_one();


        //imwrite(fileName,img_warp);

        auto t1 = std::chrono::steady_clock::now();
        int grab_ms    = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();

        if(grab_ms>500)
        {
            timeoutLog(grab_ms,3);
        }


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
    unsigned long int lastTurnFrame=0;
    int triggerBackTurnNum=0;


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



    unsigned char *yuv_data=NULL;
    void *camera= sp_init_vio_module();
    ret = sp_open_camera_v2(camera, 0, -1, 1, &parms, widths, heights);

//    cv::VideoCapture cap(8);
//    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
//    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
//       if (!cap.isOpened()) {
//       printf("open usb cam err");
//       return;
//       }

    printf("saveImg is %d\n",m_savePic);
    sleep(2); // wait for isp stability

    // malloc buffer

     yuv_data = (unsigned char*)malloc(yuv_size * sizeof(char));


     if (ret != 0)
     {
         printf("[Error] sp_open_camera failed!\n");
         free(yuv_data);

         sp_vio_close(camera);
         sp_release_vio_module(camera);
         return;

     }

     int continueTurn=0;
     int moveNum=0;
     int backTurnNum=0;
     int stopNum=0;
     int cccc1;
     Size imageSize(640,480);
     ImageWidthReader reader;//=new ImageWidthReader();
     errInfo errObj;

     //auto startT = std::chrono::high_resolution_clock::now();
     //while(myCount<1)
     while(1)
     {
        ++myCount;

        //auto curT = std::chrono::high_resolution_clock::now();
        //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(curT-startT);
        //int frameTime=duration.count();
        //printf("frame:%ld %dms \n", myCount,frameTime);

        //startT = std::chrono::high_resolution_clock::now();
        auto t0 = std::chrono::steady_clock::now();
        sp_vio_get_yuv(camera, (char*)yuv_data, args.width, args.height, 100);
        auto t1 = std::chrono::steady_clock::now();
        int grab_ms    = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();

        if(grab_ms>500)
        {
            timeoutLog(grab_ms,1);
        }

        Mat matFrame=NV12ToBGR(yuv_data,args.width,args.height);


        //img_warp=imread("abc.png",1);

        adjustPic(matFrame);
        if(myCount<30)
        {
            if(myCount%5==0)
                stopRobot();
            continue;
        }



        double area;
        Point pos;
        std::vector<Point> v1;
        double allAreaRatio=0;
        auto getBlockT = std::chrono::steady_clock::now();


        v1=reader.getBlock(img_warp,m_svm,m_svm_green,imageSize,pos,allAreaRatio,cccc1,errObj);

        auto endBlockT = std::chrono::steady_clock::now();

        auto duraBlock = std::chrono::duration_cast<std::chrono::milliseconds>(endBlockT-getBlockT);

        int blockTime=duraBlock.count();

        printf("\n检测时间:%dms ret:%d\n",blockTime,cccc1);
        if(blockTime>500)
        {
            timeoutLog(blockTime,0);
            cccc1=blockTime;
            saveFramePic("/userdata/robotly/src/robotly/robotly_x3piCam/saveImg/timeout_",myCount,cccc1,errObj);
        }
        Rect rect(0,0,1,1);

        //printf("getBlock\n");
        Point robotP(1000,1000);


        if(v1.size()&&pos.y>80)
        {


            robotP=pic2robotAxis(pos,0.40322,1.5118,640,480);
            area=contourArea(v1);
            rect=boundingRect(v1);



            if(allAreaRatio>0.7||robotP.y<300)
            {
                ++stopNum;
                printf("物体 x:%d y:%d allAreaRatio:%.2f\n",pos.x,pos.y,allAreaRatio);
                if(moveNum>10)//&&backTurnNum<5)
                {

                    ++backTurnNum;

                    if(errObj.b_backMove)
                    {
                        saveFramePic("/userdata/robotly/src/robotly/robotly_x3piCam/saveImg/backT_backMove_",myCount,cccc1,errObj);
                        backMove();
                        timeoutLog(0,5);
                        errObj.b_preBlock=false;
                        continue;
                    }


                    ++triggerBackTurnNum;
                    if(triggerBackTurnNum==1)
                    {
                        stopRobot();
                        waitMs(800);
                        continue;
                    }

                    if(triggerBackTurnNum==3)
                    {
                        errObj.b_preBlock=true;
                    }
                    else
                    {
                        errObj.b_preBlock=false;
                    }

                    saveFramePic("/userdata/robotly/src/robotly/robotly_x3piCam/saveImg/backT_",myCount,cccc1,errObj);


                    Point backP=robotP;

                    if(errObj.greenRatio<5&&myCount==lastTurnFrame+1)
                    {
                        backP=lastTurnP;
                    }

                    backTurnRobot(backP.x>0?true:false);


                    stopRobot();
                }
                else
                {
                    stopRobot();
                }


            }
            else
            {
                stopNum=0;
                backTurnNum=0;
                triggerBackTurnNum=0;


                if(area>50)
                {

                    //lastTurnFrame=myCount;


                    ++continueTurn;


                    saveFramePic("/userdata/robotly/src/robotly/robotly_x3piCam/saveImg/turn_",myCount,cccc1,errObj);
                    lastTurnP=robotP;

                    lastTurnFrame=myCount;

                    turnRobot(robotP.x>0?true:false);
                }
                else
                {
                    ++moveNum;
                    moveRobot();
                }


            }

        }
        else if(v1.size()&&pos.y<=140)
        {
            stopNum=0;
            backTurnNum=0;
            triggerBackTurnNum=0;
            continueTurn=0;
            ++moveNum;
            if(moveNum>10000000)
                moveNum=11;
            moveRobot();

            if(moveNum%100==0)
            {
                 saveFramePic("/userdata/robotly/src/robotly/robotly_x3piCam/saveImg/auto_",myCount,0,errObj);
            }
        }

        if(robotP.x==1000)
        {
            stopNum=0;
            backTurnNum=0;
            triggerBackTurnNum=0;
            continueTurn=0;
            ++moveNum;
            if(moveNum>10000000)
                moveNum=11;
            moveRobot();

            if(moveNum%100==0)
            {
                saveFramePic("/userdata/robotly/src/robotly/robotly_x3piCam/saveImg/auto_",myCount,0,errObj);

            }
            printf("返回:无障碍，前进 frame:%ld \n",myCount);
        }
        else
        {

            printf("返回:rect:%d %d %d %d 障碍物:%d %d pos.y:%d frame:%ld \n",rect.x,rect.y,rect.width,rect.height,robotP.x,robotP.y,pos.y, myCount);

        }

    }

     if(yuv_data)
       free(yuv_data);

     //cap.release();



     RCLCPP_INFO(this->get_logger(),"你好节点");


}
void OriginbotTeleop::stopRobot()
{
    m_stopFg=true;
    speed=0;
    turn=0;
    cmdvel_.linear.x = speed * _speed_linear_x;
    cmdvel_.angular.z = turn * _speed_angular_z;
    auto t0 = std::chrono::steady_clock::now();
    pub_cmd->publish(cmdvel_);
    auto t1 = std::chrono::steady_clock::now();
    int publish_ms    = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    if(publish_ms>500)
    {
        timeoutLog(publish_ms,2);
    }
    printf("stop robot\n");
    //sleep(1);
    waitMs(200);
    //std::this_thread::sleep_for(std::chrono::milliseconds(500));
}


void OriginbotTeleop::turnRobot(bool isTurnLeft)
{

    if(continueTurnNum>5)
    {
        continueTurnNum=0;
        backTurnRobot(isTurnLeft);
        return;
    }


    //60 ＝30转/min
    //m_stopFg=true;

    //continueMoveNum=0;
    speed=60;
    turn=134;
    cmdvel_.linear.x = speed * _speed_linear_x;
    cmdvel_.angular.z = turn * _speed_angular_z;
    if(isTurnLeft==false)
    {
        cmdvel_.angular.z*=-1;
    }
    //if(m_sendMove)
     //   pub_cmd->publish(cmdvel_);
    if(isTurnLeft)
    {
         printf("左转 robot\n");
    }
    else
    {
         printf("右转 robot\n");
    }
    ++continueTurnNum;


    waitMs(1000);
    //stopRobot();
    //stopRobot();
    //sleep(2);
}

void OriginbotTeleop::moveRobot()
{

    continueBackTLeftNum=0;
    continueBackTRightNum=0;
    continueBackTurnNum=0;

    speed=60;//3;
    turn=0;
    cmdvel_.linear.x = speed * _speed_linear_x;
    cmdvel_.angular.z = turn * _speed_angular_z;
    auto t0 = std::chrono::steady_clock::now();
    pub_cmd->publish(cmdvel_);
    auto t1 = std::chrono::steady_clock::now();
    int publish_ms    = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    if(publish_ms>500)
    {
        timeoutLog(publish_ms,2);
    }
    if(m_stopFg)
    {
        waitMs(1000);
    }

    m_stopFg=false;

}
void OriginbotTeleop::backMove()
{
    stopRobot();
    speed=-40;
    turn=0;
    cmdvel_.linear.x = speed * _speed_linear_x;
    cmdvel_.angular.z = turn * _speed_angular_z;


    printf("后退 robot\n");

    waitMs(2000);

    stopRobot();
    backTurnRobot((continueBackTLeftNum+continueBackTRightNum)%2);
}

void OriginbotTeleop::backTurnRobot(bool isTurnLeft)
{


    stopRobot();

    speed=0;
    turn=200;
    cmdvel_.linear.x = speed * _speed_linear_x;
    cmdvel_.angular.z = turn * _speed_angular_z;
    if(isTurnLeft==false)
    {
        cmdvel_.angular.z*=-1;
    }
   ++continueBackTurnNum;



    if(isTurnLeft)
    {
         ++continueBackTLeftNum;
         printf("原地倒向左转 robot\n");


    }
    else
    {
         ++continueBackTRightNum;
         printf("原地倒向右转 robot\n");

    }



    if(continueBackTurnNum>6&&continueBackTurnNum<36)
    {
        if(cmdvel_.angular.z<0)
            cmdvel_.angular.z*=-1;
    }
    if(continueBackTurnNum>35)
    {
        if(cmdvel_.angular.z>0)
            cmdvel_.angular.z*=-1;
    }


    if(continueBackTurnNum==4)
    {
        //waitMs(1400);
        timeoutLog(0,4);
    }

    if(continueBackTurnNum<7)
    {
        //转小角度
        waitMs(1400);
    }
    else
    {
        waitMs(1000);
    }


    //sleep(2);
    stopRobot();
    stopRobot();
    stopRobot();
}

void OriginbotTeleop::waitMs(int ms)
{
    int step=ms/100;
    for(int i=0;i<step;i++)
    {
        auto t0 = std::chrono::steady_clock::now();
        pub_cmd->publish(cmdvel_);
        auto t1 = std::chrono::steady_clock::now();
        int publish_ms    = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
        if(publish_ms>500)
        {
            timeoutLog(publish_ms,2);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void OriginbotTeleop::rotate_image_folder(const std::string& base_path)
{
    size_t max_files=2000;
    std::string save_path = base_path + "/saveImg";
    std::string backup_path = base_path + "/saveImg_back";

    DIR* dir = opendir(save_path.c_str());
    if (!dir)
    {
        mkdir(save_path.c_str(), 0755);
        sync();

    }

    size_t file_count = 0;
    struct dirent* entry;

    while ((entry = readdir(dir)) != nullptr)
    {
        if (entry->d_type == DT_REG)
            ++file_count;
    }

    closedir(dir);

    if (file_count >= max_files)
    {
        try
        {
            // 如果 backup 存在，先删除
            struct stat st;
            if (stat(backup_path.c_str(), &st) == 0)
            {
                std::string cmd = "rm -rf " + backup_path;
                system(cmd.c_str());
            }

            //重命名目录（原子）
            rename(save_path.c_str(), backup_path.c_str());

            //创建新目录
            mkdir(save_path.c_str(), 0755);

            sync();   // SD 卡强制刷盘
        }
        catch (const std::exception& e)
        {

        }
    }


}

void OriginbotTeleop::timeoutLog(int ms,int index)
{
    std::time_t now = std::time(nullptr);
    std::tm tm1;
    localtime_r(&now, &tm1); // 将时间转换为本地时间，线程安全
    char buffer[80]={0};
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H%M%S_", &tm1);

    FILE *fp = fopen("/userdata/robotly/src/robotly/robotly_x3piCam/timeout.txt", "a");
    if (fp != NULL) {

        if(index==0)
        {
            fprintf(fp, "%s_detect_%dms\n",buffer, ms);
        }
        else if(index==1)
        {
            fprintf(fp, "%s_grab_%dms\n",buffer, ms);
        }
        else if(index==2)
        {
            fprintf(fp, "%s_publish_%dms\n",buffer, ms);
        }
        else if(index==3)
        {
            fprintf(fp, "%s_imwrite_%dms\n",buffer, ms);
        }
        else if(index==4)
        {
            fprintf(fp, "%s_bigBackT_%dms\n",buffer, ms);
        }
        else if(index==5)
        {
            fprintf(fp, "%s_backMove_%dms\n",buffer, ms);
        }


        fclose(fp);
    }
}

void OriginbotTeleop::saveImageThread()
{
    while (!exitThread)
    {
        SaveTask task;

        {
            std::unique_lock<std::mutex> lock(saveMutex);

            saveCond.wait(lock, [this]{
                return exitThread || !saveQueue.empty();
            });

            if (exitThread)
                break;

            task = std::move(saveQueue.front());
            saveQueue.pop();
        }

        cv::imwrite(task.filename, task.img);
    }
}