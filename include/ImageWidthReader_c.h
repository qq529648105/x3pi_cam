#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  ifdef IMAGE_WIDTH_READER_EXPORTS
#    define IWR_API __declspec(dllexport)
#  else
#    define IWR_API __declspec(dllimport)
#  endif
#else
#  define IWR_API __attribute__((visibility("default")))
#endif


struct errInfo
{
    bool b_greenLawn;
    bool b_backMove;
    bool b_preBlock;
    float greenRatio;
    errInfo()
    {
        b_greenLawn=false;
        b_preBlock=false;
        greenRatio=0;
    }
};

typedef struct ImageWidthReaderHandle ImageWidthReaderHandle;

IWR_API ImageWidthReaderHandle* IWR_create();
IWR_API void IWR_destroy(ImageWidthReaderHandle* h);

IWR_API int  IWR_getImageWidth(ImageWidthReaderHandle* h, const char* path);
IWR_API void IWR_getImageSize(ImageWidthReaderHandle* h, const char* path,
                              int* w, int* hgt);
IWR_API std::vector<cv::Point> imageGetBlock(ImageWidthReaderHandle *h, cv::Mat &src, cv::Ptr<cv::ml::SVM> m_svm, cv::Ptr<cv::ml::SVM> m_svm_green, cv::Size imageSize, cv::Point& pos, double &allAreaRatio, int& c1, errInfo &err);

#ifdef __cplusplus
}
#endif
