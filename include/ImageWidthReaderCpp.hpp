#pragma once
#include "ImageWidthReader_c.h"

class ImageWidthReader {
public:
    ImageWidthReader() : h(IWR_create()) {}
    ~ImageWidthReader() { IWR_destroy(h); }

    int getImageWidth(const std::string& p) {
        return IWR_getImageWidth(h, p.c_str());
    }
    std::vector<cv::Point> getBlock(cv::Mat &src, cv::Ptr<cv::ml::SVM> m_svm,cv::Ptr<cv::ml::SVM> m_svm_green,cv::Size imageSize,cv::Point& pos, double &allAreaRatio,int& c1,errInfo& err)
    {
        return imageGetBlock(h,src,m_svm,m_svm_green,imageSize,pos,allAreaRatio,c1,err);
    }


    cv::Size getImageSize(const std::string& p) {
        int w, hgt;
        IWR_getImageSize(h, p.c_str(), &w, &hgt);
        return {w, hgt};
    }



private:
    ImageWidthReaderHandle* h;
};
