/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

void cv::cuda::blendLinear(InputArray, InputArray, InputArray, InputArray, OutputArray, Stream&) { throw_no_cuda(); }

#else

////////////////////////////////////////////////////////////////////////
// blendLinear

namespace cv { namespace cuda { namespace device
{
    namespace blend
    {
        template <typename T>
        void blendLinearCaller(int rows, int cols, int cn, PtrStep<T> img1, PtrStep<T> img2, PtrStepf weights1, PtrStepf weights2, PtrStep<T> result, cudaStream_t stream);

        void blendLinearCaller8UC4(int rows, int cols, PtrStepb img1, PtrStepb img2, PtrStepf weights1, PtrStepf weights2, PtrStepb result, cudaStream_t stream);
    }
}}}

using namespace ::cv::cuda::device::blend;

void cv::cuda::blendLinear(InputArray _img1, InputArray _img2, InputArray _weights1, InputArray _weights2,
                          OutputArray _result, Stream& stream)
{
    GpuMat img1 = _img1.getGpuMat();
    GpuMat img2 = _img2.getGpuMat();

    GpuMat weights1 = _weights1.getGpuMat();
    GpuMat weights2 = _weights2.getGpuMat();

    CV_Assert( img1.size() == img2.size() );
    CV_Assert( img1.type() == img2.type() );
    CV_Assert( weights1.size() == img1.size() );
    CV_Assert( weights2.size() == img2.size() );
    CV_Assert( weights1.type() == CV_32FC1 );
    CV_Assert( weights2.type() == CV_32FC1 );

    const Size size = img1.size();
    const int depth = img1.depth();
    const int cn = img1.channels();

    _result.create(size, CV_MAKE_TYPE(depth, cn));
    GpuMat result = _result.getGpuMat();

    switch (depth)
    {
    case CV_8U:
        if (cn != 4)
            blendLinearCaller<uchar>(size.height, size.width, cn, img1, img2, weights1, weights2, result, StreamAccessor::getStream(stream));
        else
            blendLinearCaller8UC4(size.height, size.width, img1, img2, weights1, weights2, result, StreamAccessor::getStream(stream));
        break;
    case CV_32F:
        blendLinearCaller<float>(size.height, size.width, cn, img1, img2, weights1, weights2, result, StreamAccessor::getStream(stream));
        break;
    default:
        CV_Error(cv::Error::StsUnsupportedFormat, "bad image depth in linear blending function");
    }
}

// todo move it out of blend.cpp
// must figure out how to do that since it is complicated
void cv::cuda::adaptiveThreshold(InputArray _src, OutputArray _dst, double maxValue,
    int method, int type, int blockSize, double delta, Stream& stream) {

    // only this method is supported
    // box border - based method
    CV_Assert(method == cv::AdaptiveThresholdTypes::ADAPTIVE_THRESH_MEAN_C);

    // adaptiveThreshold checks
    CV_Assert(_src.type() == CV_8UC1);
    CV_Assert(blockSize % 2 == 1 && blockSize > 1);

    // init vars
    GpuMat src = _src.getGpuMat();

    _dst.create(_src.size(), CV_8UC1);
    GpuMat dst = _dst.getGpuMat();

    /*
    // -- create and apply a box filter
    Ptr<Filter> boxFilter = createBoxFilter(_src.type(), _dst.type(), Size(blockSize, blockSize), Point(-1, -1), BORDER_REPLICATE | BORDER_ISOLATED);
    boxFilter->apply(_src, _dst);
    */


    // setup arguments
    const NppiSize nppSize = NppiSize{ src.size().width, src.size().height };
    const NppiPoint offset = NppiPoint{ 0, 0 }; // idk todo check it
    const NppiSize roi = NppiSize{ src.cols, src.rows };
    const NppiSize maskSize = NppiSize{ blockSize, blockSize };

    Npp8u le = 0;
    Npp8u gt = static_cast<Npp8u> (maxValue);

    switch (type)
    {
    case THRESH_BINARY:
        break;
    case THRESH_BINARY_INV:
        // swap max and min values for pixels
        std::swap(le, gt);
        break;
    default:
        CV_Error(cv::Error::StsBadArg, "unsupported threshold type");
        break;
    }

    // call NPP adaptive threshold
    // cudaSafeCall does not work there, resolves to a wrong macro
    checkNppError(
        nppiFilterThresholdAdaptiveBoxBorder_8u_C1R(
            src.ptr<Npp8u>(), static_cast<int> (src.step), nppSize, offset,
            dst.ptr<Npp8u>(), static_cast<int> (dst.step), roi, maskSize,
            static_cast<Npp32f> (delta), gt, le, NPP_BORDER_REPLICATE
        ), __FILE__, __LINE__, "nppiFilterThresholdAdaptiveBoxBorder_8u_C1R"
    );

    // finalize
    if (stream == 0) {
        checkCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__, "cudaDeviceSynchronize");
    }

}

#endif
