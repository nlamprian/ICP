/*! \file ocl_icp_reg.hpp
 *  \brief Declares the classes for setting up the OpenCL %ICP pipeline.
 *  \author Nick Lamprianidis
 *  \version 1.1.0
 *  \date 2015
 *  \copyright The MIT License (MIT)
 *  \par
 *  Copyright (c) 2015 Nick Lamprianidis
 *  \par
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  \par
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *  \par
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

#ifndef OCL_ICP_SBS_HPP
#define OCL_ICP_SBS_HPP

#include <GL/glew.h>  // Add before CLUtils.hpp
#include <CLUtils.hpp>
#include <ICP/algorithms.hpp>
#include <eigen3/Eigen/Dense>


/*! \brief Creates an OpenCL environment with CL-GL interoperability. */
class CLEnvGL : public clutils::CLEnv
{
public:
    /*! \brief Initializes the OpenCL environment. */
    CLEnvGL (GLuint *glPC4DBuffer, GLuint *glRGBABuffer, int width, int height);

private:
    /*! \brief Initializes the OpenGL memory buffers. */
    void initGLMemObjects ();

    GLuint *glPC4DBuffer, *glRGBABuffer;
    int width, height;
};


/*! \brief Performs the %ICP iterations.
 *  \details Estimates, step by step, the homogeneous transformation between two point clouds, 
 *           and transforms the relevant point cloud according to that transformation.
 */
template <cl_algo::ICP::ICPStepConfigT RC, cl_algo::ICP::ICPStepConfigW WC>
class ICPReg
{
public:
    ICPReg (GLuint *glPC4DBuffer, GLuint *glRGBABuffer);
    void init (std::vector<cl_float8> pc8d1, std::vector<cl_float8> pc8d2);
    void registerPC ();

private:
    unsigned int width, height, n, m, r;
    CLEnvGL env;
    clutils::CLEnvInfo<1> infoRBC, infoICP;
    cl::Context &context;
    cl::CommandQueue &queue;
    
    GLuint *glPC4DBuffer, *glRGBABuffer;
    cl_float blue[4], green[4], dummy[4];
    std::vector<cl_float4> vBlue, vGreen, vDummy;
    std::vector<cl::BufferGL> dBufferGL;
    
    cl::size_t<3> src_origin_g, src_origin_c;
    cl::size_t<3> dst_origin_1, dst_origin_2;
    cl::size_t<3> region;
    float a, c;
    unsigned int max_iterations;
    double angle_threshold;
    double translation_threshold;

    cl_algo::ICP::ICPLMs fLM, mLM;
    cl_algo::ICP::ICP<RC, WC> reg;
    cl_algo::ICP::ICPTransform<cl_algo::ICP::ICPTransformConfig::QUATERNION> transform;

};

#endif  // OCL_ICP_SBS_HPP
