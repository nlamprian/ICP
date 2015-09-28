/*! \file ocl_icp_reg.cpp
 *  \brief Defines the classes for setting up the OpenCL %ICP pipeline.
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

#include <ocl_icp_reg.hpp>


const std::vector<std::string> kernel_files_rbc = { "kernels/RBC/reduce_kernels.cl", 
                                                    "kernels/RBC/scan_kernels.cl", 
                                                    "kernels/RBC/rbc_kernels.cl" };

const std::vector<std::string> kernel_files_icp = { "kernels/ICP/reduce_kernels.cl", 
                                                    "kernels/ICP/icp_kernels.cl" };


/*! \param[out] glPC4DBuffer OpenGL vertex buffer id for the geometry points. 
 *                           The buffer holds the information for two point clouds.
 *  \param[out] glRGBABuffer OpenGL vertex buffer id for the color points. 
 *                           The buffer holds the information for two point clouds.
 *  \param[in] width width (in pixels) of the associated point clouds.
 *  \param[in] height height (in pixels) of the associated point clouds.
 */
CLEnvGL::CLEnvGL (GLuint *glPC4DBuffer, GLuint *glRGBABuffer, int width, int height) : 
    CLEnv (), glPC4DBuffer (glPC4DBuffer), glRGBABuffer (glRGBABuffer), width (width), height (height)
{
    addContext (0, true);
    addQueueGL (0);
    addProgram (0, kernel_files_rbc);
    addProgram (0, kernel_files_icp);
}


/*! \note Do not call directly. `initGLMemObjects` is called by `addContext`
 *        when creating the GL-shared CL context.
 */
void CLEnvGL::initGLMemObjects ()
{
    glGenBuffers (1, glPC4DBuffer);
    glBindBuffer (GL_ARRAY_BUFFER, *glPC4DBuffer);
    glBufferData (GL_ARRAY_BUFFER, 2 * (width * height * sizeof (cl_float4)), NULL, GL_DYNAMIC_DRAW);
    glGenBuffers (1, glRGBABuffer);
    glBindBuffer (GL_ARRAY_BUFFER, *glRGBABuffer);
    glBufferData (GL_ARRAY_BUFFER, 2 * (width * height * sizeof (cl_float4)), NULL, GL_DYNAMIC_DRAW);
    glBindBuffer (GL_ARRAY_BUFFER, 0);
}


/*! \details Initializes the OpenCL environment, the OpenGL buffers, and the processing classes.
 *  
 *  \param[in] glPC4DBuffer OpenGL vertex buffer id for the geometry points. 
 *                          The buffer holds the information for two point clouds.
 *  \param[in] glRGBABuffer OpenGL vertex buffer id for the color points. 
 *                          The buffer holds the information for two point clouds.
 */
template <cl_algo::ICP::ICPStepConfigT RC, cl_algo::ICP::ICPStepConfigW WC>
ICPReg<RC, WC>::ICPReg (GLuint *glPC4DBuffer, GLuint *glRGBABuffer) : 
    width (640), height (480), n (640 * 480), m (16384), r (256), 
    env (glPC4DBuffer, glRGBABuffer, width, height), 
    infoRBC (0, 0, 0, { 0 }, 0), infoICP (0, 0, 0, { 0 }, 1), 
    context (env.getContext (0)), queue (env.getQueue (0, 0)), 
    glPC4DBuffer (glPC4DBuffer), glRGBABuffer (glRGBABuffer), 
    blue { 0.f, 0.15f, 1.f, 1.f }, green { 0.3f, 1.f, 0.f, 1.f }, dummy { 0.f, 0.f, 0.f, 0.f }, 
    vBlue (n, *(cl_float4 *) blue), vGreen (n, *(cl_float4 *) green), vDummy (n, *(cl_float4 *) dummy), 
    a (2e2f), c (1e-6f), max_iterations (40), angle_threshold (0.001), translation_threshold (0.01), 
    fLM (env, infoICP), mLM (env, infoICP), reg (env, infoRBC, infoICP), transform (env, infoICP)
{
    // OpenGL buffer copy parameters
    src_origin_g[0] = 0;                  src_origin_g[1] = 0; src_origin_g[2] = 0;
    src_origin_c[0] = sizeof (cl_float4); src_origin_c[1] = 0; src_origin_c[2] = 0;
    dst_origin_1[0] = 0; dst_origin_1[1] = 0; dst_origin_1[2] = 0;
    dst_origin_2[0] = 0; dst_origin_2[1] = n; dst_origin_2[2] = 0;
    region[0] = sizeof (cl_float4); region[1] = n; region[2] = 1;

    // Create GL-shared buffers
    dBufferGL.emplace_back (context, CL_MEM_WRITE_ONLY, *glPC4DBuffer);
    dBufferGL.emplace_back (context, CL_MEM_WRITE_ONLY, *glRGBABuffer);

    // Initialize classes
    fLM.get (cl_algo::ICP::ICPLMs::Memory::D_OUT) = 
        cl::Buffer (context, CL_MEM_READ_WRITE, m * sizeof (cl_float8));
    fLM.init (cl_algo::ICP::Staging::I);

    mLM.get (cl_algo::ICP::ICPLMs::Memory::D_OUT) = 
        cl::Buffer (context, CL_MEM_READ_WRITE, m * sizeof (cl_float8));
    mLM.init (cl_algo::ICP::Staging::I);

    reg.get (cl_algo::ICP::ICPStep<RC, WC>::Memory::D_IN_F) = fLM.get (cl_algo::ICP::ICPLMs::Memory::D_OUT);
    reg.get (cl_algo::ICP::ICPStep<RC, WC>::Memory::D_IN_M) = mLM.get (cl_algo::ICP::ICPLMs::Memory::D_OUT);
    reg.init (m, r, a, c, max_iterations, angle_threshold, translation_threshold, cl_algo::ICP::Staging::NONE);

    transform.get (cl_algo::ICP::ICPTransform
        <cl_algo::ICP::ICPTransformConfig::QUATERNION>::Memory::D_IN_M) = 
        mLM.get (cl_algo::ICP::ICPLMs::Memory::D_IN);
    transform.get (cl_algo::ICP::ICPTransform
        <cl_algo::ICP::ICPTransformConfig::QUATERNION>::Memory::D_IN_T) = 
        reg.get (cl_algo::ICP::ICPStep<RC, WC>::Memory::D_IO_T);
    transform.init (n, cl_algo::ICP::Staging::NONE);
}


/*! \brief Initializes the OpenGL buffers.
 *  
 *  \param[in] pc8d1 fixed point cloud.
 *  \param[in] pc8d2 moving point cloud.
 */
template <cl_algo::ICP::ICPStepConfigT RC, cl_algo::ICP::ICPStepConfigW WC>
void ICPReg<RC, WC>::init (std::vector<cl_float8> pc8d1, std::vector<cl_float8> pc8d2)
{
    fLM.write (cl_algo::ICP::ICPLMs::Memory::D_IN, (cl_float *) pc8d1.data ());
    mLM.write (cl_algo::ICP::ICPLMs::Memory::D_IN, (cl_float *) pc8d2.data ());

    fLM.run ();
    mLM.run ();


    glFinish ();  // Wait for OpenGL pending operations on buffers to finish

    // Take ownership of OpenGL buffers
    queue.enqueueAcquireGLObjects ((std::vector<cl::Memory> *) &dBufferGL);

    // Initialize OpenGL buffers
    queue.enqueueCopyBufferRect ((cl::Buffer &) fLM.get (cl_algo::ICP::ICPLMs::Memory::D_IN), dBufferGL[0], src_origin_g, dst_origin_1, region, sizeof (cl_float8), 0, sizeof (cl_float4), 0);
    queue.enqueueCopyBufferRect ((cl::Buffer &) mLM.get (cl_algo::ICP::ICPLMs::Memory::D_IN), dBufferGL[0], src_origin_g, dst_origin_2, region, sizeof (cl_float8), 0, sizeof (cl_float4), 0);
    queue.enqueueCopyBufferRect ((cl::Buffer &) fLM.get (cl_algo::ICP::ICPLMs::Memory::D_IN), dBufferGL[1], src_origin_c, dst_origin_1, region, sizeof (cl_float8), 0, sizeof (cl_float4), 0);
    queue.enqueueCopyBufferRect ((cl::Buffer &) mLM.get (cl_algo::ICP::ICPLMs::Memory::D_IN), dBufferGL[1], src_origin_c, dst_origin_2, region, sizeof (cl_float8), 0, sizeof (cl_float4), 0);
    // queue.enqueueWriteBuffer (dBufferGL[1], CL_FALSE, 0, n * sizeof (cl_float4), vBlue.data ());
    // queue.enqueueWriteBuffer (dBufferGL[1], CL_FALSE, n * sizeof (cl_float4), n * sizeof (cl_float4), vGreen.data ());

    // Give up ownership of OpenGL buffers
    queue.enqueueReleaseGLObjects ((std::vector<cl::Memory> *) &dBufferGL);

    queue.finish ();
}


/*! \brief Performs an %ICP registration and transforms the moving point cloud.
 *  \details Refines an initial estimation of the transformation between 
 *           the two point clouds, and displays relevant parameters.
 */
template <cl_algo::ICP::ICPStepConfigT RC, cl_algo::ICP::ICPStepConfigW WC>
void ICPReg<RC, WC>::registerPC ()
{
    static clutils::CPUTimer<double, std::milli> timer;

    reg.buildRBC ();  // Builde the RBC data structure
    
    timer.start ();
    reg.run ();  // Perform the ICP registration
    timer.stop ();

    transform.run ();  // Transform the moving point cloud

    glFinish ();  // Wait for OpenGL pending operations on buffers to finish

    // Take ownership of OpenGL buffers
    queue.enqueueAcquireGLObjects ((std::vector<cl::Memory> *) &dBufferGL);

    // Transfer the transformed point cloud to the OpenGL buffer
    queue.enqueueCopyBufferRect (
        (cl::Buffer &) transform.get (cl_algo::ICP::ICPTransform
            <cl_algo::ICP::ICPTransformConfig::QUATERNION>::Memory::D_OUT), 
        dBufferGL[0], src_origin_g, dst_origin_2, region, 
        sizeof (cl_float8), 0, sizeof (cl_float4), 0);

    // Give up ownership of OpenGL buffers
    queue.enqueueReleaseGLObjects ((std::vector<cl::Memory> *) &dBufferGL);

    queue.finish ();

    // Print results ===========================================================
    
    double sinth_2 = reg.q.vec ().norm ();
    double angle = 180.f / M_PI * 2 * std::atan2 (sinth_2, reg.q.w ());
    Eigen::Vector3f axis ((sinth_2 == 0.0) ? Eigen::Vector3f::Zero () : reg.q.vec ().normalized ());

    std::cout << std::endl << "================" << std::endl << std::endl;
    std::cout << "    Iterations            :    " << reg.k << std::endl;
    std::cout << "    Latency               :    " << timer.duration () << " ms" << std::endl;
    std::cout << "    Rotation angle        :    " << angle << " degrees" << std::endl;
    std::cout << "    Rotation axis         :    " << axis.transpose () << std::endl;
    std::cout << "    Translation vector    :    " << reg.t.transpose () << std::endl;
    std::cout << "    Scale                 :    " << reg.s << std::endl;
}


/*! \brief Instantiation that uses the Eigen library to estimate the rotation, and considers regular residual errors. */
template class ICPReg<cl_algo::ICP::ICPStepConfigT::EIGEN, cl_algo::ICP::ICPStepConfigW::REGULAR>;
/*! \brief Instantiation that uses the Eigen library to estimate the rotation, and considers weighted residual errors. */
template class ICPReg<cl_algo::ICP::ICPStepConfigT::EIGEN, cl_algo::ICP::ICPStepConfigW::WEIGHTED>;
/*! \brief Instantiation that uses the Power Method to estimate the rotation, and considers regular residual errors. */
template class ICPReg<cl_algo::ICP::ICPStepConfigT::POWER_METHOD, cl_algo::ICP::ICPStepConfigW::REGULAR>;
/*! \brief Instantiation that uses the Power Method to estimate the rotation, and considers weighted residual errors. */
template class ICPReg<cl_algo::ICP::ICPStepConfigT::POWER_METHOD, cl_algo::ICP::ICPStepConfigW::WEIGHTED>;
