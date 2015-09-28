/*! \file testsICP.cpp
 *  \brief Google Test Unit Tests for the `%ICP` kernels.
 *  \note Use the `--profiling` flag to enable profiling of the kernels.
 *  \note The benchmarks in these tests are against naive CPU implementations 
 *        of the associated algorithms. They are used only for testing purposes, 
 *        and not for examining the performance of their GPU alternatives.
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

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <random>
#include <limits>
#include <cmath>
#include <gtest/gtest.h>
#include <CLUtils.hpp>
#include <RBC/data_types.hpp>
#include <ICP/algorithms.hpp>
#include <ICP/tests/helper_funcs.hpp>


// Kernel filenames
const std::string kernel_filename_reduce { "kernels/ICP/reduce_kernels.cl" };
const std::string kernel_filename_icp    { "kernels/ICP/icp_kernels.cl"    };

// Uniform random number generators
namespace ICP
{
    extern std::function<unsigned char ()> rNum_0_255;
    extern std::function<unsigned short ()> rNum_0_10000;
    extern std::function<float ()> rNum_R_0_1;
}

bool profiling;  // Flag to enable profiling of the kernels (--profiling)


/*! \brief Tests the **icpGetLMs** kernel.
 *  \details The kernel samples a set of landmarks.
 */
TEST (ICP, getLMs)
{
    try
    {
        const unsigned int n = 640 * 480;  // 307200
        const unsigned int m = 1 << 14;    //  16384
        const unsigned int d = 8;
        // const unsigned int bufferInSize = n * sizeof (cl_float8);
        // const unsigned int bufferOutSize = m * sizeof (cl_float8);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_icp);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::ICP::ICPLMs glm (clEnv, info);
        glm.init ();

        // Initialize data (writes on staging buffer directly)
        std::generate (glm.hPtrIn, glm.hPtrIn + n * d, ICP::rNum_0_10000);
        // ICP::printBufferF ("Original:", glm.hPtrIn, d, n, 3);

        glm.write ();  // Copy data to device
        
        glm.run ();  // Execute kernels (13 us)
        
        cl_float *results = (cl_float *) glm.read ();  // Copy results to host
        // ICP::printBufferF ("Received:", results, d, m, 3);

        // Produce reference landmarks
        cl_float *refLM = new cl_float[m * d];
        ICP::cpuICPLMs (glm.hPtrIn, refLM);
        // ICP::printBufferF ("Expected:", refLM, d, m, 3);

        // Verify landmarks
        for (uint j = 0; j < m; ++j)
            for (uint k = 0; k < d; ++k)
                ASSERT_EQ (refLM[j * d + k], results[j * d + k]);

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                ICP::cpuICPLMs (glm.hPtrIn, refLM);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = glm.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "ICPLMs");
        }

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}


/*! \brief Tests the **icpGetReps** kernel.
 *  \details The kernel samples a set of representatives.
 */
TEST (ICP, getReps)
{
    try
    {
        const unsigned int m  = 1 << 14;  //  16384
        const unsigned int nr = 1 <<  8;  //    256
        const unsigned int d = 8;
        // const unsigned int bufferInSize = m * sizeof (cl_float8);
        // const unsigned int bufferOutSize = nr * sizeof (cl_float8);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_icp);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::ICP::ICPReps grp (clEnv, info);
        grp.init (nr);

        // Initialize data (writes on staging buffer directly)
        std::generate (grp.hPtrIn, grp.hPtrIn + m * d, ICP::rNum_0_10000);
        // ICP::printBufferF ("Original:", grp.hPtrIn, d, m, 3);

        grp.write ();  // Copy data to device
        
        grp.run ();  // Execute kernels (7 us)
        
        cl_float *results = (cl_float *) grp.read ();  // Copy results to host
        // ICP::printBufferF ("Received:", results, d, nr, 3);

        // Produce reference representatives
        cl_float *refRep = new cl_float[nr * d];
        ICP::cpuICPReps (grp.hPtrIn, refRep, nr);
        // ICP::printBufferF ("Expected:", refRep, d, nr, 3);

        // Verify representatives
        for (uint j = 0; j < nr; ++j)
            for (uint k = 0; k < d; ++k)
                ASSERT_EQ (refRep[j * d + k], results[j * d + k]);

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                ICP::cpuICPReps (grp.hPtrIn, refRep, nr);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = grp.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "ICPReps");
        }

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}


/*! \brief Tests the **icpComputeReduceWeights** kernel.
 *  \details The kernel computes weights and their sum.
 */
TEST (ICP, icpComputeReduceWeights)
{
    try
    {
        const std::vector<std::string> kernel_files = { kernel_filename_reduce, 
                                                        kernel_filename_icp };

        const unsigned int n = 1 << 14;  // 16384
        // const unsigned int bufferInSize = n * sizeof (rbc_dist_id);
        // const unsigned int bufferOutWSize = n * sizeof (cl_float);
        // const unsigned int bufferOutSWSize = sizeof (cl_float);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_files);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::ICP::ICPWeights w (clEnv, info);
        w.init (n);

        // Initialize data (writes on staging buffer directly)
        unsigned int idx = 0;
        std::generate (w.hPtrIn, w.hPtrIn + n, 
            [&]()
            {
                rbc_dist_id dist;
                dist.dist = ICP::rNum_R_0_1 ();
                dist.id = 0;
                return dist;
            }
        );
        // ICP::printBufferF ("Original:", (float *) w.hPtrIn, 2, n, 3);

        w.write ();  // Copy data to device
        
        w.run ();  // Execute kernels (~ 13 us)
        
        // Copy results to host
        cl_float *W = (cl_float *) w.read (cl_algo::ICP::ICPWeights::Memory::H_OUT_W, CL_FALSE);
        cl_double *SUM_W = (cl_double *) w.read (cl_algo::ICP::ICPWeights::Memory::H_OUT_SUM_W);
        // ICP::printBufferF ("Received W:", W, 1, n, 3);
        // ICP::printBufferF ("Received SUM_W:", SUM_W, 1, 1, 3);

        // Produce reference weights and their sum
        cl_float *refW = new cl_float[n];
        cl_double refSUM_W[1];
        ICP::cpuICPWeights (w.hPtrIn, refW, refSUM_W, n);
        // ICP::printBufferF ("Expected W:", refW, 1, n, 3);
        // ICP::printBufferF ("Expected SUM_W:", refSUM_W, 1, 1, 3);

        // Verify weights and sum
        float eps = 4200 * std::numeric_limits<float>::epsilon ();  // 0.000500679
        ASSERT_LT (std::abs (refSUM_W[0] - SUM_W[0]), eps);
        eps = 42 * std::numeric_limits<float>::epsilon ();  // 5.00679e-06
        for (uint i = 0; i < n; ++i)
            ASSERT_LT (std::abs (refW[i] - W[i]), eps);

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                ICP::cpuICPWeights (w.hPtrIn, refW, refSUM_W, n);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = w.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "ICPWeights");
        }

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}


/*! \brief Tests the **icpMean** kernel.
 *  \details The kernel computes the means of sets of points.
 */
TEST (ICP, icpMean)
{
    try
    {
        const unsigned int n = 1 << 14;  // 16384
        const unsigned int d = 8;
        // const unsigned int bufferInSize = n * sizeof (cl_float8);
        // const unsigned int bufferOutSize = 2 * sizeof (cl_float4);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_icp);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        const cl_algo::ICP::ICPMeanConfig C = cl_algo::ICP::ICPMeanConfig::REGULAR;
        cl_algo::ICP::ICPMean<C> mean (clEnv, info);
        mean.init (n);

        // Initialize data (writes on staging buffer directly)
        std::generate (mean.hPtrInF, mean.hPtrInF + n * d, ICP::rNum_0_10000);
        std::generate (mean.hPtrInM, mean.hPtrInM + n * d, ICP::rNum_0_255);
        // ICP::printBufferF ("Original F:", mean.hPtrInF, d, n, 3);
        // ICP::printBufferF ("Original M:", mean.hPtrInM, d, n, 3);

        // Copy data to device
        mean.write (cl_algo::ICP::ICPMean<C>::Memory::D_IN_F);
        mean.write (cl_algo::ICP::ICPMean<C>::Memory::D_IN_M);
        
        mean.run ();  // Execute kernels (~ 20 us)
        
        cl_float *results = (cl_float *) mean.read ();  // Copy results to host
        // ICP::printBufferF ("Received:", results, 4, 2, 3);

        // Produce reference mean vector
        cl_float refMean[8];
        ICP::cpuICPMean (mean.hPtrInF, mean.hPtrInM, refMean, n);
        // ICP::printBufferF ("Expected:", refMean, 4, 2, 3);

        // Verify mean vector
        float eps = 420000 * std::numeric_limits<float>::epsilon ();  // 0.0500679
        for (uint i = 0; i < 8; ++i)
            ASSERT_LT (std::abs (refMean[i] - results[i]), eps);

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                ICP::cpuICPMean (mean.hPtrInF, mean.hPtrInM, refMean, n);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = mean.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "ICPMean<REGULAR>");
        }

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}


/*! \brief Tests the **icpMean_Weighted** kernel.
 *  \details The kernel computes the means of sets of points.
 */
TEST (ICP, icpMean_Weighted)
{
    try
    {
        const unsigned int n = 1 << 14;  // 16384
        const unsigned int d = 8;
        // const unsigned int bufferInFMSize = n * sizeof (cl_float8);
        // const unsigned int bufferInWSize = n * sizeof (cl_float);
        // const unsigned int bufferInSWSize = sizeof (cl_double);
        // const unsigned int bufferOutSize = 2 * sizeof (cl_float4);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_icp);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        const cl_algo::ICP::ICPMeanConfig C = cl_algo::ICP::ICPMeanConfig::WEIGHTED;
        cl_algo::ICP::ICPMean<C> mean (clEnv, info);
        mean.init (n);

        // Initialize data (writes on staging buffer directly)
        std::generate (mean.hPtrInF, mean.hPtrInF + n * d, ICP::rNum_0_10000);
        std::generate (mean.hPtrInM, mean.hPtrInM + n * d, ICP::rNum_0_255);
        std::generate (mean.hPtrInW, mean.hPtrInW + n, ICP::rNum_R_0_1);
        mean.hPtrInSW[0] = std::accumulate (mean.hPtrInW, mean.hPtrInW + n, 0.0);
        // ICP::printBufferF ("Original F:", mean.hPtrInF, d, n, 3);
        // ICP::printBufferF ("Original M:", mean.hPtrInM, d, n, 3);
        // ICP::printBufferF ("Original W:", mean.hPtrInW, 1, n, 3);

        // Copy data to device
        mean.write (cl_algo::ICP::ICPMean<C>::Memory::D_IN_F);
        mean.write (cl_algo::ICP::ICPMean<C>::Memory::D_IN_M);
        mean.write (cl_algo::ICP::ICPMean<C>::Memory::D_IN_W);
        mean.write (cl_algo::ICP::ICPMean<C>::Memory::D_IN_SUM_W);
        
        mean.run ();  // Execute kernels (~ 25 us)
        
        cl_float *results = (cl_float *) mean.read ();  // Copy results to host
        // ICP::printBufferF ("Received:", results, 4, 2, 3);

        // Produce reference mean vector
        cl_float refMean[8];
        ICP::cpuICPMeanWeighted (mean.hPtrInF, mean.hPtrInM, refMean, mean.hPtrInW, n);
        // ICP::printBufferF ("Expected:", refMean, 4, 2, 3);

        // Verify mean vector
        float eps = 420000 * std::numeric_limits<float>::epsilon ();  // 0.0500679
        for (uint i = 0; i < 8; ++i)
            ASSERT_LT (std::abs (refMean[i] - results[i]), eps);

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                ICP::cpuICPMeanWeighted (mean.hPtrInF, mean.hPtrInM, refMean, mean.hPtrInW, n);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = mean.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "ICPMean<WEIGHTED>");
        }

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}


/*! \brief Tests the **icpSubtractMean** kernel.
 *  \details The kernel computes the deviations of a set of points from their mean.
 */
TEST (ICP, icpSubtractMean)
{
    try
    {
        const unsigned int n = 1 << 14;  // 16384
        const unsigned int d = 8;
        // const unsigned int bufferInXSize = n * sizeof (cl_float8);
        // const unsigned int bufferInMSize = 2 * sizeof (cl_float4);
        // const unsigned int bufferOutSize = n * sizeof (cl_float4);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_icp);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::ICP::ICPDevs dev (clEnv, info);
        dev.init (n);

        // Initialize data (writes on staging buffer directly)
        std::generate (dev.hPtrInF, dev.hPtrInF + n * d, ICP::rNum_0_10000);
        std::generate (dev.hPtrInM, dev.hPtrInM + n * d, ICP::rNum_0_255);
        std::generate (dev.hPtrInMean, dev.hPtrInMean + 4, ICP::rNum_0_10000);
        std::generate (dev.hPtrInMean + 4, dev.hPtrInMean + 8, ICP::rNum_0_255);
        // ICP::printBufferF ("Original F:", dev.hPtrInF, d, n, 3);
        // ICP::printBufferF ("Original M:", dev.hPtrInM, d, n, 3);
        // ICP::printBufferF ("Original Mean:", dev.hPtrInMean, 4, 2, 3);

        // Copy data to device
        dev.write (cl_algo::ICP::ICPDevs::Memory::D_IN_F);
        dev.write (cl_algo::ICP::ICPDevs::Memory::D_IN_M);
        dev.write (cl_algo::ICP::ICPDevs::Memory::D_IN_MEAN);
        
        dev.run ();  // Execute kernels (13 us)
        
        // Copy results to host
        cl_float *devF = (cl_float *) dev.read (cl_algo::ICP::ICPDevs::Memory::H_OUT_DEV_F, CL_FALSE);
        cl_float *devM = (cl_float *) dev.read (cl_algo::ICP::ICPDevs::Memory::H_OUT_DEV_M);
        // ICP::printBufferF ("Received DF:", devF, 4, n, 3);
        // ICP::printBufferF ("Received DM:", devM, 4, n, 3);

        // Produce reference deviations array
        cl_float *refDevF = new cl_float[n * 4];
        cl_float *refDevM = new cl_float[n * 4];
        ICP::cpuICPDevs (dev.hPtrInF, dev.hPtrInM, refDevF, refDevM, dev.hPtrInMean, n);
        // ICP::printBufferF ("Expected DF:", refDevF, 4, n, 3);
        // ICP::printBufferF ("Expected DM:", refDevM, 4, n, 3);

        // Verify deviations
        float eps = 42 * std::numeric_limits<float>::epsilon ();  // 5.00679e-06
        for (uint i = 0; i < n; ++i)
        {
            ASSERT_LT (std::abs (refDevF[i] - devF[i]), eps);
            ASSERT_LT (std::abs (refDevM[i] - devM[i]), eps);
        }

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                ICP::cpuICPDevs (dev.hPtrInF, dev.hPtrInM, refDevF, refDevM, dev.hPtrInMean, n);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = dev.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "ICPDevs");
        }

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}


/*! \brief Tests the **icpSijProducts** kernel.
 *  \details The kernel produces the products in the Si elements of the S matrix.
 */
TEST (ICP, icpSijProducts)
{
    try
    {
        const std::vector<std::string> kernel_files = { kernel_filename_reduce, 
                                                        kernel_filename_icp };

        const unsigned int m = 1 << 14;  // 16384
        const unsigned int d = 4;
        const float c = 1e-6f;
        // const unsigned int bufferInSize = m * sizeof (cl_float4);
        // const unsigned int bufferOutSize = 11 * sizeof (cl_float);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_files);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        const cl_algo::ICP::ICPSConfig C = cl_algo::ICP::ICPSConfig::REGULAR;
        cl_algo::ICP::ICPS<C> s (clEnv, info);
        s.init (m, c);

        // Initialize data (writes on staging buffer directly)
        std::function<float ()> rNum_R__1000_1000 = std::bind (
            std::uniform_real_distribution<float> (-1000.f, 1000.f), 
            std::default_random_engine (std::chrono::system_clock::now ().time_since_epoch ().count ()));
        std::generate (s.hPtrInDevM, s.hPtrInDevM + m * d, rNum_R__1000_1000);
        rNum_R__1000_1000 = std::bind (
            std::uniform_real_distribution<float> (-1000.f, 1000.f), 
            std::default_random_engine (std::chrono::system_clock::now ().time_since_epoch ().count ()));
        std::generate (s.hPtrInDevF, s.hPtrInDevF + m * d, rNum_R__1000_1000);
        // ICP::printBufferF ("Original M:", s.hPtrInM, d, m, 3);
        // ICP::printBufferF ("Original F:", s.hPtrInF, d, m, 3);

        // Copy data to device
        s.write (cl_algo::ICP::ICPS<C>::Memory::D_IN_DEV_M);
        s.write (cl_algo::ICP::ICPS<C>::Memory::D_IN_DEV_F);
        
        s.run ();  // Execute kernels (~ 20 us)
        
        cl_float *results = (cl_float *) s.read ();  // Copy results to host
        // ICP::printBufferF ("Received:", results, 3, 4, 7);

        // Produce reference S matrix
        cl_float *refS = new cl_float[11];
        ICP::cpuICPS (s.hPtrInDevM, s.hPtrInDevF, refS, m, c);
        // ICP::printBufferF ("Expected:", refS, 3, 4, 7);

        // Verify S matrix
        float eps = 4200 * std::numeric_limits<float>::epsilon ();  // 0.000500679
        for (uint i = 0; i < 11; ++i)
            ASSERT_LT (std::abs (refS[i] - results[i]), eps);

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                ICP::cpuICPS (s.hPtrInDevM, s.hPtrInDevF, refS, m, c);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = s.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "ICPS<ICPSConfig::REGULAR>");
        }

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}


/*! \brief Tests the **icpSijProducts_Weighted** kernel.
 *  \details The kernel produces the weighted products in the Si elements of the S matrix.
 */
TEST (ICP, icpSijProducts_Weighted)
{
    try
    {
        const std::vector<std::string> kernel_files = { kernel_filename_reduce, 
                                                        kernel_filename_icp };

        const unsigned int m = 1 << 14;  // 16384
        const unsigned int d = 4;
        const float c = 1e-6f;
        // const unsigned int bufferInFMSize = m * sizeof (cl_float4);
        // const unsigned int bufferInWSize = m * sizeof (cl_float);
        // const unsigned int bufferOutSize = 11 * sizeof (cl_float);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_files);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        const cl_algo::ICP::ICPSConfig C = cl_algo::ICP::ICPSConfig::WEIGHTED;
        cl_algo::ICP::ICPS<C> s (clEnv, info);
        s.init (m, c);

        // Initialize data (writes on staging buffer directly)
        std::function<float ()> rNum_R__1000_1000 = std::bind (
            std::uniform_real_distribution<float> (-1000.f, 1000.f), 
            std::default_random_engine (std::chrono::system_clock::now ().time_since_epoch ().count ()));
        std::generate (s.hPtrInDevM, s.hPtrInDevM + m * d, rNum_R__1000_1000);
        rNum_R__1000_1000 = std::bind (
            std::uniform_real_distribution<float> (-1000.f, 1000.f), 
            std::default_random_engine (std::chrono::system_clock::now ().time_since_epoch ().count ()));
        std::generate (s.hPtrInDevF, s.hPtrInDevF + m * d, rNum_R__1000_1000);
        std::generate (s.hPtrInW, s.hPtrInW + m, ICP::rNum_R_0_1);
        // ICP::printBufferF ("Original M:", s.hPtrInM, d, m, 3);
        // ICP::printBufferF ("Original F:", s.hPtrInF, d, m, 3);
        // ICP::printBufferF ("Original W:", s.hPtrInW, 1, m, 3);

        // Copy data to device
        s.write (cl_algo::ICP::ICPS<C>::Memory::D_IN_DEV_M);
        s.write (cl_algo::ICP::ICPS<C>::Memory::D_IN_DEV_F);
        s.write (cl_algo::ICP::ICPS<C>::Memory::D_IN_W);
        
        s.run ();  // Execute kernels (~ 20 us)
        
        cl_float *results = (cl_float *) s.read ();  // Copy results to host
        // ICP::printBufferF ("Received:", results, 3, 4, 7);

        // Produce reference S matrix
        cl_float *refS = new cl_float[11];
        ICP::cpuICPSw (s.hPtrInDevM, s.hPtrInDevF, s.hPtrInW, refS, m, c);
        // ICP::printBufferF ("Expected:", refS, 3, 4, 7);

        // Verify S matrix
        float eps = 4200 * std::numeric_limits<float>::epsilon ();  // 0.000500679
        for (uint i = 0; i < 11; ++i)
            ASSERT_LT (std::abs (refS[i] - results[i]), eps);

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                ICP::cpuICPSw (s.hPtrInDevM, s.hPtrInDevF, s.hPtrInW, refS, m, c);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = s.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "ICPS<ICPSConfig::WEIGHTED>");
        }

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}


/*! \brief Tests the **icpTransform_Quaternion** kernel.
 *  \details The kernel transforms a set of points using a 
 *           unit quaternion and a translation vector.
 */
TEST (ICP, icpTransform_Quaternion)
{
    try
    {
        const unsigned int m = 1 << 14;  // 16384
        const unsigned int d = 8;
        // const unsigned int bufferInSize = m * sizeof (cl_float8);
        // const unsigned int bufferOutSize = m * sizeof (cl_float8);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_icp);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        const cl_algo::ICP::ICPTransformConfig C = cl_algo::ICP::ICPTransformConfig::QUATERNION;
        cl_algo::ICP::ICPTransform<C> transform (clEnv, info);
        transform.init (m);

        // Initialize data (writes on staging buffer directly)
        // Randomized set of points
        std::generate (transform.hPtrInM, transform.hPtrInM + m * d, ICP::rNum_0_255);
        // Some random unit quaternion
        transform.hPtrInT[0] = 0.5144; transform.hPtrInT[1] = 0.5743;
        transform.hPtrInT[2] = 0.5632; transform.hPtrInT[3] = 0.2973;
        // Randomized translation vector
        std::generate (transform.hPtrInT + 4, transform.hPtrInT + 7, ICP::rNum_0_255);
        // Some random scaling factor
        std::generate (transform.hPtrInT + 7, transform.hPtrInT + 8, ICP::rNum_R_0_1);
        // transform.hPtrInT[7] = 1;
        // ICP::printBufferF ("Original M:", transform.hPtrInM, d, m, 3);
        // ICP::printBufferF ("Original T:", transform.hPtrInT, 1, 8, 3);

        // Copy data to device
        transform.write (cl_algo::ICP::ICPTransform<C>::Memory::D_IN_M);
        transform.write (cl_algo::ICP::ICPTransform<C>::Memory::D_IN_T);
        
        transform.run ();  // Execute kernels (18 us)
        
        cl_float *results = (cl_float *) transform.read ();  // Copy results to host
        // ICP::printBufferF ("Received:", results, d, m, 3);

        // Produce reference transformed set
        cl_float *refTM = new cl_float[m * d];
        ICP::cpuICPTransformQ (transform.hPtrInM, refTM, transform.hPtrInT, m);
        // ICP::printBufferF ("Expected:", refTM, d, m, 3);

        // Verify transformed set
        float eps = 4200 * std::numeric_limits<float>::epsilon ();  // 0.000500679
        for (uint i = 0; i < m; ++i)
            for (uint k = 0; k < d; ++k)
                ASSERT_LT (std::abs (refTM[i * d + k] - results[i * d + k]), eps);

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                ICP::cpuICPTransformQ (transform.hPtrInM, refTM, transform.hPtrInT, m);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = transform.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "ICPTransform<QUATERNION>");
        }

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}


/*! \brief Tests the **icpTransform_Matrix** kernel.
 *  \details The kernel transforms a set of points using a transformation matrix.
 */
TEST (ICP, icpTransform_Matrix)
{
    try
    {
        const unsigned int m = 1 << 14;  // 16384
        const unsigned int d = 8;
        // const unsigned int bufferInSize = m * sizeof (cl_float8);
        // const unsigned int bufferOutSize = m * sizeof (cl_float8);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_icp);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        const cl_algo::ICP::ICPTransformConfig C = cl_algo::ICP::ICPTransformConfig::MATRIX;
        cl_algo::ICP::ICPTransform<C> transform (clEnv, info);
        transform.init (m);

        // Initialize data (writes on staging buffer directly)
        // Randomized set of points
        std::generate (transform.hPtrInM, transform.hPtrInM + m * d, ICP::rNum_0_255);
        // Some random scaling factor
        float s = ICP::rNum_R_0_1 ();
        // Some random rotation matrix (axis = (1/sqrt(3), 1/sqrt(3), 1/sqrt(3)), angle = 36.21deg)
        transform.hPtrInT[0] = s *  0.871238; transform.hPtrInT[1] = s * -0.276687;
        transform.hPtrInT[2] = s *  0.405449; transform.hPtrInT[3] = ICP::rNum_0_255 ();
        transform.hPtrInT[4] = s *  0.405449; transform.hPtrInT[5] = s *  0.871238;
        transform.hPtrInT[6] = s * -0.276687; transform.hPtrInT[7] = ICP::rNum_0_255 ();
        transform.hPtrInT[8] = s * -0.276687; transform.hPtrInT[9] = s *  0.405449;
        transform.hPtrInT[10] = s * 0.871238; transform.hPtrInT[11] = ICP::rNum_0_255 ();
        transform.hPtrInT[12] = 0.f; transform.hPtrInT[13] = 0.f;
        transform.hPtrInT[14] = 0.f; transform.hPtrInT[15] = 1.f;
        // ICP::printBufferF ("Original M:", transform.hPtrInM, d, m, 3);
        // ICP::printBufferF ("Original T:", transform.hPtrInT, 4, 4, 3);

        // Copy data to device
        transform.write (cl_algo::ICP::ICPTransform<C>::Memory::D_IN_M);
        transform.write (cl_algo::ICP::ICPTransform<C>::Memory::D_IN_T);
        
        transform.run ();  // Execute kernels (18 us)
        
        cl_float *results = (cl_float *) transform.read ();  // Copy results to host
        // ICP::printBufferF ("Received:", results, d, m, 3);

        // Produce reference transformed set
        cl_float *refTM = new cl_float[m * d];
        ICP::cpuICPTransformM (transform.hPtrInM, refTM, transform.hPtrInT, m);
        // ICP::printBufferF ("Expected:", refTM, d, m, 3);

        // Verify transformed set
        float eps = 42000 * std::numeric_limits<float>::epsilon ();  // 0.00500679
        for (uint i = 0; i < m; ++i)
            for (uint k = 0; k < d; ++k)
                ASSERT_LT (std::abs (refTM[i * d + k] - results[i * d + k]), eps);

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                ICP::cpuICPTransformM (transform.hPtrInM, refTM, transform.hPtrInT, m);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = transform.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "ICPTransform<MATRIX>");
        }

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}


/*! \brief Tests the **icpPowerMethod** kernel.
 *  \details The kernel uses the Power Method to estimate the 
 *           incremental development in the transformation estimation.
 */
TEST (ICP, icpPowerMethod)
{
    try
    {
        // const unsigned int bufferInSSize = 11 * sizeof (cl_float);
        // const unsigned int bufferOutMeanSize = 2 * sizeof (cl_float4);
        // const unsigned int bufferOutTkSize = 2 * sizeof (cl_float4);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_icp);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::ICP::ICPPowerMethod pm (clEnv, info);
        pm.init ();

        // Initialize data (writes on staging buffer directly)
        cl_float S[11] = 
        {
              0.00168053,   0.000131408, -0.000775179, 
              0.000156595,  0.00102674,  -0.000563479, 
             -0.000722137, -0.000559463,  0.00246661, 
              0.00521271,   0.00515292
        };
        cl_float means[8] = 
        {
            -33.9694f, -17.6421f, 1494.22f, 0.f, 
            -44.8322f, -19.3835f, 1485.93f, 0.f 
        };
        // ICP::printBufferF ("Original S:", S, 3, 4, 9);
        // ICP::printBufferF ("Original Mean:", means, 4, 2, 4);

        // Copy data to device
        pm.write (cl_algo::ICP::ICPPowerMethod::Memory::D_IN_S, S);
        pm.write (cl_algo::ICP::ICPPowerMethod::Memory::D_IN_MEAN, means);
        
        pm.run ();  // Execute kernel (27 us for 56 iterations)
        
        cl_float *results = (cl_float *) pm.read ();  // Copy results to host
        // ICP::printBufferF ("Received Tk:", results, 4, 2, 7);

        // Produce reference transformed set
        cl_float refTk[8];
        ICP::cpuICPPowerMethod (pm.hPtrInS, pm.hPtrInMean, refTk);
        // ICP::printBufferF ("Expected Tk:", refTk, 4, 2, 7);

        // Verify transformation against reference implementation
        float eps = 420 * std::numeric_limits<float>::epsilon ();  // 5.00679e-05
        for (uint k = 0; k < 8; ++k)
            ASSERT_LT (std::abs (refTk[k] - results[k]), eps);

        cl_float svdTk[8] = 
        {
            0.00111412f, 0.00730956f, -0.00647493f, 0.999952f, 
              -10.4598f,    4.74009f,   -0.762817f,  1.00578f
        };
        // ICP::printBufferF ("SVD Tk:", svdTk, 4, 2, 7);

        // Verify transformation against SVD solution
        eps = 42000 * std::numeric_limits<float>::epsilon ();  // 0.00500679
        for (uint k = 0; k < 8; ++k)
            ASSERT_LT (std::abs (svdTk[k] - results[k]), eps);

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                ICP::cpuICPPowerMethod (pm.hPtrInS, pm.hPtrInMean, refTk);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = pm.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "ICPPowerMethod");
        }

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}


int main (int argc, char **argv)
{
    profiling = ICP::setProfilingFlag (argc, argv);

    ::testing::InitGoogleTest (&argc, argv);

    return RUN_ALL_TESTS ();
}
