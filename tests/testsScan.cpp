/*! \file testsScan.cpp
 *  \brief Google Test Unit Tests for the `Scan` kernels.
 *  \note Use the `--profiling` flag to enable profiling of the kernels.
 *  \note The benchmarks in these tests are against naive CPU implementations 
 *        of the associated algorithms. They are used only for testing purposes, 
 *        and not for examining the performance of their GPU alternatives.
 *  \author Nick Lamprianidis
 *  \version 1.0
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
#include <ICP/algorithms.hpp>
#include <ICP/tests/helper_funcs.hpp>


// Kernel filenames
const std::string kernel_filename_scan { "kernels/ICP/scan_kernels.cl" };

// Uniform random number generators
namespace ICP
{
    extern std::function<unsigned char ()> rNum_0_255;
    extern std::function<unsigned short ()> rNum_0_10000;
    extern std::function<float ()> rNum_R_0_1;
    extern std::function<float ()> rNum_R_1_255_E__6;
}

bool profiling;  // Flag to enable profiling of the kernels (--profiling)


/*! \brief Tests the **inclusiveScan** kernel.
 *  \details The operation is an inclusive scan on the columns of an array.
 */
TEST (Scan, inclusiveScan)
{
    try
    {
        const unsigned int cols = 1024, rows = 1024;
        const unsigned int bufferSize = cols * rows * sizeof (cl_int);

        // Set up OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        cl::CommandQueue &queue (clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE));
        clEnv.addProgram (0, kernel_filename_scan);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        const cl_algo::ICP::ScanConfig C = cl_algo::ICP::ScanConfig::INCLUSIVE;
        cl_algo::ICP::Scan<C> scan (clEnv, info);
        scan.init (cols, rows);

        // Initialize data (writes on staging buffer directly)
        std::generate (scan.hPtrIn, scan.hPtrIn + bufferSize / sizeof (cl_int), ICP::rNum_0_255);
        // ICP::printBuffer ("Original:", scan.hPtrIn, cols, rows);

        scan.write ();  // Copy data to device

        scan.run ();  // Execute kernels (~ 151 us)
        
        // Check group sums
        // cl_int groupSums[4 * bufferSize / cols];
        // queue.enqueueReadBuffer ((cl::Buffer &) scan.get (cl_algo::ICP::Scan::Memory::D_SUMS), 
        //                          CL_TRUE, 0, 4 * bufferSize / cols, groupSums);
        // ICP::printBuffer ("\nGroup Sums:", groupSums, 4, rows);

        cl_int *results = (cl_int *) scan.read ();  // Copy results to host
        // ICP::printBuffer ("Received:", results, cols, rows);

        // Produce reference scan array
        cl_int *refScan = new cl_int[cols * rows];
        ICP::cpuInScan (scan.hPtrIn, refScan, cols, rows);
        // ICP::printBuffer ("Expected:", refScan, cols, rows);

        // Verify scan output
        for (uint row = 0; row < rows; ++row)
            for (uint col = 0; col < cols; ++col)
                ASSERT_EQ (refScan[row * cols + col], results[row * cols + col]);

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
                ICP::cpuInScan (scan.hPtrIn, refScan, cols, rows);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = scan.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "Scan<INCLUSIVE_INT>");
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


/*! \brief Tests the **exclusiveScan** kernel.
 *  \details The operation is an exclusive scan on the columns of an array.
 */
TEST (Scan, exclusiveScan)
{
    try
    {
        const unsigned int cols = 1024, rows = 1024;
        const unsigned int bufferSize = cols * rows * sizeof (cl_int);

        // Set up OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        cl::CommandQueue &queue (clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE));
        clEnv.addProgram (0, kernel_filename_scan);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        const cl_algo::ICP::ScanConfig C = cl_algo::ICP::ScanConfig::EXCLUSIVE;
        cl_algo::ICP::Scan<C> scan (clEnv, info);
        scan.init (cols, rows);

        // Initialize data (writes on staging buffer directly)
        std::generate (scan.hPtrIn, scan.hPtrIn + bufferSize / sizeof (cl_int), ICP::rNum_0_255);
        // ICP::printBuffer ("Original:", scan.hPtrIn, cols, rows);

        scan.write ();  // Copy data to device

        scan.run ();  // Execute kernels (~ 151 us)
        
        // Check group sums
        // cl_int groupSums[4 * bufferSize / cols];
        // queue.enqueueReadBuffer ((cl::Buffer &) scan.get (cl_algo::ICP::Scan::Memory::D_SUMS), 
        //                          CL_TRUE, 0, 4 * bufferSize / cols, groupSums);
        // ICP::printBuffer ("\nGroup Sums:", groupSums, 4, rows);

        cl_int *results = (cl_int *) scan.read ();  // Copy results to host
        // ICP::printBuffer ("Received:", results, cols, rows);

        // Produce reference scan array
        cl_int *refScan = new cl_int[cols * rows];
        ICP::cpuExScan (scan.hPtrIn, refScan, cols, rows);
        // ICP::printBuffer ("Expected:", refScan, cols, rows);

        // Verify scan output
        for (uint row = 0; row < rows; ++row)
            for (uint col = 0; col < cols; ++col)
                ASSERT_EQ (refScan[row * cols + col], results[row * cols + col]);

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
                ICP::cpuExScan (scan.hPtrIn, refScan, cols, rows);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = scan.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "Scan<EXCLUSIVE_INT>");
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
