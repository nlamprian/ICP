/*! \file algorithms.hpp
 *  \brief Declares classes that organize the execution of OpenCL kernels.
 *  \details Each class hides the details of kernel execution. They
 *           initialize the necessary buffers, set up the workspaces, 
 *           and run the kernels.
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

#ifndef ICP_ALGORITHMS_HPP
#define ICP_ALGORITHMS_HPP

#include <CLUtils.hpp>
#include <ICP/common.hpp>
#include <RBC/data_types.hpp>
#include <RBC/algorithms.hpp>
#include <eigen3/Eigen/Dense>


/*! \brief Offers classes which set up kernel execution parameters and 
 *         provide interfaces for the handling of memory objects.
 */
namespace cl_algo
{
/*! \brief Offers classes associated with the `%ICP` pipeline. */
namespace ICP
{

    /*! \brief Enumerates configurations for the `Reduce` class. */
    enum class ReduceConfig : uint8_t
    {
        MIN,  /*!< Identifies the case of `min` reduce. */
        MAX,  /*!< Identifies the case of `max` reduce. */
        SUM   /*!< Identifies the case of `sum` reduce. */
    };


    /*! \brief Interface class for the `reduce` kernels.
     *  \details The `reduce` kernels reduce each row of an array to a single element. 
     *           For more details, look at the kernels' documentation.
     *  \note The `reduce` kernels are available in `kernels/reduce_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `Reduce` instance:<br>
     *        |  Name  | Type | Placement | I/O | Use | Properties | Size |
     *        |  ---   |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN   | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$columns*rows*sizeof\ (T)\f$ |
     *        | H_OUT  | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$        rows*sizeof\ (T)\f$ |
     *        | D_IN   | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$columns*rows*sizeof\ (T)\f$ |
     *        | D_OUT  | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$        rows*sizeof\ (T)\f$ |
     *  
     *  \tparam C configures the class for different types of reduction.
     *  \tparam T configures the class to work with different types of data.
     */
    template <ReduceConfig C, typename T = cl_float>
    class Reduce
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN,   /*!< Input staging buffer. */
            H_OUT,  /*!< Output staging buffer. */
            D_IN,   /*!< Input buffer. */
            D_RED,  /*!< Buffer of reduced elements per work-group. */
            D_OUT   /*!< Output buffer. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        Reduce (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (Reduce::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _cols, unsigned int _rows, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (Reduce::Memory mem = Reduce::Memory::D_IN, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (Reduce::Memory mem = Reduce::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        T *hPtrIn;  /*!< Mapping of the input staging buffer. */
        T *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel recKernel, groupRecKernel;
        cl::NDRange globalR, globalGR, local;
        Staging staging;
        size_t wgMultiple, wgXdim;
        unsigned int cols, rows;
        unsigned int bufferInSize, bufferGRSize, bufferOutSize;
        cl::Buffer hBufferIn, hBufferOut;
        cl::Buffer dBufferIn, dBufferR, dBufferOut;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            double pTime;

            if (wgXdim == 1)
            {
                queue.enqueueNDRangeKernel (recKernel, cl::NullRange, globalR, local, events, &timer.event ());
                queue.flush (); timer.wait ();
                pTime = timer.duration ();
            }
            else
            {
                queue.enqueueNDRangeKernel (recKernel, cl::NullRange, globalR, local, events, &timer.event ());
                queue.flush (); timer.wait ();
                pTime = timer.duration ();

                queue.enqueueNDRangeKernel (groupRecKernel, cl::NullRange, globalGR, local, nullptr, &timer.event ());
                queue.flush (); timer.wait ();
                pTime += timer.duration ();
            }

            return pTime;
        }

    };


    /*! \brief Enumerates configurations for the `Scan` class. */
    enum class ScanConfig : uint8_t
    {
        INCLUSIVE,  /*!< Identifies the case of `inclusive` scan. */
        EXCLUSIVE   /*!< Identifies the case of `exclusive` scan. */
    };


    /*! \brief Interface class for the `scan` kernels.
     *  \details `scan` performs a scan operation on each row in an array. 
     *           For more details, look at the kernel's documentation.
     *  \note The `scan` kernel is available in `kernels/scan_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `Scan` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_int)\f$ |
     *        | H_OUT| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_int)\f$ |
     *        | D_IN | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$width*height*sizeof\ (cl\_int)\f$ |
     *        | D_OUT| Buffer | Device | O | Processing  | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_int)\f$ |
     *  
     *  \tparam C configures the class to perform either `inclusive` or `exclusive` scan.
     *  \tparam T configures the class to work with different types of data.
     */
    template <ScanConfig C, typename T = cl_int>
    class Scan
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN,    /*!< Input staging buffer. */
            H_OUT,   /*!< Output staging buffer. */
            D_IN,    /*!< Input buffer. */
            D_SUMS,  /*!< Output buffer of partial group sums. */
            D_OUT    /*!< Output buffer. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        Scan (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (Scan::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _cols, unsigned int _rows, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (Scan::Memory mem = Scan::Memory::D_IN, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (Scan::Memory mem = Scan::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        T *hPtrIn;  /*!< Mapping of the input staging buffer. */
        T *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernelScan, kernelSumsScan, kernelAddSums;
        cl::NDRange globalScan, globalSumsScan, localScan;
        cl::NDRange globalAddSums, localAddSums, offsetAddSums;
        Staging staging;
        size_t wgMultiple, wgXdim;
        unsigned int cols, rows, bufferSize, bufferSumsSize;
        cl::Buffer hBufferIn, hBufferOut;
        cl::Buffer dBufferIn, dBufferOut, dBufferSums;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            double pTime;

            if (wgXdim == 1)
            {
                queue.enqueueNDRangeKernel (
                    kernelScan, cl::NullRange, globalScan, localScan, events, &timer.event ());
                queue.flush (); timer.wait ();
                pTime = timer.duration ();
            }
            else
            {
                queue.enqueueNDRangeKernel (
                    kernelScan, cl::NullRange, globalScan, localScan, events, &timer.event ());
                queue.flush (); timer.wait ();
                pTime = timer.duration ();

                queue.enqueueNDRangeKernel (
                    kernelSumsScan, cl::NullRange, globalSumsScan, localScan, nullptr, &timer.event ());
                queue.flush (); timer.wait ();
                pTime += timer.duration ();

                queue.enqueueNDRangeKernel (
                    kernelAddSums, offsetAddSums, globalAddSums, localAddSums, nullptr, &timer.event ());
                queue.flush (); timer.wait ();
                pTime += timer.duration ();
            }

            return pTime;
        }

    };


    /*! \brief Interface class for the `getLMs` kernel.
     *  \details `getLMs` samples a point cloud for landmarks.
     *           For more details, look at the kernel's documentation.
     *  \note The `getLMs` kernel is available in `kernels/icp_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `ICPLMs` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN  | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$640*480*sizeof\ (cl\_float8)\f$ |
     *        | H_OUT | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$128*128*sizeof\ (cl\_float8)\f$ |
     *        | D_IN  | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$640*480*sizeof\ (cl\_float8)\f$ |
     *        | D_OUT | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$128*128*sizeof\ (cl\_float8)\f$ |
     */
    class ICPLMs
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN,   /*!< Input staging buffer. */
            H_OUT,  /*!< Output staging buffer. */
            D_IN,   /*!< Input buffer. */
            D_OUT   /*!< Output buffer. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        ICPLMs (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (ICPLMs::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (ICPLMs::Memory mem = ICPLMs::Memory::D_IN, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (ICPLMs::Memory mem = ICPLMs::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        cl_float *hPtrIn;  /*!< Mapping of the input staging buffer. */
        cl_float *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global;
        Staging staging;
        unsigned int n, m, d;
        unsigned int bufferInSize, bufferOutSize;
        cl::Buffer hBufferIn, hBufferOut, dBufferIn, dBufferOut;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, &timer.event ());
            queue.flush (); timer.wait ();
            
            return timer.duration ();
        }

    };


    /*! \brief Interface class for the `getReps` kernel.
     *  \details `getReps` samples a set of landmarks `(|LM|=16384)` for representatives.
     *           For more details, look at the kernel's documentation.
     *  \note The `getReps` kernel is available in `kernels/icp_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `ICPReps` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN  | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$16384*sizeof\ (cl\_float8)\f$ |
     *        | H_OUT | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$n_r  *sizeof\ (cl\_float8)\f$ |
     *        | D_IN  | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$16384*sizeof\ (cl\_float8)\f$ |
     *        | D_OUT | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$n_r  *sizeof\ (cl\_float8)\f$ |
     */
    class ICPReps
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN,   /*!< Input staging buffer. */
            H_OUT,  /*!< Output staging buffer. */
            D_IN,   /*!< Input buffer. */
            D_OUT   /*!< Output buffer. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        ICPReps (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (ICPReps::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _nr, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (ICPReps::Memory mem = ICPReps::Memory::D_IN, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (ICPReps::Memory mem = ICPReps::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        cl_float *hPtrIn;  /*!< Mapping of the input staging buffer. */
        cl_float *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global;
        Staging staging;
        unsigned int m, nr, nrx, nry, d;
        unsigned int bufferInSize, bufferOutSize;
        cl::Buffer hBufferIn, hBufferOut, dBufferIn, dBufferOut;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, &timer.event ());
            queue.flush (); timer.wait ();
            
            return timer.duration ();
        }

    };


    /*! \brief Interface class for the `icpComputeReduceWeights` kernel.
     *  \details The `icpComputeReduceWeights` kernel computes weights for pairs of points 
     *           in the fixed and moving sets, and also reduces them to get their sum. 
     *           For more details, look at the kernel's documentation.
     *  \note The `icpComputeReduceWeights` kernel is available in `kernels/icp_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by an `ICPWeights` instance:<br>
     *        |  Name  | Type | Placement | I/O | Use | Properties | Size |
     *        |  ---   |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN        | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n*sizeof\ (rbc\_dist\_id)\f$ |
     *        | H_OUT_W     | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$n*sizeof\ (cl\_float)    \f$ |
     *        | H_OUT_SUM_W | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$  sizeof\ (cl\_double)   \f$ |
     *        | D_IN        | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n*sizeof\ (rbc\_dist\_id)\f$ |
     *        | D_OUT_W     | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$n*sizeof\ (cl\_float)    \f$ |
     *        | D_OUT_SUM_W | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$  sizeof\ (cl\_double)   \f$ |
     */
    class ICPWeights
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN,         /*!< Input staging buffer for the distances between pairs 
                           *   of points in the fixed and moving sets. */
            H_OUT_W,      /*!< Output staging buffer for the weights. */
            H_OUT_SUM_W,  /*!< Output staging buffer for the sum of the weights. */
            D_IN,         /*!< Input buffer for the distances between pairs 
                           *   of points in the fixed and moving sets. */
            D_OUT_W,      /*!< Output buffer for the weights. */
            D_GW,         /*!< Buffer of block sums of weights. */
            D_OUT_SUM_W,  /*!< Output buffer for the sum of the weights. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        ICPWeights (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (ICPWeights::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _n, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (ICPWeights::Memory mem = ICPWeights::Memory::D_IN, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (ICPWeights::Memory mem = ICPWeights::Memory::H_OUT_SUM_W, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        rbc_dist_id *hPtrIn;   /*!< Mapping of the input staging buffer for the distances.       */
        cl_float *hPtrOutW;    /*!< Mapping of the output staging buffer for the weights.        */
        cl_double *hPtrOutSW;  /*!< Mapping of the output staging buffer for the sum of weights. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel weightKernel, groupWeightKernel;
        cl::NDRange globalW, globalGW, local;
        Staging staging;
        size_t wgMultiple, wgXdim;
        unsigned int n;
        unsigned int bufferInSize, bufferOutWSize, bufferGWSize, bufferOutSWSize;
        cl::Buffer hBufferIn, hBufferOutW, hBufferOutSW;
        cl::Buffer dBufferIn, dBufferOutW, dBufferOutSW;
        cl::Buffer dBufferGW;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            double pTime;

            if (wgXdim == 1)
            {
                queue.enqueueNDRangeKernel (weightKernel, cl::NullRange, globalW, local, events, &timer.event ());
                queue.flush (); timer.wait ();
                pTime = timer.duration ();
            }
            else
            {
                queue.enqueueNDRangeKernel (weightKernel, cl::NullRange, globalW, local, events, &timer.event ());
                queue.flush (); timer.wait ();
                pTime = timer.duration ();

                queue.enqueueNDRangeKernel (groupWeightKernel, cl::NullRange, globalGW, local, nullptr, &timer.event ());
                queue.flush (); timer.wait ();
                pTime += timer.duration ();
            }

            return pTime;
        }

    };


    /*! \brief Enumerates configurations for the `ICPMean` class. */
    enum class ICPMeanConfig : uint8_t
    { 
        REGULAR,  /*!< Identifies the case of regular mean calculation, 
                   *   \f$ \bar{x}_j = \sum^{n}_{i}{\frac{x_{ij}}{n}} \f$. */
        WEIGHTED  /*!< Identifies the case of weighted mean calculation, 
                   *   \f$ \bar{x}_j = \frac{\sum^{n}_{i}{w_i*x_{ij}}}{\sum^{n}_{i}{w_i}} \f$. */
    };


    /*! \brief Interface class for the calculation of the fixed and moving set means.
     *  \details Computes the mean on the xyz dimensions of the fixed and moving 
     *           sets of 8-D (4-D geometric and 4-D photometric information) points. 
     *           For more details, look at the kernels' documentation.
     *  \note The associated kernels are available in `kernels/icp_kernels.cl`.
     *  \note This is just a declaration. Look at the explicit template specializations
     *        for specific instantiations of the class.
     *        
     *  \tparam C configures the class for calculation of regular or weighted means.
     */
    template <ICPMeanConfig C>
    class ICPMean;


    /*! \brief Interface class for the `icpMean` kernel.
     *  \details The `icpMean` kernel computes the mean on the xyz dimensions 
     *           of the fixed and moving sets of 8-D (4-D geometric and 4-D 
     *           photometric information) points. 
     *           For more details, look at the kernel's documentation.
     *  \note The `icpMean` kernel is available in `kernels/icp_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created 
     *        by an `ICPMean<ICPMeanConfig::REGULAR>` instance:<br>
     *        |  Name  | Type | Placement | I/O | Use | Properties | Size |
     *        |  ---   |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_F | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n*sizeof\ (cl\_float8)\f$ |
     *        | H_IN_M | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n*sizeof\ (cl\_float8)\f$ |
     *        | H_OUT  | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$2*sizeof\ (cl\_float4)\f$ |
     *        | D_IN_F | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n*sizeof\ (cl\_float8)\f$ |
     *        | D_IN_M | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n*sizeof\ (cl\_float8)\f$ |
     *        | D_OUT  | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$2*sizeof\ (cl\_float4)\f$ |
     */
    template <>
    class ICPMean<ICPMeanConfig::REGULAR>
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN_F,  /*!< Input staging buffer for the fixed set. */
            H_IN_M,  /*!< Input staging buffer for the moving set. */
            H_OUT,   /*!< Output staging buffer. The first `cl_float4` contains the mean of the fixed set, 
                      *   and the second `cl_float` contains the mean of the moving set. */
            D_IN_F,  /*!< Input buffer for the fixed set. */
            D_IN_M,  /*!< Input buffer for the moving set. */
            D_GM,    /*!< Buffer of block means. */
            D_OUT    /*!< Output buffer. The first `cl_float4` contains the mean of the fixed set, 
                      *   and the second `cl_float` contains the mean of the moving set. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        ICPMean (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (ICPMean::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _n, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (ICPMean::Memory mem = ICPMean::Memory::D_IN_F, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (ICPMean::Memory mem = ICPMean::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        cl_float *hPtrInF;  /*!< Mapping of the input staging buffer for the fixed set. */
        cl_float *hPtrInM;  /*!< Mapping of the input staging buffer for the moving set. */
        cl_float *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel meanKernel, groupMeanKernel;
        cl::NDRange globalM, globalGM, local;
        Staging staging;
        size_t wgMultiple, wgXdim;
        unsigned int n, d;
        unsigned int bufferInSize, bufferGMSize, bufferOutSize;
        cl::Buffer hBufferInF, hBufferInM, hBufferOut;
        cl::Buffer dBufferInF, dBufferInM, dBufferGM, dBufferOut;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            double pTime;

            if (wgXdim == 1)
            {
                queue.enqueueNDRangeKernel (meanKernel, cl::NullRange, globalM, local, events, &timer.event ());
                queue.flush (); timer.wait ();
                pTime = timer.duration ();
            }
            else
            {
                queue.enqueueNDRangeKernel (meanKernel, cl::NullRange, globalM, local, events, &timer.event ());
                queue.flush (); timer.wait ();
                pTime = timer.duration ();

                queue.enqueueNDRangeKernel (groupMeanKernel, cl::NullRange, globalGM, local, nullptr, &timer.event ());
                queue.flush (); timer.wait ();
                pTime += timer.duration ();
            }

            return pTime;
        }

    };


    /*! \brief Interface class for the `icpMean_Weighted` kernel.
     *  \details The `icpMean_Weighted` kernel computes the weighted mean on the 
     *           xyz dimensions of the fixed and moving sets of 8-D (4-D geometric 
     *           and 4-D photometric information) points. 
     *           For more details, look at the kernel's documentation.
     *  \note The `icpMean_Weighted` kernel is available in `kernels/icp_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created 
     *        by an `ICPMean<ICPMeanConfig::WEIGHTED>` instance:<br>
     *        |  Name  | Type | Placement | I/O | Use | Properties | Size |
     *        |  ---   |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_F     | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n*sizeof\ (cl\_float8)\f$ |
     *        | H_IN_M     | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n*sizeof\ (cl\_float8)\f$ |
     *        | H_IN_W     | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n*sizeof\ (cl\_float) \f$ |
     *        | H_IN_SUM_W | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$  sizeof\ (cl\_double)\f$ |
     *        | H_OUT      | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$2*sizeof\ (cl\_float4)\f$ |
     *        | D_IN_F     | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n*sizeof\ (cl\_float8)\f$ |
     *        | D_IN_M     | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n*sizeof\ (cl\_float8)\f$ |
     *        | D_IN_W     | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n*sizeof\ (cl\_float) \f$ |
     *        | D_IN_SUM_W | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$  sizeof\ (cl\_double)\f$ |
     *        | D_OUT      | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$2*sizeof\ (cl\_float4)\f$ |
     */
    template <>
    class ICPMean<ICPMeanConfig::WEIGHTED>
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN_F,      /*!< Input staging buffer for the fixed set. */
            H_IN_M,      /*!< Input staging buffer for the moving set. */
            H_IN_W,      /*!< Input staging buffer for the weights. */
            H_IN_SUM_W,  /*!< Input staging buffer for the sum of weights. */
            H_OUT,       /*!< Output staging buffer. The first `cl_float4` contains the mean of the fixed set, 
                          *   and the second `cl_float` contains the mean of the moving set. */
            D_IN_F,      /*!< Input buffer for the fixed set. */
            D_IN_M,      /*!< Input buffer for the moving set. */
            D_IN_W,      /*!< Input buffer for the weights. */
            D_IN_SUM_W,  /*!< Input buffer for the sum of weights. */
            D_GM,        /*!< Buffer of block means. */
            D_OUT        /*!< Output buffer. The first `cl_float4` contains the mean of the fixed set, 
                          *   and the second `cl_float` contains the mean of the moving set. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        ICPMean (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (ICPMean::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _n, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (ICPMean::Memory mem = ICPMean::Memory::D_IN_F, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (ICPMean::Memory mem = ICPMean::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        cl_float *hPtrInF;    /*!< Mapping of the input staging buffer for the fixed set.      */
        cl_float *hPtrInM;    /*!< Mapping of the input staging buffer for the moving set.     */
        cl_float *hPtrInW;    /*!< Mapping of the input staging buffer for the weights.        */
        cl_double *hPtrInSW;  /*!< Mapping of the input staging buffer for the sum of weights. */
        cl_float *hPtrOut;    /*!< Mapping of the output staging buffer for the means.         */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel meanKernel, groupMeanKernel;
        cl::NDRange globalM, globalGM, local;
        Staging staging;
        size_t wgMultiple, wgXdim;
        unsigned int n, d;
        unsigned int bufferInFMSize, bufferInWSize, bufferInSWSize, bufferGMSize, bufferOutSize;
        cl::Buffer hBufferInF, hBufferInM, hBufferInW, hBufferInSW, hBufferOut;
        cl::Buffer dBufferInF, dBufferInM, dBufferInW, dBufferInSW, dBufferOut;
        cl::Buffer dBufferGM;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            double pTime;

            if (wgXdim == 1)
            {
                queue.enqueueNDRangeKernel (meanKernel, cl::NullRange, globalM, local, events, &timer.event ());
                queue.flush (); timer.wait ();
                pTime = timer.duration ();
            }
            else
            {
                queue.enqueueNDRangeKernel (meanKernel, cl::NullRange, globalM, local, events, &timer.event ());
                queue.flush (); timer.wait ();
                pTime = timer.duration ();

                queue.enqueueNDRangeKernel (groupMeanKernel, cl::NullRange, globalGM, local, nullptr, &timer.event ());
                queue.flush (); timer.wait ();
                pTime += timer.duration ();
            }

            return pTime;
        }

    };


    /*! \brief Interface class for the `icpSubtractMean` kernel.
     *  \details `icpSubtractMean` subtracts the mean of a set of points from 
     *           the points themselves to get their deviations from that mean.
     *           For more details, look at the kernel's documentation.
     *  \note The `icpSubtractMean` kernel is available in `kernels/icp_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `ICPDevs` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_F      | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n*sizeof\ (cl\_float8)\f$ |
     *        | H_IN_M      | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n*sizeof\ (cl\_float8)\f$ |
     *        | H_IN_MEAN   | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$2*sizeof\ (cl\_float4)\f$ |
     *        | H_OUT_DEV_F | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$n*sizeof\ (cl\_float4)\f$ |
     *        | H_OUT_DEV_M | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$n*sizeof\ (cl\_float4)\f$ |
     *        | D_IN_F      | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n*sizeof\ (cl\_float8)\f$ |
     *        | D_IN_M      | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n*sizeof\ (cl\_float8)\f$ |
     *        | D_IN_MEAN   | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$2*sizeof\ (cl\_float4)\f$ |
     *        | D_OUT_DEV_F | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$n*sizeof\ (cl\_float4)\f$ |
     *        | D_OUT_DEV_M | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$n*sizeof\ (cl\_float4)\f$ |
     */
    class ICPDevs
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN_F,       /*!< Input staging buffer for the fixed set. */
            H_IN_M,       /*!< Input staging buffer for the moving set. */
            H_IN_MEAN,    /*!< Input staging buffer for the fixed and moving set means. */
            H_OUT_DEV_F,  /*!< Output staging buffer for the deviations of the fixed set. */
            H_OUT_DEV_M,  /*!< Output staging buffer for the deviations of the moving set. */
            D_IN_F,       /*!< Input buffer for the fixed set. */
            D_IN_M,       /*!< Input buffer for the moving set. */
            D_IN_MEAN,    /*!< Input buffer for the fixed and moving set means. */
            D_OUT_DEV_F,  /*!< Output buffer for the deviations of the fixed set. */
            D_OUT_DEV_M   /*!< Output buffer for the deviations of the moving set. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        ICPDevs (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (ICPDevs::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _n, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (ICPDevs::Memory mem = ICPDevs::Memory::D_IN_F, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (ICPDevs::Memory mem = ICPDevs::Memory::H_OUT_DEV_F, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        cl_float *hPtrInF;      /*!< Mapping of the input staging buffer for the fixed set. */
        cl_float *hPtrInM;      /*!< Mapping of the input staging buffer for the moving set. */
        cl_float *hPtrInMean;   /*!< Mapping of the input staging buffer for the mean. */
        cl_float *hPtrOutDevF;  /*!< Mapping of the output staging buffer for the deviations of the fixed set. */
        cl_float *hPtrOutDevM;  /*!< Mapping of the output staging buffer for the deviations of the moving set. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global;
        Staging staging;
        unsigned int n, d;
        unsigned int bufferInFMSize, bufferInMeanSize, bufferOutSize;
        cl::Buffer hBufferInF, hBufferInM, hBufferInMean, hBufferOutDF, hBufferOutDM;
        cl::Buffer dBufferInF, dBufferInM, dBufferInMean, dBufferOutDF, dBufferOutDM;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, &timer.event ());
            queue.flush (); timer.wait ();
            
            return timer.duration ();
        }

    };


    /*! \brief Enumerates configurations for the `ICPS` class. */
    enum class ICPSConfig : uint8_t
    { 
        REGULAR,  /*!< Identifies the case of regular sums of products. */
        WEIGHTED  /*!< Identifies the case of weighted sums of products. */
    };


    /*! \brief Interface class for calculating the S matrix and the s scale factor constituents.
     *  \details The class uses the `icpSiProducts` kernels to produce the products 
     *           of deviations and then reduces them with the `reduce_sum` kernel.
     *           For more details, look at the kernels' documentation.
     *  \note The `icpSiProducts` kernels are available in `kernels/icp_kernels.cl`, and
     *        the `reduce_sum` kernel is available in `kernels/reduce_kernels.cl`.
     *  \note This is just a declaration. Look at the explicit template specializations
     *        for specific instantiations of the class.
     *        
     *  \tparam C configures the class for calculation of regular or weighted sums of products.
     */
    template <ICPSConfig C>
    class ICPS;


    /*! \brief Interface class for calculating the S matrix and the s scale factor constituents, 
     *         while considering regular residual errors.
     *  \details The class uses the `icpSiProducts` kernel to produce the products 
     *           of deviations and then reduces them with the `reduce_sum` kernel.
     *           For more details, look at the kernels' documentation.
     *  \note The `icpSiProducts` kernel is available in `kernels/icp_kernels.cl`, and
     *        the `reduce_sum` kernel is available in `kernels/reduce_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `ICPS` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_DEV_M | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$m*sizeof\ (cl\_float4)\f$ |
     *        | H_IN_DEV_F | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$m*sizeof\ (cl\_float4)\f$ |
     *        | H_OUT      | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$11*sizeof\ (cl\_float)\f$ |
     *        | D_IN_DEV_M | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$m*sizeof\ (cl\_float4)\f$ |
     *        | D_IN_DEV_F | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$m*sizeof\ (cl\_float4)\f$ |
     *        | D_OUT      | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$11*sizeof\ (cl\_float)\f$ |
     */
    template <>
    class ICPS<ICPSConfig::REGULAR>
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN_DEV_M,     /*!< Input staging buffer for the deviations of the moving set. */
            H_IN_DEV_F,     /*!< Input staging buffer for the deviations of the fixed set. */
            H_OUT,          /*!< Output staging buffer with the `S` matrix (first 9 floats) and 
                             *   the constituents of the scale factor `s` (last 2 floats). */
            D_IN_DEV_M,     /*!< Input buffer for the deviations of the moving set. */
            D_IN_DEV_F,     /*!< Input buffer for the deviations of the fixed set. */
            D_SIJ,          /*!< Buffer for the products of deviations. */
            D_OUT           /*!< Output buffer with the `S` matrix (first 9 floats) and 
                             *   the constituents of the scale factor `s` (last 2 floats). */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        ICPS (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (ICPS::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _m, float _c, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (ICPS::Memory mem = ICPS::Memory::D_IN_DEV_M, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (ICPS::Memory mem = ICPS::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Gets the scaling factor c. */
        float getScaling ();
        /*! \brief Sets the scaling factor c. */
        void setScaling (float _c);

        cl_float *hPtrInDevM;  /*!< Mapping of the input staging buffer for the deviations of the moving set. */
        cl_float *hPtrInDevF;  /*!< Mapping of the input staging buffer for the deviations of the fixed set. */
        cl_float *hPtrOut;  /*!< Mapping of the output staging buffer with the S matrix and scale factor s. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global;
        Reduce<ReduceConfig::SUM, cl_float> reduceSij;
        Staging staging;
        float c;
        unsigned int m, d;
        unsigned int bufferInSize, bufferSijSize, bufferOutSize;
        cl::Buffer hBufferInDM, hBufferInDF, hBufferSij, hBufferOut;
        cl::Buffer dBufferInDM, dBufferInDF, dBufferSij, dBufferOut;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            double pTime;

            queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, &timer.event ());
            queue.flush (); timer.wait ();
            pTime = timer.duration ();
            
            pTime += reduceSij.run (timer);

            return pTime;
        }

    };


    /*! \brief Interface class for calculating the S matrix and the s scale factor constituents, 
     *         while considering weighted residual errors.
     *  \details The class uses the `icpSiProducts` kernel to produce the products 
     *           of deviations and then reduces them with the `reduce_sum` kernel.
     *           For more details, look at the kernels' documentation.
     *  \note The `icpSiProducts` kernel is available in `kernels/icp_kernels.cl`, and
     *        the `reduce_sum` kernel is available in `kernels/reduce_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `ICPS` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_DEV_M | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$m*sizeof\ (cl\_float4)\f$ |
     *        | H_IN_DEV_F | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$m*sizeof\ (cl\_float4)\f$ |
     *        | H_IN_W     | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$m*sizeof\ (cl\_float) \f$ |
     *        | H_OUT      | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$11*sizeof\ (cl\_float)\f$ |
     *        | D_IN_DEV_M | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$m*sizeof\ (cl\_float4)\f$ |
     *        | D_IN_DEV_F | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$m*sizeof\ (cl\_float4)\f$ |
     *        | D_IN_W     | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$m*sizeof\ (cl\_float) \f$ |
     *        | D_OUT      | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$11*sizeof\ (cl\_float)\f$ |
     */
    template <>
    class ICPS<ICPSConfig::WEIGHTED>
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN_DEV_M,     /*!< Input staging buffer for the deviations of the moving set. */
            H_IN_DEV_F,     /*!< Input staging buffer for the deviations of the fixed set. */
            H_IN_W,         /*!< Input staging buffer for the weights. */
            H_OUT,          /*!< Output staging buffer with the `S` matrix (first 9 floats) and 
                             *   the constituents of the scale factor `s` (last 2 floats). */
            D_IN_DEV_M,     /*!< Input buffer for the deviations of the moving set. */
            D_IN_DEV_F,     /*!< Input buffer for the deviations of the fixed set. */
            D_IN_W,         /*!< Input buffer for the weights. */
            D_SIJ,          /*!< Buffer for the products of deviations. */
            D_OUT           /*!< Output buffer with the `S` matrix (first 9 floats) and 
                             *   the constituents of the scale factor `s` (last 2 floats). */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        ICPS (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (ICPS::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _m, float _c, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (ICPS::Memory mem = ICPS::Memory::D_IN_DEV_M, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (ICPS::Memory mem = ICPS::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Gets the scaling factor c. */
        float getScaling ();
        /*! \brief Sets the scaling factor c. */
        void setScaling (float _c);

        cl_float *hPtrInDevM;  /*!< Mapping of the input staging buffer for the deviations of the moving set. */
        cl_float *hPtrInDevF;  /*!< Mapping of the input staging buffer for the deviations of the fixed set. */
        cl_float *hPtrInW;  /*!< Mapping of the input staging buffer for the weights. */
        cl_float *hPtrOut;  /*!< Mapping of the output staging buffer with the S matrix and scale factor s. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global;
        Reduce<ReduceConfig::SUM, cl_float> reduceSij;
        Staging staging;
        float c;
        unsigned int m, d;
        unsigned int bufferInFMSize, bufferInWSize, bufferSijSize, bufferOutSize;
        cl::Buffer hBufferInDM, hBufferInDF, hBufferInW, hBufferSij, hBufferOut;
        cl::Buffer dBufferInDM, dBufferInDF, dBufferInW, dBufferSij, dBufferOut;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            double pTime;

            queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, &timer.event ());
            queue.flush (); timer.wait ();
            pTime = timer.duration ();
            
            pTime += reduceSij.run (timer);

            return pTime;
        }

    };


    /*! \brief Enumerates configurations for the `ICPTransform` class. */
    enum class ICPTransformConfig : uint8_t
    { 
        QUATERNION, /*!< Identifies the case where the homogeneous transformation 
                     *   is characterized by a quaternion and a translation vector, 
                     *   \f$\ p' = s\dot{q}\dot{p}\dot{q}^*+t \f$. */
        MATRIX  /*!< Identifies the case where the homogeneous transformation 
                 *   is characterized by a transformation matrix, 
                 *   \f$ p' = Tp = \left[ \begin{matrix} sR & t \\ 0 & 1 \end{matrix} \right]
                 *   \left[ \begin{matrix} p \\ 1 \end{matrix} \right] = sRp+t \f$. */
    };


    /*! \brief Interface class for the `icpTransform` kernels.
     *  \details The `icpTransform` kenrels perform a homogeneous transformation on a set of points. 
     *           For more details, look at the kernel's documentation.
     *  \note The `icpTransform` kernels are available in `kernels/icp_kernels.cl`.
     *  \note This is just a declaration. Look at the explicit template specializations
     *        for specific instantiations of the class.
     *        
     *  \tparam C configures the class for different types of transformation parameters.
     */
    template <ICPTransformConfig C>
    class ICPTransform;


    /*! \brief Interface class for the `icpTransform_Quaternion` kernel.
     *  \details The `icpTransform_Quaternion` kernel performs a homogeneous 
     *           transformation on a set of points using a quaternion and a 
     *           translation vector. 
     *           For more details, look at the kernels' documentation.
     *  \note The `icpTransform_Quaternion` kernel is available in `kernels/icp_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a 
     *        `ICPTransform<ICPTransformConfig::QUATERNION>` instance:<br>
     *        |  Name  | Type | Placement | I/O | Use | Properties | Size |
     *        |  ---   |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_M | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | H_IN_T | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$2*sizeof\ (cl\_float4)\f$ |
     *        | H_OUT  | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | D_IN_M | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | D_IN_T | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$2*sizeof\ (cl\_float4)\f$ |
     *        | D_OUT  | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$m*sizeof\ (cl\_float8)\f$ |
     */
    template <>
    class ICPTransform<ICPTransformConfig::QUATERNION>
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        { 
            H_IN_M,  /*!< Input staging buffer for the set of points. */
            H_IN_T,  /*!< Input staging buffer for the quaternion and the translation vector. 
                      *   The first `cl_float4` element contains the quaternion, \f$ \dot{q} = 
                      *   \left[ \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$, 
                      *   and the second `cl_float4` element contains the translation vector, 
                      *   \f$ t = \left[ \begin{matrix} t_x & t_y & t_z & 1 \end{matrix} \right]^T \f$. 
                      *   If scaling is desired, the factor should be placed in the last element 
                      *   of the translation vector, \f$ t = \left[ \begin{matrix} t_x & t_y 
                      *   & t_z & s \end{matrix} \right]^T \f$. */
            H_OUT,   /*!< Output staging buffer for the transformed set of points. */
            D_IN_M,  /*!< Input buffer for the set of points. */
            D_IN_T,  /*!< Input buffer for the quaternion and the translation vector. 
                      *   The first `cl_float4` element contains the quaternion, \f$ \dot{q} = 
                      *   \left[ \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$, 
                      *   and the second `cl_float4` element contains the translation vector, 
                      *   \f$ t = \left[ \begin{matrix} t_x & t_y & t_z & 1 \end{matrix} \right]^T \f$. 
                      *   If scaling is desired, the factor should be placed in the last element 
                      *   of the translation vector, \f$ t = \left[ \begin{matrix} t_x & t_y 
                      *   & t_z & s \end{matrix} \right]^T \f$. */
            D_OUT    /*!< Output buffer for the transformed set of points. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        ICPTransform (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (ICPTransform::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _m, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (ICPTransform::Memory mem = ICPTransform::Memory::D_IN_M, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (ICPTransform::Memory mem = ICPTransform::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        cl_float *hPtrInM;  /*!< Mapping of the input staging buffer for the set of points. */
        cl_float *hPtrInT;  /*!< Mapping of the input staging buffer for the quaternion and the translation vector. */
        cl_float *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global;
        Staging staging;
        unsigned int m, d;
        unsigned int bufferInMSize, bufferInTSize, bufferOutSize;
        cl::Buffer hBufferInM, hBufferInT, hBufferOut;
        cl::Buffer dBufferInM, dBufferInT, dBufferOut;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, &timer.event ());
            queue.flush (); timer.wait ();

            return timer.duration ();
        }

    };


    /*! \brief Interface class for the `icpTransform_Matrix` kernel.
     *  \details The `icpTransform_Matrix` kernel performs a homogeneous 
     *           transformation on a set of points using a transformation matrix. 
     *           For more details, look at the kernel's documentation.
     *  \note The `icpTransform_Matrix` kernel is available in `kernels/icp_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a 
     *        `ICPTransform<ICPTransformConfig::MATRIX>` instance:<br>
     *        |  Name  | Type | Placement | I/O | Use | Properties | Size |
     *        |  ---   |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_M | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | H_IN_T | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$4*sizeof\ (cl\_float4)\f$ |
     *        | H_OUT  | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | D_IN_M | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | D_IN_T | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$4*sizeof\ (cl\_float4)\f$ |
     *        | D_OUT  | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$m*sizeof\ (cl\_float8)\f$ |
     */
    template <>
    class ICPTransform<ICPTransformConfig::MATRIX>
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN_M,  /*!< Input staging buffer for the set of points. */
            H_IN_T,  /*!< Input staging buffer for the transformation matrix, 
                      *   \f$ T = \left[ \begin{matrix} R' & t \\ 0 & 1 \end{matrix} \right] = 
                      *   \left[ \begin{matrix} sR & t \\ 0 & 1 \end{matrix} \right] \f$. 
                      *   The elements should be laid out in row major order.
                      *   If scaling is desired, the factor should already be incorporated 
                      *   in the rotation matrix. */
            H_OUT,   /*!< Output staging buffer for the transformed set of points. */
            D_IN_M,  /*!< Input buffer for the set of points. */
            D_IN_T,  /*!< Input buffer for the transformation matrix, 
                      *   \f$ T = \left[ \begin{matrix} R' & t \\ 0 & 1 \end{matrix} \right] = 
                      *   \left[ \begin{matrix} sR & t \\ 0 & 1 \end{matrix} \right] \f$. 
                      *   The elements should be laid out in row major order.
                      *   If scaling is desired, the factor should already be incorporated 
                      *   in the rotation matrix. */
            D_OUT    /*!< Output buffer for the transformed set of points. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        ICPTransform (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (ICPTransform::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _m, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (ICPTransform::Memory mem = ICPTransform::Memory::D_IN_M, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (ICPTransform::Memory mem = ICPTransform::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        cl_float *hPtrInM;  /*!< Mapping of the input staging buffer for the set of points. */
        cl_float *hPtrInT;  /*!< Mapping of the input staging buffer for the transformation matrix. */
        cl_float *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global;
        Staging staging;
        unsigned int m, d;
        unsigned int bufferInMSize, bufferInTSize, bufferOutSize;
        cl::Buffer hBufferInM, hBufferInT, hBufferOut;
        cl::Buffer dBufferInM, dBufferInT, dBufferOut;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, &timer.event ());
            queue.flush (); timer.wait ();

            return timer.duration ();
        }

    };


    /*! \brief Interface class for the `icpTransform_Matrix` kernel.
     *  \details The `icpTransform_Matrix` kernel performs a homogeneous 
     *           transformation on a set of points using a transformation matrix. 
     *           For more details, look at the kernel's documentation.
     *  \note The `icpTransform_Matrix` kernel is available in `kernels/icp_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a 
     *        `ICPTransform<ICPTransformConfig::MATRIX>` instance:<br>
     *        |  Name  | Type | Placement | I/O | Use | Properties | Size |
     *        |  ---   |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_S    | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$11*sizeof\ (cl\_float)\f$ |
     *        | H_IN_MEAN | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$2*sizeof\ (cl\_float4)\f$ |
     *        | H_OUT_T_K | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$2*sizeof\ (cl\_float4)\f$ |
     *        | D_IN_S    | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$11*sizeof\ (cl\_float)\f$ |
     *        | D_IN_MEAN | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$2*sizeof\ (cl\_float4)\f$ |
     *        | D_OUT_T_K | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$2*sizeof\ (cl\_float4)\f$ |
     */
    class ICPPowerMethod
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN_S,     /*!< Input staging buffer for the sums of products. The first `9 cl_float` 
                         *   elements (in row major order) are the \f$S_k\f$ matrix, and the next 
                         *   `2 cl_float` are the numerator and denominator of the scale \f$s_k\f$. */
            H_IN_MEAN,  /*!< Input staging buffer for the set means. The first `cl_float4` is 
                         *   the fixed set mean, and the second one is the moving set mean. */
            H_OUT_T_K,  /*!< Output staging buffer for the parameters that represent the incremental 
                         *   development in the transformation estimation. The first `float4` 
                         *   is the **unit quaternion** \f$ \dot{q_k} = q_w + q_x i + q_y j + q_z k = 
                         *   \left[ \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$, 
                         *   and the second one is the **translation vector** \f$ t_k=\left[ \begin{matrix} 
                         *   t_x & t_y & t_z & 1 \end{matrix} \right]^T \f$. The scale is placed 
                         *   in the last element of the translation vector. That is, \f$ t_k = 
                         *   \left[ \begin{matrix} t_x & t_y & t_z & s_k \end{matrix} \right]^T \f$. */
            D_IN_S,     /*!< Input buffer for the sums of products. The first `9 cl_float` elements 
                         *   (in row major order) are the \f$S_k\f$ matrix, and the next `2 cl_float` 
                         *   are the numerator and denominator of the scale \f$s_k\f$. */
            D_IN_MEAN,  /*!< Input buffer for the set means. The first `cl_float4` is the 
                         *   fixed set mean, and the second one is the moving set mean. */
            D_OUT_T_K   /*!< Output buffer for the parameters that represent the incremental 
                         *   development in the transformation estimation. The first `float4` 
                         *   is the **unit quaternion** \f$ \dot{q_k} = q_w + q_x i + q_y j + q_z k = 
                         *   \left[ \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$, 
                         *   and the second one is the **translation vector** \f$ t_k=\left[ \begin{matrix} 
                         *   t_x & t_y & t_z & 1 \end{matrix} \right]^T \f$. The scale is placed 
                         *   in the last element of the translation vector. That is, \f$ t_k = 
                         *   \left[ \begin{matrix} t_x & t_y & t_z & s_k \end{matrix} \right]^T \f$. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        ICPPowerMethod (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (ICPPowerMethod::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (ICPPowerMethod::Memory mem = ICPPowerMethod::Memory::D_IN_S, 
                    void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (ICPPowerMethod::Memory mem = ICPPowerMethod::Memory::H_OUT_T_K, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        cl_float *hPtrInS;     /*!< Mapping of the input staging buffer for the sums of products. */
        cl_float *hPtrInMean;  /*!< Mapping of the input staging buffer for the fixed and moving set means. */
        cl_float *hPtrOutTk;   /*!< Mapping of the output staging buffer for the incremental parameters. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global;
        Staging staging;
        unsigned int bufferInSSize, bufferInMeanSize, bufferOutTkSize;
        cl::Buffer hBufferInS, hBufferInMean, hBufferOutTk;
        cl::Buffer dBufferInS, dBufferInMean, dBufferOutTk;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            queue.enqueueTask (kernel, events, &timer.event ());
            queue.flush (); timer.wait ();

            return timer.duration ();
        }

    };


    /*! \brief Enumerates configurations for the `ICPStep` class.
     *  \details All computation is done on the `GPU`. The only point of divergence 
     *           in the `%ICP` data flow is the rotation computation.
     */
    enum class ICPStepConfigT : uint8_t
    { 
        EIGEN, /*!< [**CPU**] Identifies the case where the **JacobiSVD** in **Eigen3** 
                *   is used to compute the singular value decomposition of matrix \f$S\f$. 
                *   Given \f$ S=USV^T \f$, the rotation matrix is then built as \f$R=VU^T\f$. */
        POWER_METHOD,  /*!< [**GPU**] Identifies the case where the **Power Method** is used to 
                        *   find the unit quaternion \f$\dot{q}\f$ that describes the rotation.
                        *   It computes the eigenvector \f$\mathcal{v}=\dot{q}\f$ that corresponds 
                        *   to the maximum eigenvalue of matrix \f$N\f$. */
        JACOBI  /*!< \todo [**GPU**] Identifies the case where one of the **Jacobi methods** 
                 *         is used to compute the unit quaternion \f$\dot{q}\f$ that describes 
                 *         the rotation. */
    };


    /*! \brief Enumerates configurations for the `ICPStep` class. */
    enum class ICPStepConfigW : uint8_t
    { 
        REGULAR,  /*!< Identifies the case of regular sum of errors. */
        WEIGHTED  /*!< Identifies the case of weighted sum of errors. */
    };


    /*! \brief Interface class for the `%ICP` pipeline.
     *  \details Performs one `%ICP` iteration. It accepts two sets (fixed and moving) of 
     *           landmarks, estimates the relative homogeneous transformation \f$ T_k \f$ 
     *           between them, and transforms the moving set according to this transformation.
     *  \note The implemented algorithm is described by `Horn` in two of his papers, 
     *        [Closed-Form Solution of Absolute Orientation Using Unit Quaternions][1] and 
     *        [Closed-Form Solution of Absolute Orientation Using Orthonormal Matrices][2].
     *        [1]: http://people.csail.mit.edu/bkph/papers/Absolute_Orientation.pdf
     *        [2]: http://graphics.stanford.edu/~smr/ICP/comparison/horn-hilden-orientation-josa88.pdf
     *  \note This is just a declaration. Look at the explicit template specializations
     *        for specific instantiations of the class.
     *        
     *  \tparam C configures the class for different methods of rotation computation.
     */
    template <ICPStepConfigT CR, ICPStepConfigW CW>
    class ICPStep;


    /*! \brief Interface class for the `%ICP` pipeline using the Eigen library 
     *         to estimate the rotation and considering regular residual errors.
     *  \details Performs one `%ICP` iteration. It accepts two sets (fixed and moving) 
     *           of landmarks, estimates the relative homogeneous transformation 
     *           \f$ (\dot{q}_k,t_k) \f$ between them, and transforms the moving set 
     *           according to this transformation.
     *  \note All computation is done on the `GPU`, except from the rotation 
     *        computation which is solved on the `CPU` with `Eigen::JacobiSVD`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created 
     *        by an `ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::REGULAR>` instance:<br>
     *        |  Name  | Type | Placement | I/O | Use | Properties | Size |
     *        |  ---   |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_F | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | H_IN_M | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | H_IO_T | Buffer | Host   | IO| Staging     | CL_MEM_READ_WRITE | \f$2*sizeof\ (cl\_float4)\f$ |
     *        | D_IN_F | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | D_IN_M | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | D_IO_T | Buffer | Device | IO| Processing  | CL_MEM_READ_WRITE | \f$2*sizeof\ (cl\_float4)\f$ |
     */
    template <>
    class ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::REGULAR>
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN_F,    /*!< Input staging buffer for the fixed set of landmarks. */
            H_IN_M,    /*!< Input staging buffer for the moving set of landmarks. */
            H_IO_T,    /*!< Input-output staging buffer for the quaternion and the translation vector. 
                        *   It is loaded with an initial estimation of the transformation 
                        *   and gets refined with every %ICP iteration.
                        *   The first `cl_float4` element contains the quaternion, \f$ \dot{q} = 
                        *   \left[ \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$, 
                        *   and the second `cl_float4` element contains the translation vector, 
                        *   \f$ t = \left[ \begin{matrix} t_x & t_y & t_z & 1 \end{matrix} \right]^T \f$. 
                        *   If scaling is desired, the factor should be placed in the last element 
                        *   of the translation vector, \f$ t = \left[ \begin{matrix} t_x & t_y 
                        *   & t_z & s \end{matrix} \right]^T \f$. */
            D_IN_F,    /*!< Input buffer for the fixed set of landmarks. */
            D_IN_M,    /*!< Input buffer for the moving set of landmarks. */
            D_IO_T,    /*!< Input-output buffer for the quaternion and the translation vector. 
                        *   It is loaded with an initial estimation of the transformation 
                        *   and gets refined with every %ICP iteration.
                        *   The first `cl_float4` element contains the quaternion, \f$ \dot{q} = 
                        *   \left[ \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$, 
                        *   and the second `cl_float4` element contains the translation vector, 
                        *   \f$ t = \left[ \begin{matrix} t_x & t_y & t_z & 1 \end{matrix} \right]^T \f$. 
                        *   If scaling is desired, the factor should be placed in the last element 
                        *   of the translation vector, \f$ t = \left[ \begin{matrix} t_x & t_y 
                        *   & t_z & s \end{matrix} \right]^T \f$. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        ICPStep (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _infoRBC, clutils::CLEnvInfo<1> _infoICP);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (ICPStep::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _m, unsigned int _nr, 
            float _a = 1e2f, float _c = 1e-6f, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (ICPStep::Memory mem = ICPStep::Memory::D_IN_F, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (ICPStep::Memory mem = ICPStep::Memory::H_IO_T, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr, bool config = false);
        /*! \brief Gets the scaling parameter \f$ \alpha \f$ involved in 
         *         the distance calculations of the `RBC` data structure. */
        float getAlpha ();
        /*! \brief Sets the scaling parameter \f$ \alpha \f$ involved in 
         *         the distance calculations of the `RBC` data structure. */
        void setAlpha (float _a);
        /*! \brief Gets the scaling factor c used when computing the `S` matrix. */
        float getScaling ();
        /*! \brief Sets the scaling factor c used when computing the `S` matrix. */
        void setScaling (float _c);

        cl_float *hPtrInF;  /*!< Mapping of the input staging buffer for the fixed set of points. */
        cl_float *hPtrInM;  /*!< Mapping of the input staging buffer for the moving set of points. */
        cl_float *hPtrIOT;  /*!< Mapping of the input-output staging buffer for the estimated 
                             *   quaternion and translation vector. */
        
        Eigen::Matrix3f Rk;     /*!< Represents the incremental development in the rotation estimation in 
                                 *   iteration `k`, given in rotation matrix representation, \f$ R_k \f$. */
        Eigen::Quaternionf qk;  /*!< Represents the incremental development in the rotation estimation in 
                                 *   iteration `k`, given in quaternion representation, \f$ \dot{q}_k = \left[ 
                                 *   \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$. */
        Eigen::Vector3f tk;     /*!< Represents the incremental development in the translation estimation 
                                 *   in iteration `k`, given as a vector in 3-D, \f$ t_k \f$. */
        cl_float sk;            /*!< Represents the incremental development in the scale estimation 
                                 *   in iteration `k`, given as a scalar, \f$ s_k \f$. */

        Eigen::Matrix3f R;     /*!< Represents the rotation estimation up to iteration `k`, 
                                *   given in rotation matrix representation, \f$ R \f$. */
        Eigen::Quaternionf q;  /*!< Represents the rotation estimation up to iteration `k`, 
                                *   given in quaternion representation, \f$ \dot{q} = \left[ 
                                *   \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$. */
        Eigen::Vector3f t;     /*!< Represents the translation estimation up to iteration `k`, 
                                *   given as a vector in 3-D, \f$ t \f$. */
        cl_float s;            /*!< Represents the scale estimation up to iteration `k`,
                                *   given as a scalar, \f$ s \f$. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> infoRBC, infoICP;
        cl::Context context;
        cl::CommandQueue queue;
        Staging staging;
        ICPReps fReps;
        RBC::RBCConstruct
            <RBC::KernelTypeC::KINECT_R, RBC::RBCPermuteConfig::GENERIC> rbcC;
        ICPTransform<ICPTransformConfig::QUATERNION> transform;
        RBC::RBCSearch
            <RBC::KernelTypeC::KINECT_R, 
                RBC::RBCPermuteConfig::GENERIC, RBC::KernelTypeS::KINECT> rbcS;
        ICPMean<ICPMeanConfig::REGULAR> means;
        ICPDevs devs;
        ICPS<ICPSConfig::REGULAR> matrixS;

        cl_float *mean, *Sij;
        Eigen::Vector3f mf, mm;
        Eigen::Matrix3f S;

        float a, c;
        unsigned int m, nr, d;
        unsigned int bufferFMSize, bufferTSize;
        cl::Buffer hBufferInF, hBufferInM, hBufferIOT;
        cl::Buffer dBufferInF, dBufferInM, dBufferIOT;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \param[in] config flag. If true, configures the `RBC search` process. 
         *                    Set to true once, when the `RBC` data structure is reset.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, 
            const std::vector<cl::Event> *events = nullptr, bool config = false)
        {
            clutils::CPUTimer<double, std::milli> cTimer;
            double pTime = 0.0;

            if (config)
            {
                fReps.run (events);
                rbcC.run ();
            }

            pTime += transform.run (timer, events);
            pTime += rbcS.run (timer, nullptr, config);
            pTime += means.run (timer);
            pTime += devs.run (timer);
            pTime += matrixS.run (timer);

            cTimer.start ();

            mean = (cl_float *) means.read (ICPMean<ICPMeanConfig::REGULAR>::Memory::H_OUT, CL_FALSE);
            Sij = (cl_float *) matrixS.read (ICPS<ICPSConfig::REGULAR>::Memory::H_OUT);
            sk = std::sqrt (Sij[9] / Sij[10]);

            mf = Eigen::Map<Eigen::Vector3f> (mean);
            mm = Eigen::Map<Eigen::Vector3f> (mean + 4);
            S = Eigen::Map<Eigen::Matrix3f, Eigen::Unaligned, Eigen::Stride<1, 3> > (Sij);
            
            Eigen::JacobiSVD<Eigen::MatrixXf, Eigen::NoQRPreconditioner> 
                svd (S, Eigen::ComputeThinU | Eigen::ComputeThinV);

            Rk = svd.matrixV () * svd.matrixU ().transpose ();
            if (Rk.determinant () < 0)
            {
                Eigen::Matrix3f B = Eigen::Matrix3f::Identity ();
                B (2, 2) = Rk.determinant ();
                Rk = svd.matrixV () * B * svd.matrixU ().transpose ();
            }
            qk = Eigen::Quaternionf (Rk);

            tk = mf - sk * Rk * mm;

            R = Rk * R;
            q = Eigen::Quaternionf (R);
            t = sk * Rk * t + tk;
            s = sk * s;
            
            Eigen::Map<Eigen::Vector4f> (hPtrIOT, 4) = q.coeffs ();  // Quaternion
            Eigen::Map<Eigen::Vector4f> (hPtrIOT + 4, 4) = t.homogeneous ();  // Translation
            hPtrIOT[7] = s;  // Scale

            pTime += cTimer.stop ();

            queue.enqueueWriteBuffer (dBufferIOT, CL_FALSE, 0, bufferTSize, hPtrIOT, nullptr, &timer.event ());
            queue.flush (); timer.wait ();
            pTime += timer.duration ();

            return pTime;
        }

    };


    /*! \brief Interface class for the `%ICP` pipeline using the Eigen library 
     *         to estimate the rotation and considering weighted residual errors.
     *  \details Performs one `%ICP` iteration. It accepts two sets (fixed and moving) 
     *           of landmarks, estimates the relative homogeneous transformation 
     *           \f$ (\dot{q}_k,t_k) \f$ between them, and transforms the moving set 
     *           according to this transformation.
     *  \note All computation is done on the `GPU`, except from the rotation 
     *        computation which is solved on the `CPU` with `Eigen::JacobiSVD`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created 
     *        by an `ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::WEIGHTED>` instance:<br>
     *        |  Name  | Type | Placement | I/O | Use | Properties | Size |
     *        |  ---   |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_F | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | H_IN_M | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | H_IO_T | Buffer | Host   | IO| Staging     | CL_MEM_READ_WRITE | \f$2*sizeof\ (cl\_float4)\f$ |
     *        | D_IN_F | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | D_IN_M | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | D_IO_T | Buffer | Device | IO| Processing  | CL_MEM_READ_WRITE | \f$2*sizeof\ (cl\_float4)\f$ |
     */
    template <>
    class ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::WEIGHTED>
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN_F,    /*!< Input staging buffer for the fixed set of landmarks. */
            H_IN_M,    /*!< Input staging buffer for the moving set of landmarks. */
            H_IO_T,    /*!< Input-output staging buffer for the quaternion and the translation vector. 
                        *   It is loaded with an initial estimation of the transformation 
                        *   and gets refined with every %ICP iteration.
                        *   The first `cl_float4` element contains the quaternion, \f$ \dot{q} = 
                        *   \left[ \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$, 
                        *   and the second `cl_float4` element contains the translation vector, 
                        *   \f$ t = \left[ \begin{matrix} t_x & t_y & t_z & 1 \end{matrix} \right]^T \f$. 
                        *   If scaling is desired, the factor should be placed in the last element 
                        *   of the translation vector, \f$ t = \left[ \begin{matrix} t_x & t_y 
                        *   & t_z & s \end{matrix} \right]^T \f$. */
            D_IN_F,    /*!< Input buffer for the fixed set of landmarks. */
            D_IN_M,    /*!< Input buffer for the moving set of landmarks. */
            D_IO_T,    /*!< Input-output buffer for the quaternion and the translation vector. 
                        *   It is loaded with an initial estimation of the transformation 
                        *   and gets refined with every %ICP iteration.
                        *   The first `cl_float4` element contains the quaternion, \f$ \dot{q} = 
                        *   \left[ \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$, 
                        *   and the second `cl_float4` element contains the translation vector, 
                        *   \f$ t = \left[ \begin{matrix} t_x & t_y & t_z & 1 \end{matrix} \right]^T \f$. 
                        *   If scaling is desired, the factor should be placed in the last element 
                        *   of the translation vector, \f$ t = \left[ \begin{matrix} t_x & t_y 
                        *   & t_z & s \end{matrix} \right]^T \f$. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        ICPStep (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _infoRBC, clutils::CLEnvInfo<1> _infoICP);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (ICPStep::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _m, unsigned int _nr, 
            float _a = 1e2f, float _c = 1e-6f, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (ICPStep::Memory mem = ICPStep::Memory::D_IN_F, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (ICPStep::Memory mem = ICPStep::Memory::H_IO_T, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr, bool config = false);
        /*! \brief Gets the scaling parameter \f$ \alpha \f$ involved in 
         *         the distance calculations of the `RBC` data structure. */
        float getAlpha ();
        /*! \brief Sets the scaling parameter \f$ \alpha \f$ involved in 
         *         the distance calculations of the `RBC` data structure. */
        void setAlpha (float _a);
        /*! \brief Gets the scaling factor c used when computing the `S` matrix. */
        float getScaling ();
        /*! \brief Sets the scaling factor c used when computing the `S` matrix. */
        void setScaling (float _c);

        cl_float *hPtrInF;  /*!< Mapping of the input staging buffer for the fixed set of points. */
        cl_float *hPtrInM;  /*!< Mapping of the input staging buffer for the moving set of points. */
        cl_float *hPtrIOT;  /*!< Mapping of the input-output staging buffer for the estimated 
                             *   quaternion and translation vector. */
        
        Eigen::Matrix3f Rk;     /*!< Represents the incremental development in the rotation estimation in 
                                 *   iteration `k`, given in rotation matrix representation, \f$ R_k \f$. */
        Eigen::Quaternionf qk;  /*!< Represents the incremental development in the rotation estimation in 
                                 *   iteration `k`, given in quaternion representation, \f$ \dot{q}_k = \left[ 
                                 *   \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$. */
        Eigen::Vector3f tk;     /*!< Represents the incremental development in the translation estimation 
                                 *   in iteration `k`, given as a vector in 3-D, \f$ t_k \f$. */
        cl_float sk;            /*!< Represents the incremental development in the scale estimation 
                                 *   in iteration `k`, given as a scalar, \f$ s_k \f$. */

        Eigen::Matrix3f R;     /*!< Represents the rotation estimation up to iteration `k`, 
                                *   given in rotation matrix representation, \f$ R \f$. */
        Eigen::Quaternionf q;  /*!< Represents the rotation estimation up to iteration `k`, 
                                *   given in quaternion representation, \f$ \dot{q} = \left[ 
                                *   \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$. */
        Eigen::Vector3f t;     /*!< Represents the translation estimation up to iteration `k`, 
                                *   given as a vector in 3-D, \f$ t \f$. */
        cl_float s;            /*!< Represents the scale estimation up to iteration `k`,
                                *   given as a scalar, \f$ s \f$. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> infoRBC, infoICP;
        cl::Context context;
        cl::CommandQueue queue;
        Staging staging;
        ICPReps fReps;
        RBC::RBCConstruct
            <RBC::KernelTypeC::KINECT_R, RBC::RBCPermuteConfig::GENERIC> rbcC;
        ICPTransform<ICPTransformConfig::QUATERNION> transform;
        RBC::RBCSearch
            <RBC::KernelTypeC::KINECT_R, 
                RBC::RBCPermuteConfig::GENERIC, RBC::KernelTypeS::KINECT> rbcS;
        ICPWeights weights;
        ICPMean<ICPMeanConfig::WEIGHTED> means;
        ICPDevs devs;
        ICPS<ICPSConfig::WEIGHTED> matrixS;

        cl_float *mean, *Sij;
        Eigen::Vector3f mf, mm;
        Eigen::Matrix3f S;

        float a, c;
        unsigned int m, nr, d;
        unsigned int bufferFMSize, bufferTSize;
        cl::Buffer hBufferInF, hBufferInM, hBufferIOT;
        cl::Buffer dBufferInF, dBufferInM, dBufferIOT;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \param[in] config flag. If true, configures the `RBC search` process. 
         *                    Set to true once, when the `RBC` data structure is reset.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, 
            const std::vector<cl::Event> *events = nullptr, bool config = false)
        {
            clutils::CPUTimer<double, std::milli> cTimer;
            double pTime = 0.0;

            if (config)
            {
                fReps.run (events);
                rbcC.run ();
            }

            pTime += transform.run (timer, events);
            pTime += rbcS.run (timer, nullptr, config);
            pTime += weights.run (timer);
            pTime += means.run (timer);
            pTime += devs.run (timer);
            pTime += matrixS.run (timer);

            cTimer.start ();

            mean = (cl_float *) means.read (ICPMean<ICPMeanConfig::WEIGHTED>::Memory::H_OUT, CL_FALSE);
            Sij = (cl_float *) matrixS.read (ICPS<ICPSConfig::WEIGHTED>::Memory::H_OUT);
            sk = std::sqrt (Sij[9] / Sij[10]);

            mf = Eigen::Map<Eigen::Vector3f> (mean);
            mm = Eigen::Map<Eigen::Vector3f> (mean + 4);
            S = Eigen::Map<Eigen::Matrix3f, Eigen::Unaligned, Eigen::Stride<1, 3> > (Sij);
            
            Eigen::JacobiSVD<Eigen::MatrixXf, Eigen::NoQRPreconditioner> 
                svd (S, Eigen::ComputeThinU | Eigen::ComputeThinV);

            Rk = svd.matrixV () * svd.matrixU ().transpose ();
            if (Rk.determinant () < 0)
            {
                Eigen::Matrix3f B = Eigen::Matrix3f::Identity ();
                B (2, 2) = Rk.determinant ();
                Rk = svd.matrixV () * B * svd.matrixU ().transpose ();
            }
            qk = Eigen::Quaternionf (Rk);

            tk = mf - sk * Rk * mm;

            R = Rk * R;
            q = Eigen::Quaternionf (R);
            t = sk * Rk * t + tk;
            s = sk * s;
            
            Eigen::Map<Eigen::Vector4f> (hPtrIOT, 4) = q.coeffs ();  // Quaternion
            Eigen::Map<Eigen::Vector4f> (hPtrIOT + 4, 4) = t.homogeneous ();  // Translation
            hPtrIOT[7] = s;  // Scale

            pTime += cTimer.stop ();

            queue.enqueueWriteBuffer (dBufferIOT, CL_FALSE, 0, bufferTSize, hPtrIOT, nullptr, &timer.event ());
            queue.flush (); timer.wait ();
            pTime += timer.duration ();

            return pTime;
        }

    };


    /*! \brief Interface class for the `%ICP` pipeline using the Power Method 
     *         to estimate the rotation and considering regular residual errors.
     *  \details Performs one `%ICP` iteration. It accepts two sets (fixed and moving) 
     *           of landmarks, estimates the relative homogeneous transformation 
     *           \f$ (\dot{q}_k,t_k) \f$ between them, and transforms the moving set 
     *           according to this transformation.
     *  \note All computation is done on the `GPU`. The rotation computation 
     *        is also solved on the `GPU` with the `Power Method`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created 
     *        by a `ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::REGULAR>` instance:<br>
     *        |  Name  | Type | Placement | I/O | Use | Properties | Size |
     *        |  ---   |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_F | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | H_IN_M | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | H_IO_T | Buffer | Host   | IO| Staging     | CL_MEM_READ_WRITE | \f$2*sizeof\ (cl\_float4)\f$ |
     *        | D_IN_F | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | D_IN_M | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | D_IO_T | Buffer | Device | IO| Processing  | CL_MEM_READ_WRITE | \f$2*sizeof\ (cl\_float4)\f$ |
     */
    template <>
    class ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::REGULAR>
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN_F,    /*!< Input staging buffer for the fixed set of landmarks. */
            H_IN_M,    /*!< Input staging buffer for the moving set of landmarks. */
            H_IO_T,    /*!< Input-output staging buffer for the quaternion and the translation vector. 
                        *   It is loaded with an initial estimation of the transformation 
                        *   and gets refined with every %ICP iteration.
                        *   The first `cl_float4` element contains the quaternion, \f$ \dot{q} = 
                        *   \left[ \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$, 
                        *   and the second `cl_float4` element contains the translation vector, 
                        *   \f$ t = \left[ \begin{matrix} t_x & t_y & t_z & 1 \end{matrix} \right]^T \f$. 
                        *   If scaling is desired, the factor should be placed in the last element 
                        *   of the translation vector, \f$ t = \left[ \begin{matrix} t_x & t_y 
                        *   & t_z & s \end{matrix} \right]^T \f$. */
            D_IN_F,    /*!< Input buffer for the fixed set of landmarks. */
            D_IN_M,    /*!< Input buffer for the moving set of landmarks. */
            D_IO_T,    /*!< Input-output buffer for the quaternion and the translation vector. 
                        *   It is loaded with an initial estimation of the transformation 
                        *   and gets refined with every %ICP iteration.
                        *   The first `cl_float4` element contains the quaternion, \f$ \dot{q} = 
                        *   \left[ \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$, 
                        *   and the second `cl_float4` element contains the translation vector, 
                        *   \f$ t = \left[ \begin{matrix} t_x & t_y & t_z & 1 \end{matrix} \right]^T \f$. 
                        *   If scaling is desired, the factor should be placed in the last element 
                        *   of the translation vector, \f$ t = \left[ \begin{matrix} t_x & t_y 
                        *   & t_z & s \end{matrix} \right]^T \f$. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        ICPStep (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _infoRBC, clutils::CLEnvInfo<1> _infoICP);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (ICPStep::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _m, unsigned int _nr, 
            float _a = 1e2f, float _c = 1e-6f, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (ICPStep::Memory mem = ICPStep::Memory::D_IN_F, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (ICPStep::Memory mem = ICPStep::Memory::H_IO_T, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr, bool config = false);
        /*! \brief Gets the scaling parameter \f$ \alpha \f$ involved in 
         *         the distance calculations of the `RBC` data structure. */
        float getAlpha ();
        /*! \brief Sets the scaling parameter \f$ \alpha \f$ involved in 
         *         the distance calculations of the `RBC` data structure. */
        void setAlpha (float _a);
        /*! \brief Gets the scaling factor c used when computing the `S` matrix. */
        float getScaling ();
        /*! \brief Sets the scaling factor c used when computing the `S` matrix. */
        void setScaling (float _c);

        cl_float *hPtrInF;  /*!< Mapping of the input staging buffer for the fixed set of points. */
        cl_float *hPtrInM;  /*!< Mapping of the input staging buffer for the moving set of points. */
        cl_float *hPtrIOT;  /*!< Mapping of the input-output staging buffer for the estimated 
                             *   quaternion and translation vector. */
        
        Eigen::Matrix3f Rk;     /*!< Represents the incremental development in the rotation estimation in 
                                 *   iteration `k`, given in rotation matrix representation, \f$ R_k \f$. */
        Eigen::Quaternionf qk;  /*!< Represents the incremental development in the rotation estimation in 
                                 *   iteration `k`, given in quaternion representation, \f$ \dot{q}_k = \left[ 
                                 *   \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$. */
        Eigen::Vector3f tk;     /*!< Represents the incremental development in the translation estimation 
                                 *   in iteration `k`, given as a vector in 3-D, \f$ t_k \f$. */
        cl_float sk;            /*!< Represents the incremental development in the scale estimation 
                                 *   in iteration `k`, given as a scalar, \f$ s_k \f$. */

        Eigen::Matrix3f R;     /*!< Represents the rotation estimation up to iteration `k`, 
                                *   given in rotation matrix representation, \f$ R \f$. */
        Eigen::Quaternionf q;  /*!< Represents the rotation estimation up to iteration `k`, 
                                *   given in quaternion representation, \f$ \dot{q} = \left[ 
                                *   \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$. */
        Eigen::Vector3f t;     /*!< Represents the translation estimation up to iteration `k`, 
                                *   given as a vector in 3-D, \f$ t \f$. */
        cl_float s;            /*!< Represents the scale estimation up to iteration `k`,
                                *   given as a scalar, \f$ s \f$. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> infoRBC, infoICP;
        cl::Context context;
        cl::CommandQueue queue;
        Staging staging;
        ICPReps fReps;
        RBC::RBCConstruct
            <RBC::KernelTypeC::KINECT_R, RBC::RBCPermuteConfig::GENERIC> rbcC;
        ICPTransform<ICPTransformConfig::QUATERNION> transform;
        RBC::RBCSearch
            <RBC::KernelTypeC::KINECT_R, 
                RBC::RBCPermuteConfig::GENERIC, RBC::KernelTypeS::KINECT> rbcS;
        ICPMean<ICPMeanConfig::REGULAR> means;
        ICPDevs devs;
        ICPS<ICPSConfig::REGULAR> matrixS;
        ICPPowerMethod powMethod;

        cl_float *Tk;

        float a, c;
        unsigned int m, nr, d;
        unsigned int bufferFMSize, bufferTSize;
        cl::Buffer hBufferInF, hBufferInM, hBufferIOT;
        cl::Buffer dBufferInF, dBufferInM, dBufferIOT;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \param[in] config flag. If true, configures the `RBC search` process. 
         *                    Set to true once, when the `RBC` data structure is reset.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, 
            const std::vector<cl::Event> *events = nullptr, bool config = false)
        {
            clutils::CPUTimer<double, std::milli> cTimer;
            double pTime = 0.0;

            if (config)
            {
                fReps.run (events);
                rbcC.run ();
            }

            pTime += transform.run (timer, events);
            pTime += rbcS.run (timer, nullptr, config);
            pTime += means.run (timer);
            pTime += devs.run (timer);
            pTime += matrixS.run (timer);
            pTime += powMethod.run (timer);
            
            cTimer.start ();

            Tk = (cl_float *) powMethod.read (ICPPowerMethod::Memory::H_OUT_T_K);

            qk = Eigen::Quaternionf (Tk);
            Rk = Eigen::Matrix3f (qk);
            tk = Eigen::Map<Eigen::Vector3f> (Tk + 4, 3);
            sk = Tk[7];

            R = Rk * R;
            q = Eigen::Quaternionf (R);
            t = sk * Rk * t + tk;
            s = sk * s;
            
            Eigen::Map<Eigen::Vector4f> (hPtrIOT, 4) = q.coeffs ();  // Quaternion
            Eigen::Map<Eigen::Vector4f> (hPtrIOT + 4, 4) = t.homogeneous ();  // Translation
            hPtrIOT[7] = s;  // Scale

            pTime += cTimer.stop ();

            queue.enqueueWriteBuffer (dBufferIOT, CL_FALSE, 0, bufferTSize, hPtrIOT, nullptr, &timer.event ());
            queue.flush (); timer.wait ();
            pTime += timer.duration ();

            return pTime;
        }

    };


    /*! \brief Interface class for the `%ICP` pipeline using the Power Method 
     *         to estimate the rotation and considering weighted residual errors.
     *  \details Performs one `%ICP` iteration. It accepts two sets (fixed and moving) 
     *           of landmarks, estimates the relative homogeneous transformation 
     *           \f$ (\dot{q}_k,t_k) \f$ between them, and transforms the moving set 
     *           according to this transformation.
     *  \note All computation is done on the `GPU`. The rotation computation 
     *        is also solved on the `GPU` with the `Power Method`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created 
     *        by a `ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::WEIGHTED>` instance:<br>
     *        |  Name  | Type | Placement | I/O | Use | Properties | Size |
     *        |  ---   |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_F | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | H_IN_M | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | H_IO_T | Buffer | Host   | IO| Staging     | CL_MEM_READ_WRITE | \f$2*sizeof\ (cl\_float4)\f$ |
     *        | D_IN_F | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | D_IN_M | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$m*sizeof\ (cl\_float8)\f$ |
     *        | D_IO_T | Buffer | Device | IO| Processing  | CL_MEM_READ_WRITE | \f$2*sizeof\ (cl\_float4)\f$ |
     */
    template <>
    class ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::WEIGHTED>
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         */
        enum class Memory : uint8_t
        {
            H_IN_F,    /*!< Input staging buffer for the fixed set of landmarks. */
            H_IN_M,    /*!< Input staging buffer for the moving set of landmarks. */
            H_IO_T,    /*!< Input-output staging buffer for the quaternion and the translation vector. 
                        *   It is loaded with an initial estimation of the transformation 
                        *   and gets refined with every %ICP iteration.
                        *   The first `cl_float4` element contains the quaternion, \f$ \dot{q} = 
                        *   \left[ \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$, 
                        *   and the second `cl_float4` element contains the translation vector, 
                        *   \f$ t = \left[ \begin{matrix} t_x & t_y & t_z & 1 \end{matrix} \right]^T \f$. 
                        *   If scaling is desired, the factor should be placed in the last element 
                        *   of the translation vector, \f$ t = \left[ \begin{matrix} t_x & t_y 
                        *   & t_z & s \end{matrix} \right]^T \f$. */
            D_IN_F,    /*!< Input buffer for the fixed set of landmarks. */
            D_IN_M,    /*!< Input buffer for the moving set of landmarks. */
            D_IO_T,    /*!< Input-output buffer for the quaternion and the translation vector. 
                        *   It is loaded with an initial estimation of the transformation 
                        *   and gets refined with every %ICP iteration.
                        *   The first `cl_float4` element contains the quaternion, \f$ \dot{q} = 
                        *   \left[ \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$, 
                        *   and the second `cl_float4` element contains the translation vector, 
                        *   \f$ t = \left[ \begin{matrix} t_x & t_y & t_z & 1 \end{matrix} \right]^T \f$. 
                        *   If scaling is desired, the factor should be placed in the last element 
                        *   of the translation vector, \f$ t = \left[ \begin{matrix} t_x & t_y 
                        *   & t_z & s \end{matrix} \right]^T \f$. */
        };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        ICPStep (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _infoRBC, clutils::CLEnvInfo<1> _infoICP);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (ICPStep::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _m, unsigned int _nr, 
            float _a = 1e2f, float _c = 1e-6f, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (ICPStep::Memory mem = ICPStep::Memory::D_IN_F, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (ICPStep::Memory mem = ICPStep::Memory::H_IO_T, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr, bool config = false);
        /*! \brief Gets the scaling parameter \f$ \alpha \f$ involved in 
         *         the distance calculations of the `RBC` data structure. */
        float getAlpha ();
        /*! \brief Sets the scaling parameter \f$ \alpha \f$ involved in 
         *         the distance calculations of the `RBC` data structure. */
        void setAlpha (float _a);
        /*! \brief Gets the scaling factor c used when computing the `S` matrix. */
        float getScaling ();
        /*! \brief Sets the scaling factor c used when computing the `S` matrix. */
        void setScaling (float _c);

        cl_float *hPtrInF;  /*!< Mapping of the input staging buffer for the fixed set of points. */
        cl_float *hPtrInM;  /*!< Mapping of the input staging buffer for the moving set of points. */
        cl_float *hPtrIOT;  /*!< Mapping of the input-output staging buffer for the estimated 
                             *   quaternion and translation vector. */
        
        Eigen::Matrix3f Rk;     /*!< Represents the incremental development in the rotation estimation in 
                                 *   iteration `k`, given in rotation matrix representation, \f$ R_k \f$. */
        Eigen::Quaternionf qk;  /*!< Represents the incremental development in the rotation estimation in 
                                 *   iteration `k`, given in quaternion representation, \f$ \dot{q}_k = \left[ 
                                 *   \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$. */
        Eigen::Vector3f tk;     /*!< Represents the incremental development in the translation estimation 
                                 *   in iteration `k`, given as a vector in 3-D, \f$ t_k \f$. */
        cl_float sk;            /*!< Represents the incremental development in the scale estimation 
                                 *   in iteration `k`, given as a scalar, \f$ s_k \f$. */

        Eigen::Matrix3f R;     /*!< Represents the rotation estimation up to iteration `k`, 
                                *   given in rotation matrix representation, \f$ R \f$. */
        Eigen::Quaternionf q;  /*!< Represents the rotation estimation up to iteration `k`, 
                                *   given in quaternion representation, \f$ \dot{q} = \left[ 
                                *   \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$. */
        Eigen::Vector3f t;     /*!< Represents the translation estimation up to iteration `k`, 
                                *   given as a vector in 3-D, \f$ t \f$. */
        cl_float s;            /*!< Represents the scale estimation up to iteration `k`,
                                *   given as a scalar, \f$ s \f$. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> infoRBC, infoICP;
        cl::Context context;
        cl::CommandQueue queue;
        Staging staging;
        ICPReps fReps;
        RBC::RBCConstruct
            <RBC::KernelTypeC::KINECT_R, RBC::RBCPermuteConfig::GENERIC> rbcC;
        ICPTransform<ICPTransformConfig::QUATERNION> transform;
        RBC::RBCSearch
            <RBC::KernelTypeC::KINECT_R, 
                RBC::RBCPermuteConfig::GENERIC, RBC::KernelTypeS::KINECT> rbcS;
        ICPWeights weights;
        ICPMean<ICPMeanConfig::WEIGHTED> means;
        ICPDevs devs;
        ICPS<ICPSConfig::WEIGHTED> matrixS;
        ICPPowerMethod powMethod;

        cl_float *Tk;

        float a, c;
        unsigned int m, nr, d;
        unsigned int bufferFMSize, bufferTSize;
        cl::Buffer hBufferInF, hBufferInM, hBufferIOT;
        cl::Buffer dBufferInF, dBufferInM, dBufferIOT;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \param[in] config flag. If true, configures the `RBC search` process. 
         *                    Set to true once, when the `RBC` data structure is reset.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, 
            const std::vector<cl::Event> *events = nullptr, bool config = false)
        {
            clutils::CPUTimer<double, std::milli> cTimer;
            double pTime = 0.0;

            if (config)
            {
                fReps.run (events);
                rbcC.run ();
            }

            pTime += transform.run (timer, events);
            pTime += rbcS.run (timer, nullptr, config);
            pTime += weights.run (timer);
            pTime += means.run (timer);
            pTime += devs.run (timer);
            pTime += matrixS.run (timer);
            pTime += powMethod.run (timer);
            
            cTimer.start ();

            Tk = (cl_float *) powMethod.read (ICPPowerMethod::Memory::H_OUT_T_K);

            qk = Eigen::Quaternionf (Tk);
            Rk = Eigen::Matrix3f (qk);
            tk = Eigen::Map<Eigen::Vector3f> (Tk + 4, 3);
            sk = Tk[7];

            R = Rk * R;
            q = Eigen::Quaternionf (R);
            t = sk * Rk * t + tk;
            s = sk * s;
            
            Eigen::Map<Eigen::Vector4f> (hPtrIOT, 4) = q.coeffs ();  // Quaternion
            Eigen::Map<Eigen::Vector4f> (hPtrIOT + 4, 4) = t.homogeneous ();  // Translation
            hPtrIOT[7] = s;  // Scale

            pTime += cTimer.stop ();

            queue.enqueueWriteBuffer (dBufferIOT, CL_FALSE, 0, bufferTSize, hPtrIOT, nullptr, &timer.event ());
            queue.flush (); timer.wait ();
            pTime += timer.duration ();

            return pTime;
        }

    };

}
}

#endif  // ICP_ALGORITHMS_HPP
