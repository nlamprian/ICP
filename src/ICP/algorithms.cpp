/*! \file algorithms.cpp
 *  \brief Defines classes that organize the execution of OpenCL kernels.
 *  \details Each class hides the details of the execution of a kernel. They
 *           initialize the necessary buffers, set up the workspaces, and 
 *           run the kernels.
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
#include <sstream>
#include <cmath>
#include <CLUtils.hpp>
#include <ICP/algorithms.hpp>


/*! \note All the classes assume there is a fully configured `clutils::CLEnv` 
 *        environment. This means, there is a known context on which they will 
 *        operate, there is a known command queue which they will use, and all 
 *        the necessary kernel code has been compiled. For more info on **CLUtils**, 
 *        you can check the [online documentation](https://clutils.nlamprian.me/).
 */
namespace cl_algo
{
namespace ICP
{

    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. It specifies the context, queue, etc, to be used.
     */
    template <>
    Reduce<ReduceConfig::MIN, cl_float>::Reduce (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        recKernel (env.getProgram (info.pgIdx), "reduce_min_f"), 
        groupRecKernel (env.getProgram (info.pgIdx), "reduce_min_f")
    {
        wgMultiple = recKernel.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. It specifies the context, queue, etc, to be used.
     */
    template <>
    Reduce<ReduceConfig::MAX, cl_uint>::Reduce (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        recKernel (env.getProgram (info.pgIdx), "reduce_max_ui"), 
        groupRecKernel (env.getProgram (info.pgIdx), "reduce_max_ui")
    {
        wgMultiple = recKernel.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. It specifies the context, queue, etc, to be used.
     */
    template <>
    Reduce<ReduceConfig::SUM, cl_float>::Reduce (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        recKernel (env.getProgram (info.pgIdx), "reduce_sum_f"), 
        groupRecKernel (env.getProgram (info.pgIdx), "reduce_sum_f")
    {
        wgMultiple = recKernel.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    template <ReduceConfig C, typename T>
    cl::Memory& Reduce<C, T>::get (Reduce::Memory mem)
    {
        switch (mem)
        {
            case Reduce::Memory::H_IN:
                return hBufferIn;
            case Reduce::Memory::H_OUT:
                return hBufferOut;
            case Reduce::Memory::D_IN:
                return dBufferIn;
            case Reduce::Memory::D_RED:
                return dBufferR;
            case Reduce::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _cols number of columns in the input array.
     *  \param[in] _rows number of rows in the input array.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    template <ReduceConfig C, typename T>
    void Reduce<C, T>::init (unsigned int _cols, unsigned int _rows, Staging _staging)
    {
        cols = _cols; rows = _rows;
        bufferInSize  = cols * rows * sizeof (T);
        bufferOutSize = rows * sizeof (T);
        staging = _staging;

        // Establish the number of work-groups per row
        wgXdim = std::ceil (cols / (float) (8 * wgMultiple));
        // Round up to a multiple of 4 (data are handled as float4)
        if ((wgXdim != 1) && (wgXdim % 4)) wgXdim += 4 - wgXdim % 4;

        bufferGRSize = wgXdim * rows * sizeof (T);

        try
        {
            if (wgXdim == 0)
                throw "The array cannot have zero columns";

            if (cols % 4 != 0)
                throw "The number of columns in the array must be a multiple of 4";

            // (8 * wgMultiple) elements per work-group
            // (8 * wgMultiple) work-groups maximum
            if (cols > std::pow (8 * wgMultiple, 2))
            {
                std::ostringstream ss;
                ss << "The current configuration of MinReduce supports arrays ";
                ss << "of up to " << std::pow (8 * wgMultiple, 2) << " columns";
                throw ss.str ().c_str ();
            }
        }
        catch (const char *error)
        {
            std::cerr << "Error[Reduce]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        globalR = cl::NDRange (wgXdim * wgMultiple, rows);
        globalGR = cl::NDRange (wgMultiple, rows);
        local = cl::NDRange (wgMultiple, 1);
        
        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrIn = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferIn () == nullptr)
                    hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);

                hPtrIn = (T *) queue.enqueueMapBuffer (
                    hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                queue.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                hPtrOut = (T *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io) hPtrIn = nullptr;
                break;
        }

        // Create device buffers
        if (dBufferIn () == nullptr)
            dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferR () == nullptr && wgXdim != 1)
            dBufferR = cl::Buffer (context, CL_MEM_READ_WRITE, bufferGRSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

        // Set kernel arguments
        if (wgXdim == 1)
        {
            recKernel.setArg (0, dBufferIn);
            recKernel.setArg (1, dBufferOut);
            recKernel.setArg (2, cl::Local (2 * local[0] * sizeof (T)));
            recKernel.setArg (3, cols / 4);
        }
        else
        {
            recKernel.setArg (0, dBufferIn);
            recKernel.setArg (1, dBufferR);
            recKernel.setArg (2, cl::Local (2 * local[0] * sizeof (T)));
            recKernel.setArg (3, cols / 4);

            groupRecKernel.setArg (0, dBufferR);
            groupRecKernel.setArg (1, dBufferOut);
            groupRecKernel.setArg (2, cl::Local (2 * local[0] * sizeof (T)));
            groupRecKernel.setArg (3, (cl_uint) (wgXdim / 4));
        }
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    template <ReduceConfig C, typename T>
    void Reduce<C, T>::write (Reduce::Memory mem, void *ptr, bool block, 
                              const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case Reduce::Memory::D_IN:
                    if (ptr != nullptr)
                        std::copy ((T *) ptr, (T *) ptr + cols * rows, hPtrIn);
                    queue.enqueueWriteBuffer (dBufferIn, block, 0, bufferInSize, hPtrIn, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    template <ReduceConfig C, typename T>
    void* Reduce<C, T>::read (Reduce::Memory mem, bool block, 
                              const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case Reduce::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferOutSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the kernel execution.
     */
    template <ReduceConfig C, typename T>
    void Reduce<C, T>::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (wgXdim == 1)
        {
            queue.enqueueNDRangeKernel (recKernel, cl::NullRange, globalR, local, events, event);
        }
        else
        {
            queue.enqueueNDRangeKernel (recKernel, cl::NullRange, globalR, local, events);
            queue.enqueueNDRangeKernel (groupRecKernel, cl::NullRange, globalGR, local, nullptr, event);
        }
    }


    /*! \brief Template instantiation for the case of `MIN` reduction and `cl_float` data. */
    template class Reduce<ReduceConfig::MIN, cl_float>;
    /*! \brief Template instantiation for the case of `MAX` reduction and `cl_uint` data. */
    template class Reduce<ReduceConfig::MAX, cl_uint>;
    /*! \brief Template instantiation for the case of `SUM` reduction and `cl_float` data. */
    template class Reduce<ReduceConfig::SUM, cl_float>;


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. It specifies the context, queue, etc, to be used.
     */
    template <>
    Scan<ScanConfig::INCLUSIVE, cl_int>::Scan (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernelScan (env.getProgram (info.pgIdx), "inclusiveScan_i"), 
        kernelSumsScan (env.getProgram (info.pgIdx), "inclusiveScan_i"), 
        kernelAddSums (env.getProgram (info.pgIdx), "addGroupSums_i")
    {
        wgMultiple = kernelScan.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. It specifies the context, queue, etc, to be used.
     */
    template <>
    Scan<ScanConfig::EXCLUSIVE, cl_int>::Scan (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernelScan (env.getProgram (info.pgIdx), "exclusiveScan_i"), 
        kernelSumsScan (env.getProgram (info.pgIdx), "inclusiveScan_i"), 
        kernelAddSums (env.getProgram (info.pgIdx), "addGroupSums_i")
    {
        wgMultiple = kernelScan.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    template <ScanConfig C, typename T>
    cl::Memory& Scan<C, T>::get (Scan::Memory mem)
    {
        switch (mem)
        {
            case Scan::Memory::H_IN:
                return hBufferIn;
            case Scan::Memory::H_OUT:
                return hBufferOut;
            case Scan::Memory::D_IN:
                return dBufferIn;
            case Scan::Memory::D_SUMS:
                return dBufferSums;
            case Scan::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *  \note Working with `float` elements and having large summations can be problematic.
     *        It is advised that a scaling is applied on the elements for better accuracy.
     *        
     *  \param[in] _cols number of columns in the input array.
     *  \param[in] _rows number of rows in the input array.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    template <ScanConfig C, typename T>
    void Scan<C, T>::init (unsigned int _cols, unsigned int _rows, Staging _staging)
    {
        cols = _cols; rows = _rows;
        bufferSize = cols * rows * sizeof (T);
        staging = _staging;

        // Establish the number of work-groups per row
        wgXdim = std::ceil (cols / (float) (8 * wgMultiple));
        // Round up to a multiple of 4 (data are handled as int4)
        if ((wgXdim != 1) && (wgXdim % 4)) wgXdim += 4 - wgXdim % 4;

        bufferSumsSize = wgXdim * rows * sizeof (T);

        try
        {
            if (wgXdim == 0)
                throw "The array cannot have zero columns";

            if (cols % 4 != 0)
                throw "The number of columns in the array must be a multiple of 4";

            // (8 * wgMultiple) elements per work-group
            // (8 * wgMultiple) work-groups maximum
            if (cols > std::pow (8 * wgMultiple, 2))
            {
                std::ostringstream ss;
                ss << "The current configuration of Scan supports arrays ";
                ss << "of up to " << std::pow (8 * wgMultiple, 2) << " columns";
                throw ss.str ().c_str ();
            }
        }
        catch (const char *error)
        {
            std::cerr << "Error[Scan]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        globalScan = cl::NDRange (wgXdim * wgMultiple, rows);
        localScan = cl::NDRange (wgMultiple, 1);
        globalSumsScan = cl::NDRange (wgMultiple, rows);
        globalAddSums = cl::NDRange (2 * (wgXdim - 1) * wgMultiple, rows);
        localAddSums = cl::NDRange (2 * wgMultiple, 1);
        offsetAddSums = cl::NDRange (2 * wgMultiple, 0);

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrIn = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferIn () == nullptr)
                    hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

                hPtrIn = (T *) queue.enqueueMapBuffer (
                    hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferSize);
                queue.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

                hPtrOut = (T *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io) hPtrIn = nullptr;
                break;
        }
        
        // Create device buffers
        if (dBufferIn () == nullptr)
            dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferSize);
        if (dBufferSums () == nullptr)
            dBufferSums = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSumsSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);

        // Set kernel arguments
        if (wgXdim == 1)
        {
            kernelScan.setArg (0, dBufferIn);
            kernelScan.setArg (1, dBufferOut);
            kernelScan.setArg (2, cl::Local (2 * localScan[0] * sizeof (T)));
            kernelScan.setArg (3, dBufferSums);  // Unused
            kernelScan.setArg (4, cols / 4);
        }
        else
        {
            kernelScan.setArg (0, dBufferIn);
            kernelScan.setArg (1, dBufferOut);
            kernelScan.setArg (2, cl::Local (2 * localScan[0] * sizeof (T)));
            kernelScan.setArg (3, dBufferSums);
            kernelScan.setArg (4, cols / 4);

            kernelSumsScan.setArg (0, dBufferSums);
            kernelSumsScan.setArg (1, dBufferSums);
            kernelSumsScan.setArg (2, cl::Local (2 * localScan[0] * sizeof (T)));
            kernelSumsScan.setArg (3, dBufferSums);  // Unused
            kernelSumsScan.setArg (4, (cl_uint) (wgXdim / 4));

            kernelAddSums.setArg (0, dBufferSums);
            kernelAddSums.setArg (1, dBufferOut);
            kernelAddSums.setArg (2, cols / 4);
        }
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    template <ScanConfig C, typename T>
    void Scan<C, T>::write (Scan::Memory mem, void *ptr, bool block, 
                            const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case Scan::Memory::D_IN:
                    if (ptr != nullptr)
                        std::copy ((T *) ptr, (T *) ptr + cols * rows, hPtrIn);
                    queue.enqueueWriteBuffer (dBufferIn, block, 0, bufferSize, hPtrIn, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    template <ScanConfig C, typename T>
    void* Scan<C, T>::read (Scan::Memory mem, bool block, 
                            const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case Scan::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the last kernel execution.
     */
    template <ScanConfig C, typename T>
    void Scan<C, T>::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (wgXdim == 1)
        {
            queue.enqueueNDRangeKernel (
                kernelScan, cl::NullRange, globalScan, localScan, events, event);
        }
        else
        {
            queue.enqueueNDRangeKernel (
                kernelScan, cl::NullRange, globalScan, localScan, events);

            queue.enqueueNDRangeKernel (
                kernelSumsScan, cl::NullRange, globalSumsScan, localScan);

            queue.enqueueNDRangeKernel (
                kernelAddSums, offsetAddSums, globalAddSums, localAddSums, nullptr, event);
        }
    }


    /*! \brief Template instantiation for `inclusive` scan and `int` data. */
    template class Scan<ScanConfig::INCLUSIVE, cl_int>;
    /*! \brief Template instantiation for `exclusive` scan and `int` data. */
    template class Scan<ScanConfig::EXCLUSIVE, cl_int>;


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. It specifies the context, queue, etc, to be used.
     */
    ICPLMs::ICPLMs (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "getLMs"), 
        n (640 * 480), m (128 * 128), d (8)
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& ICPLMs::get (ICPLMs::Memory mem)
    {
        switch (mem)
        {
            case ICPLMs::Memory::H_IN:
                return hBufferIn;
            case ICPLMs::Memory::H_OUT:
                return hBufferOut;
            case ICPLMs::Memory::D_IN:
                return dBufferIn;
            case ICPLMs::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void ICPLMs::init (Staging _staging)
    {
        bufferInSize = n * sizeof (cl_float8);
        bufferOutSize = m * sizeof (cl_float8);
        staging = _staging;

        // Set workspaces
        global = cl::NDRange (256, 128);

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrIn = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferIn () == nullptr)
                    hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);

                hPtrIn = (cl_float *) queue.enqueueMapBuffer (
                    hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                queue.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                hPtrOut = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io) hPtrIn = nullptr;
                break;
        }
        
        // Create device buffers
        if (dBufferIn () == nullptr)
            dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferIn);
        kernel.setArg (1, dBufferOut);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void ICPLMs::write (ICPLMs::Memory mem, void *ptr, bool block, 
                           const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPLMs::Memory::D_IN:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + n * d, hPtrIn);
                    queue.enqueueWriteBuffer (dBufferIn, block, 0, bufferInSize, hPtrIn, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* ICPLMs::read (ICPLMs::Memory mem, bool block, 
                           const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPLMs::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferOutSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the last kernel execution.
     */
    void ICPLMs::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, event);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. It specifies the context, queue, etc, to be used.
     */
    ICPReps::ICPReps (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "getReps"), 
        m (128 * 128), d (8)
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& ICPReps::get (ICPReps::Memory mem)
    {
        switch (mem)
        {
            case ICPReps::Memory::H_IN:
                return hBufferIn;
            case ICPReps::Memory::H_OUT:
                return hBufferOut;
            case ICPReps::Memory::D_IN:
                return dBufferIn;
            case ICPReps::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _nr number of representatives in the output array.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void ICPReps::init (unsigned int _nr, Staging _staging)
    {
        nr = _nr;
        bufferInSize = m * sizeof (cl_float8);
        bufferOutSize = nr * sizeof (cl_float8);
        staging = _staging;

        try
        {
            if (nr == 0)
                throw "The number of representatives cannot be zero";

            if (nr % 4)
                throw "The number of representatives has to be a multiple of 4";
        }
        catch (const char *error)
        {
            std::cerr << "Error[ICPReps]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // e.g. nr = 32 -> nrx = 8, nry = 4
        int p = std::log2 (nr);
        nrx = std::pow (2, p - p / 2);
        nry = std::pow (2, p / 2);

        // Set workspaces
        global = cl::NDRange (nrx, nry);

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrIn = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferIn () == nullptr)
                    hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);

                hPtrIn = (cl_float *) queue.enqueueMapBuffer (
                    hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                queue.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                hPtrOut = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io) hPtrIn = nullptr;
                break;
        }
        
        // Create device buffers
        if (dBufferIn () == nullptr)
            dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferIn);
        kernel.setArg (1, dBufferOut);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void ICPReps::write (ICPReps::Memory mem, void *ptr, bool block, 
                            const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPReps::Memory::D_IN:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + m * d, hPtrIn);
                    queue.enqueueWriteBuffer (dBufferIn, block, 0, bufferInSize, hPtrIn, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* ICPReps::read (ICPReps::Memory mem, bool block, 
                            const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPReps::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferOutSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the last kernel execution.
     */
    void ICPReps::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, event);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. It specifies the context, queue, etc, to be used.
     */
    ICPWeights::ICPWeights (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        weightKernel (env.getProgram (info.pgIdx), "icpComputeReduceWeights"), 
        groupWeightKernel (env.getProgram (info.pgIdx), "reduce_sum_fd")
    {
        wgMultiple = weightKernel.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& ICPWeights::get (ICPWeights::Memory mem)
    {
        switch (mem)
        {
            case ICPWeights::Memory::H_IN:
                return hBufferIn;
            case ICPWeights::Memory::H_OUT_W:
                return hBufferOutW;
            case ICPWeights::Memory::H_OUT_SUM_W:
                return hBufferOutSW;
            case ICPWeights::Memory::D_IN:
                return dBufferIn;
            case ICPWeights::Memory::D_OUT_W:
                return dBufferOutW;
            case ICPWeights::Memory::D_GW:
                return dBufferGW;
            case ICPWeights::Memory::D_OUT_SUM_W:
                return dBufferOutSW;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _n number of elements in the input sets.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void ICPWeights::init (unsigned int _n, Staging _staging)
    {
        n = _n;
        bufferInSize  = n * sizeof (rbc_dist_id);
        bufferOutWSize = n * sizeof (cl_float);
        bufferOutSWSize = sizeof (cl_double);
        staging = _staging;

        // Establish the number of work-groups per row
        wgXdim = std::ceil (n / (float) (2 * wgMultiple));
        // Round up to a multiple of 4 (data are handled as float4 in reduce_sum)
        if ((wgXdim != 1) && (wgXdim % 4)) wgXdim += 4 - wgXdim % 4;

        bufferGWSize = wgXdim * sizeof (cl_double);

        try
        {
            if (wgXdim == 0)
                throw "The array cannot have zero elements";

            if (n % 2 != 0)
                throw "The number of elements in the array must be a multiple of 2";

            // (2 * wgMultiple) elements per work-group
            // (8 * wgMultiple) work-groups maximum
            if (n > 16 * wgMultiple * wgMultiple)
            {
                std::ostringstream ss;
                ss << "The current configuration of ICPWeights supports arrays ";
                ss << "of up to " << 16 * wgMultiple * wgMultiple << " elements";
                throw ss.str ().c_str ();
            }
        }
        catch (const char *error)
        {
            std::cerr << "Error[ICPWeights]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Extract kernel
        if (wgXdim > 1)
            weightKernel = cl::Kernel (env.getProgram (info.pgIdx), "icpComputeReduceWeights_WG");

        // Set workspaces
        globalW = cl::NDRange (wgXdim * wgMultiple);
        globalGW = cl::NDRange (wgMultiple);
        local = cl::NDRange (wgMultiple);
        
        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrIn = nullptr;
                hPtrOutW = nullptr;
                hPtrOutSW = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferIn () == nullptr)
                    hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);

                hPtrIn = (rbc_dist_id *) queue.enqueueMapBuffer (
                    hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                queue.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                if (!io)
                {
                    queue.finish ();
                    hPtrOutW = nullptr;
                    hPtrOutSW = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOutW () == nullptr)
                    hBufferOutW = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutWSize);
                if (hBufferOutSW () == nullptr)
                    hBufferOutSW = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSWSize);

                hPtrOutW = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOutW, CL_FALSE, CL_MAP_READ, 0, bufferOutWSize);
                hPtrOutSW = (cl_double *) queue.enqueueMapBuffer (
                    hBufferOutSW, CL_FALSE, CL_MAP_READ, 0, bufferOutSWSize);
                queue.enqueueUnmapMemObject (hBufferOutW, hPtrOutW);
                queue.enqueueUnmapMemObject (hBufferOutSW, hPtrOutSW);
                queue.finish ();

                if (!io) hPtrIn = nullptr;
                break;
        }

        // Create device buffers
        if (dBufferIn () == nullptr)
            dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferOutW () == nullptr)
            dBufferOutW = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutWSize);
        if (dBufferGW () == nullptr && wgXdim != 1)
            dBufferGW = cl::Buffer (context, CL_MEM_READ_WRITE, bufferGWSize);
        if (dBufferOutSW () == nullptr)
            dBufferOutSW = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSWSize);

        // Set kernel arguments
        if (wgXdim == 1)
        {
            weightKernel.setArg (0, dBufferIn);
            weightKernel.setArg (1, dBufferOutW);
            weightKernel.setArg (2, dBufferOutSW);
            weightKernel.setArg (3, cl::Local (2 * local[0] * sizeof (cl_float)));
            weightKernel.setArg (4, n);
        }
        else
        {
            weightKernel.setArg (0, dBufferIn);
            weightKernel.setArg (1, dBufferOutW);
            weightKernel.setArg (2, dBufferGW);
            weightKernel.setArg (3, cl::Local (2 * local[0] * sizeof (cl_float)));
            weightKernel.setArg (4, n);

            groupWeightKernel.setArg (0, dBufferGW);
            groupWeightKernel.setArg (1, dBufferOutSW);
            groupWeightKernel.setArg (2, cl::Local (2 * local[0] * sizeof (cl_double)));
            groupWeightKernel.setArg (3, (cl_uint) (wgXdim / 4));
        }
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void ICPWeights::write (ICPWeights::Memory mem, void *ptr, bool block, 
                            const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPWeights::Memory::D_IN:
                    if (ptr != nullptr)
                        std::copy ((rbc_dist_id *) ptr, (rbc_dist_id *) ptr + n, hPtrIn);
                    queue.enqueueWriteBuffer (dBufferIn, block, 0, bufferInSize, hPtrIn, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* ICPWeights::read (ICPWeights::Memory mem, bool block, 
                            const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPWeights::Memory::H_OUT_W:
                    queue.enqueueReadBuffer (dBufferOutW, block, 0, bufferOutWSize, hPtrOutW, events, event);
                    return hPtrOutW;
                case ICPWeights::Memory::H_OUT_SUM_W:
                    queue.enqueueReadBuffer (dBufferOutSW, block, 0, bufferOutSWSize, hPtrOutSW, events, event);
                    return hPtrOutSW;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the kernel execution.
     */
    void ICPWeights::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (wgXdim == 1)
        {
            queue.enqueueNDRangeKernel (weightKernel, cl::NullRange, globalW, local, events, event);
        }
        else
        {
            queue.enqueueNDRangeKernel (weightKernel, cl::NullRange, globalW, local, events);
            queue.enqueueNDRangeKernel (groupWeightKernel, cl::NullRange, globalGW, local, nullptr, event);
        }
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. It specifies the context, queue, etc, to be used.
     */
    ICPMean<ICPMeanConfig::REGULAR>::ICPMean (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        meanKernel (env.getProgram (info.pgIdx), "icpMean"), 
        groupMeanKernel (env.getProgram (info.pgIdx), "icpGMean"), d (8)
    {
        wgMultiple = meanKernel.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& ICPMean<ICPMeanConfig::REGULAR>::get (ICPMean::Memory mem)
    {
        switch (mem)
        {
            case ICPMean::Memory::H_IN_F:
                return hBufferInF;
            case ICPMean::Memory::H_IN_M:
                return hBufferInM;
            case ICPMean::Memory::H_OUT:
                return hBufferOut;
            case ICPMean::Memory::D_IN_F:
                return dBufferInF;
            case ICPMean::Memory::D_IN_M:
                return dBufferInM;
            case ICPMean::Memory::D_GM:
                return dBufferGM;
            case ICPMean::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _n number of points in the input array.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void ICPMean<ICPMeanConfig::REGULAR>::init (unsigned int _n, Staging _staging)
    {
        n = _n;
        bufferInSize  = n * sizeof (cl_float8);
        bufferOutSize = 2 * sizeof (cl_float4);
        staging = _staging;

        // Establish the number of work-groups per row
        wgXdim = std::ceil (n / (float) (2 * wgMultiple));

        bufferGMSize = 2 * (wgXdim * sizeof (cl_float4));

        try
        {
            if (wgXdim == 0)
                throw "The array cannot have zero points";

            if (n % 2 != 0)
                throw "The number of points in the array must be a multiple of 2";

            // (2 * wgMultiple) points per work-group
            // (2 * wgMultiple) work-groups maximum
            if (n > std::pow (2 * wgMultiple, 2))
            {
                std::ostringstream ss;
                ss << "The current configuration of ICPMean<ICPMeanConfig::REGULAR> supports arrays ";
                ss << "of up to " << std::pow (2 * wgMultiple, 2) << " points";
                throw ss.str ().c_str ();
            }
        }
        catch (const char *error)
        {
            std::cerr << "Error[ICPMean<ICPMeanConfig::REGULAR>]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        globalM = cl::NDRange (wgXdim * wgMultiple, 2);
        globalGM = cl::NDRange (wgMultiple, 2);
        local = cl::NDRange (wgMultiple, 1);
        
        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInF = nullptr;
                hPtrInM = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInF () == nullptr)
                    hBufferInF = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);
                if (hBufferInM () == nullptr)
                    hBufferInM = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);

                hPtrInF = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInF, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                hPtrInM = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInM, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                queue.enqueueUnmapMemObject (hBufferInF, hPtrInF);
                queue.enqueueUnmapMemObject (hBufferInM, hPtrInM);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                hPtrOut = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io)
                {
                    hPtrInF = nullptr;
                    hPtrInM = nullptr;
                }
                break;
        }

        // Create device buffers
        if (dBufferInF () == nullptr)
            dBufferInF = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferInM () == nullptr)
            dBufferInM = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferGM () == nullptr && wgXdim != 1)
            dBufferGM = cl::Buffer (context, CL_MEM_READ_WRITE, bufferGMSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

        // Set kernel arguments
        if (wgXdim == 1)
        {
            meanKernel.setArg (0, dBufferInF);
            meanKernel.setArg (1, dBufferInM);
            meanKernel.setArg (2, dBufferOut);
            meanKernel.setArg (3, cl::Local (local[0] * 6 * sizeof (cl_float)));
            meanKernel.setArg (4, n);
        }
        else
        {
            meanKernel.setArg (0, dBufferInF);
            meanKernel.setArg (1, dBufferInM);
            meanKernel.setArg (2, dBufferGM);
            meanKernel.setArg (3, cl::Local (local[0] * 6 * sizeof (cl_float)));
            meanKernel.setArg (4, n);

            groupMeanKernel.setArg (0, dBufferGM);
            groupMeanKernel.setArg (1, dBufferOut);
            groupMeanKernel.setArg (2, cl::Local (local[0] * 6 * sizeof (cl_float)));
            groupMeanKernel.setArg (3, (cl_uint) wgXdim);
        }
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void ICPMean<ICPMeanConfig::REGULAR>::write (ICPMean::Memory mem, void *ptr, bool block, 
                                                 const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPMean::Memory::D_IN_F:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + n * d, hPtrInF);
                    queue.enqueueWriteBuffer (dBufferInF, block, 0, bufferInSize, hPtrInF, events, event);
                    break;
                case ICPMean::Memory::D_IN_M:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + n * d, hPtrInM);
                    queue.enqueueWriteBuffer (dBufferInM, block, 0, bufferInSize, hPtrInM, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* ICPMean<ICPMeanConfig::REGULAR>::read (ICPMean::Memory mem, bool block, 
                                                 const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPMean::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferOutSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the kernel execution.
     */
    void ICPMean<ICPMeanConfig::REGULAR>::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (wgXdim == 1)
        {
            queue.enqueueNDRangeKernel (meanKernel, cl::NullRange, globalM, local, events, event);
        }
        else
        {
            queue.enqueueNDRangeKernel (meanKernel, cl::NullRange, globalM, local, events);
            queue.enqueueNDRangeKernel (groupMeanKernel, cl::NullRange, globalGM, local, nullptr, event);
        }
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. It specifies the context, queue, etc, to be used.
     */
    ICPMean<ICPMeanConfig::WEIGHTED>::ICPMean (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        meanKernel (env.getProgram (info.pgIdx), "icpMean_Weighted"), 
        groupMeanKernel (env.getProgram (info.pgIdx), "icpGMean"), d (8)
    {
        wgMultiple = meanKernel.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& ICPMean<ICPMeanConfig::WEIGHTED>::get (ICPMean::Memory mem)
    {
        switch (mem)
        {
            case ICPMean::Memory::H_IN_F:
                return hBufferInF;
            case ICPMean::Memory::H_IN_M:
                return hBufferInM;
            case ICPMean::Memory::H_IN_W:
                return hBufferInW;
            case ICPMean::Memory::H_IN_SUM_W:
                return hBufferInSW;
            case ICPMean::Memory::H_OUT:
                return hBufferOut;
            case ICPMean::Memory::D_IN_F:
                return dBufferInF;
            case ICPMean::Memory::D_IN_M:
                return dBufferInM;
            case ICPMean::Memory::D_IN_W:
                return dBufferInW;
            case ICPMean::Memory::D_IN_SUM_W:
                return dBufferInSW;
            case ICPMean::Memory::D_GM:
                return dBufferGM;
            case ICPMean::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _n number of points in the input sets.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void ICPMean<ICPMeanConfig::WEIGHTED>::init (unsigned int _n, Staging _staging)
    {
        n = _n;
        bufferInFMSize  = n * sizeof (cl_float8);
        bufferInWSize  = n * sizeof (cl_float);
        bufferInSWSize  = sizeof (cl_double);
        bufferOutSize = 2 * sizeof (cl_float4);
        staging = _staging;

        // Establish the number of work-groups per row
        wgXdim = std::ceil (n / (float) (2 * wgMultiple));

        bufferGMSize = 2 * (wgXdim * sizeof (cl_float4));

        try
        {
            if (wgXdim == 0)
                throw "The array cannot have zero points";

            if (n % 2 != 0)
                throw "The number of points in the array must be a multiple of 2";

            // (2 * wgMultiple) points per work-group
            // (2 * wgMultiple) work-groups maximum
            if (n > std::pow (2 * wgMultiple, 2))
            {
                std::ostringstream ss;
                ss << "The current configuration of ICPMean<ICPMeanConfig::WEIGHTED> supports arrays ";
                ss << "of up to " << std::pow (2 * wgMultiple, 2) << " points";
                throw ss.str ().c_str ();
            }
        }
        catch (const char *error)
        {
            std::cerr << "Error[ICPMean<ICPMeanConfig::WEIGHTED>]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        globalM = cl::NDRange (wgXdim * wgMultiple, 2);
        globalGM = cl::NDRange (wgMultiple, 2);
        local = cl::NDRange (wgMultiple, 1);
        
        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInF = nullptr;
                hPtrInM = nullptr;
                hPtrInW = nullptr;
                hPtrInSW = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInF () == nullptr)
                    hBufferInF = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInFMSize);
                if (hBufferInM () == nullptr)
                    hBufferInM = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInFMSize);
                if (hBufferInW () == nullptr)
                    hBufferInW = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInWSize);
                if (hBufferInSW () == nullptr)
                    hBufferInSW = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSWSize);

                hPtrInF = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInF, CL_FALSE, CL_MAP_WRITE, 0, bufferInFMSize);
                hPtrInM = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInM, CL_FALSE, CL_MAP_WRITE, 0, bufferInFMSize);
                hPtrInW = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInW, CL_FALSE, CL_MAP_WRITE, 0, bufferInWSize);
                hPtrInSW = (cl_double *) queue.enqueueMapBuffer (
                    hBufferInSW, CL_FALSE, CL_MAP_WRITE, 0, bufferInSWSize);
                queue.enqueueUnmapMemObject (hBufferInF, hPtrInF);
                queue.enqueueUnmapMemObject (hBufferInM, hPtrInM);
                queue.enqueueUnmapMemObject (hBufferInW, hPtrInW);
                queue.enqueueUnmapMemObject (hBufferInSW, hPtrInSW);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                hPtrOut = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io)
                {
                    hPtrInF = nullptr;
                    hPtrInM = nullptr;
                    hPtrInW = nullptr;
                    hPtrInSW = nullptr;
                }
                break;
        }

        // Create device buffers
        if (dBufferInF () == nullptr)
            dBufferInF = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInFMSize);
        if (dBufferInM () == nullptr)
            dBufferInM = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInFMSize);
        if (dBufferInW () == nullptr)
            dBufferInW = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInWSize);
        if (dBufferInSW () == nullptr)
            dBufferInSW = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSWSize);
        if (dBufferGM () == nullptr && wgXdim != 1)
            dBufferGM = cl::Buffer (context, CL_MEM_READ_WRITE, bufferGMSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

        // Set kernel arguments
        if (wgXdim == 1)
        {
            meanKernel.setArg (0, dBufferInF);
            meanKernel.setArg (1, dBufferInM);
            meanKernel.setArg (2, dBufferOut);
            meanKernel.setArg (3, dBufferInW);
            meanKernel.setArg (4, dBufferInSW);
            meanKernel.setArg (5, cl::Local (local[0] * 6 * sizeof (cl_float)));
            meanKernel.setArg (6, n);
        }
        else
        {
            meanKernel.setArg (0, dBufferInF);
            meanKernel.setArg (1, dBufferInM);
            meanKernel.setArg (2, dBufferGM);
            meanKernel.setArg (3, dBufferInW);
            meanKernel.setArg (4, dBufferInSW);
            meanKernel.setArg (5, cl::Local (local[0] * 6 * sizeof (cl_float)));
            meanKernel.setArg (6, n);

            groupMeanKernel.setArg (0, dBufferGM);
            groupMeanKernel.setArg (1, dBufferOut);
            groupMeanKernel.setArg (2, cl::Local (local[0] * 6 * sizeof (cl_float)));
            groupMeanKernel.setArg (3, (cl_uint) wgXdim);
        }
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void ICPMean<ICPMeanConfig::WEIGHTED>::write (ICPMean::Memory mem, void *ptr, bool block, 
                                                  const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPMean::Memory::D_IN_F:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + n * d, hPtrInF);
                    queue.enqueueWriteBuffer (dBufferInF, block, 0, bufferInFMSize, hPtrInF, events, event);
                    break;
                case ICPMean::Memory::D_IN_M:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + n * d, hPtrInM);
                    queue.enqueueWriteBuffer (dBufferInM, block, 0, bufferInFMSize, hPtrInM, events, event);
                    break;
                case ICPMean::Memory::D_IN_W:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + n, hPtrInW);
                    queue.enqueueWriteBuffer (dBufferInW, block, 0, bufferInWSize, hPtrInW, events, event);
                    break;
                case ICPMean::Memory::D_IN_SUM_W:
                    if (ptr != nullptr)
                        std::copy ((cl_double *) ptr, (cl_double *) ptr + 1, hPtrInSW);
                    queue.enqueueWriteBuffer (dBufferInSW, block, 0, bufferInSWSize, hPtrInSW, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* ICPMean<ICPMeanConfig::WEIGHTED>::read (ICPMean::Memory mem, bool block, 
                                                  const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPMean::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferOutSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the kernel execution.
     */
    void ICPMean<ICPMeanConfig::WEIGHTED>::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (wgXdim == 1)
        {
            queue.enqueueNDRangeKernel (meanKernel, cl::NullRange, globalM, local, events, event);
        }
        else
        {
            queue.enqueueNDRangeKernel (meanKernel, cl::NullRange, globalM, local, events);
            queue.enqueueNDRangeKernel (groupMeanKernel, cl::NullRange, globalGM, local, nullptr, event);
        }
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. It specifies the context, queue, etc, to be used.
     */
    ICPDevs::ICPDevs (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "icpSubtractMean"), d (8)
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& ICPDevs::get (ICPDevs::Memory mem)
    {
        switch (mem)
        {
            case ICPDevs::Memory::H_IN_F:
                return hBufferInF;
            case ICPDevs::Memory::H_IN_M:
                return hBufferInM;
            case ICPDevs::Memory::H_IN_MEAN:
                return hBufferInMean;
            case ICPDevs::Memory::H_OUT_DEV_F:
                return hBufferOutDF;
            case ICPDevs::Memory::H_OUT_DEV_M:
                return hBufferOutDM;
            case ICPDevs::Memory::D_IN_F:
                return dBufferInF;
            case ICPDevs::Memory::D_IN_M:
                return dBufferInM;
            case ICPDevs::Memory::D_IN_MEAN:
                return dBufferInMean;
            case ICPDevs::Memory::D_OUT_DEV_F:
                return dBufferOutDF;
            case ICPDevs::Memory::D_OUT_DEV_M:
                return dBufferOutDM;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _n number of points.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void ICPDevs::init (unsigned int _n, Staging _staging)
    {
        n = _n;
        bufferInFMSize = n * sizeof (cl_float8);
        bufferInMeanSize = 2 * sizeof (cl_float4);
        bufferOutSize = n * sizeof (cl_float4);
        staging = _staging;

        try
        {
            if (n == 0)
                throw "The array cannot have zero points";
        }
        catch (const char *error)
        {
            std::cerr << "Error[ICPDevs]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        global = cl::NDRange (n, 2);

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInF = nullptr;
                hPtrInM = nullptr;
                hPtrInMean = nullptr;
                hPtrOutDevF = nullptr;
                hPtrOutDevM = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInF () == nullptr)
                    hBufferInF = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInFMSize);
                if (hBufferInM () == nullptr)
                    hBufferInM = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInFMSize);
                if (hBufferInMean () == nullptr)
                    hBufferInMean = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInMeanSize);

                hPtrInF = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInF, CL_FALSE, CL_MAP_WRITE, 0, bufferInFMSize);
                hPtrInM = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInM, CL_FALSE, CL_MAP_WRITE, 0, bufferInFMSize);
                hPtrInMean = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInMean, CL_FALSE, CL_MAP_WRITE, 0, bufferInMeanSize);
                queue.enqueueUnmapMemObject (hBufferInF, hPtrInF);
                queue.enqueueUnmapMemObject (hBufferInM, hPtrInM);
                queue.enqueueUnmapMemObject (hBufferInMean, hPtrInMean);

                if (!io)
                {
                    queue.finish ();
                    hPtrOutDevF = nullptr;
                    hPtrOutDevM = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOutDF () == nullptr)
                    hBufferOutDF = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);
                if (hBufferOutDM () == nullptr)
                    hBufferOutDM = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                hPtrOutDevF = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOutDF, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                hPtrOutDevM = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOutDM, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                queue.enqueueUnmapMemObject (hBufferOutDF, hPtrOutDevF);
                queue.enqueueUnmapMemObject (hBufferOutDM, hPtrOutDevM);
                queue.finish ();

                if (!io)
                {
                    hPtrInF = nullptr;
                    hPtrInM = nullptr;
                    hPtrInMean = nullptr;
                }
                break;
        }
        
        // Create device buffers
        if (dBufferInF () == nullptr)
            dBufferInF = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInFMSize);
        if (dBufferInM () == nullptr)
            dBufferInM = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInFMSize);
        if (dBufferInMean () == nullptr)
            dBufferInMean = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInMeanSize);
        if (dBufferOutDF () == nullptr)
            dBufferOutDF = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);
        if (dBufferOutDM () == nullptr)
            dBufferOutDM = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferInF);
        kernel.setArg (1, dBufferInM);
        kernel.setArg (2, dBufferOutDF);
        kernel.setArg (3, dBufferOutDM);
        kernel.setArg (4, dBufferInMean);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void ICPDevs::write (ICPDevs::Memory mem, void *ptr, bool block, 
                         const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPDevs::Memory::D_IN_F:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + n * d, hPtrInF);
                    queue.enqueueWriteBuffer (dBufferInF, block, 0, bufferInFMSize, hPtrInF, events, event);
                    break;
                case ICPDevs::Memory::D_IN_M:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + n * d, hPtrInM);
                    queue.enqueueWriteBuffer (dBufferInM, block, 0, bufferInFMSize, hPtrInM, events, event);
                    break;
                case ICPDevs::Memory::D_IN_MEAN:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + 4, hPtrInMean);
                    queue.enqueueWriteBuffer (dBufferInMean, block, 0, bufferInMeanSize, hPtrInMean, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* ICPDevs::read (ICPDevs::Memory mem, bool block, 
                         const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPDevs::Memory::H_OUT_DEV_F:
                    queue.enqueueReadBuffer (dBufferOutDF, block, 0, bufferOutSize, hPtrOutDevF, events, event);
                    return hPtrOutDevF;
                case ICPDevs::Memory::H_OUT_DEV_M:
                    queue.enqueueReadBuffer (dBufferOutDM, block, 0, bufferOutSize, hPtrOutDevM, events, event);
                    return hPtrOutDevM;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the last kernel execution.
     */
    void ICPDevs::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, event);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. It specifies the context, queue, etc, to be used.
     */
    ICPS<ICPSConfig::REGULAR>::ICPS (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "icpSijProducts"), 
        reduceSij (env, info), d (4)
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& ICPS<ICPSConfig::REGULAR>::get (ICPS::Memory mem)
    {
        switch (mem)
        {
            case ICPS::Memory::H_IN_DEV_M:
                return hBufferInDM;
            case ICPS::Memory::H_IN_DEV_F:
                return hBufferInDF;
            case ICPS::Memory::H_OUT:
                return hBufferOut;
            case ICPS::Memory::D_IN_DEV_M:
                return dBufferInDM;
            case ICPS::Memory::D_IN_DEV_F:
                return dBufferInDF;
            case ICPS::Memory::D_SIJ:
                return dBufferSij;
            case ICPS::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _m number of points in the sets.
     *  \param[in] _c scaling factor for dealing with floating point arithmetic issues.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void ICPS<ICPSConfig::REGULAR>::init (unsigned int _m, float _c, Staging _staging)
    {
        m = _m; c = _c;
        bufferInSize = m * sizeof (cl_float4);
        bufferOutSize = 11 * sizeof (cl_float);
        staging = _staging;

        unsigned int n = m;
        if (n % 4) n += 4 - n % 4;
        n /= 4;

        bufferSijSize = 11 * (n * sizeof (cl_float));

        try
        {
            if (m == 0)
                throw "The array cannot have zero points";
        }
        catch (const char *error)
        {
            std::cerr << "Error[ICPS<ICPSConfig::REGULAR>]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        global = cl::NDRange (n);

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInDevM = nullptr;
                hPtrInDevF = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInDM () == nullptr)
                    hBufferInDM = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);
                if (hBufferInDF () == nullptr)
                    hBufferInDF = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);

                hPtrInDevM = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInDM, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                hPtrInDevF = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInDF, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                queue.enqueueUnmapMemObject (hBufferInDM, hPtrInDevM);
                queue.enqueueUnmapMemObject (hBufferInDF, hPtrInDevF);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                hPtrOut = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io)
                {
                    hPtrInDevM = nullptr;
                    hPtrInDevF = nullptr;
                }
                break;
        }
        
        // Create device buffers
        if (dBufferInDM () == nullptr)
            dBufferInDM = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferInDF () == nullptr)
            dBufferInDF = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferSij () == nullptr)
            dBufferSij = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSijSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferInDM);
        kernel.setArg (1, dBufferInDF);
        kernel.setArg (2, dBufferSij);
        kernel.setArg (3, m);
        kernel.setArg (4, c);

        reduceSij.get (Reduce<ReduceConfig::SUM, cl_float>::Memory::D_IN) = dBufferSij;
        reduceSij.get (Reduce<ReduceConfig::SUM, cl_float>::Memory::D_OUT) = dBufferOut;
        reduceSij.init (n, 11, Staging::NONE);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void ICPS<ICPSConfig::REGULAR>::write (ICPS::Memory mem, void *ptr, bool block, 
                                           const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPS::Memory::D_IN_DEV_M:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + m * d, hPtrInDevM);
                    queue.enqueueWriteBuffer (dBufferInDM, block, 0, bufferInSize, hPtrInDevM, events, event);
                    break;
                case ICPS::Memory::D_IN_DEV_F:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + m * d, hPtrInDevF);
                    queue.enqueueWriteBuffer (dBufferInDF, block, 0, bufferInSize, hPtrInDevF, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* ICPS<ICPSConfig::REGULAR>::read (ICPS::Memory mem, bool block, 
                                           const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPS::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferOutSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the last kernel execution.
     */
    void ICPS<ICPSConfig::REGULAR>::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events);

        reduceSij.run (nullptr, event);
    }


    /*! \note The scaling factor c multiplies the points (deviations) before processing
     *        in order to deal with floating point arithmetic issues.
     *  
     *  \return The scaling factor c.
     */
    float ICPS<ICPSConfig::REGULAR>::getScaling ()
    {
        return c;
    }


    /*! \details Updates the kernel argument for the scaling factor c.
     *  \note The scaling factor c multiplies the points (deviations) before processing
     *        in order to deal with floating point arithmetic issues.
     *  
     *  \param[in] _c scaling factor.
     */
    void ICPS<ICPSConfig::REGULAR>::setScaling (float _c)
    {
        c = _c;
        kernel.setArg (4, c);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. It specifies the context, queue, etc, to be used.
     */
    ICPS<ICPSConfig::WEIGHTED>::ICPS (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "icpSijProducts_Weighted"), 
        reduceSij (env, info), d (4)
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& ICPS<ICPSConfig::WEIGHTED>::get (ICPS::Memory mem)
    {
        switch (mem)
        {
            case ICPS::Memory::H_IN_DEV_M:
                return hBufferInDM;
            case ICPS::Memory::H_IN_DEV_F:
                return hBufferInDF;
            case ICPS::Memory::H_IN_W:
                return hBufferInW;
            case ICPS::Memory::H_OUT:
                return hBufferOut;
            case ICPS::Memory::D_IN_DEV_M:
                return dBufferInDM;
            case ICPS::Memory::D_IN_DEV_F:
                return dBufferInDF;
            case ICPS::Memory::D_IN_W:
                return dBufferInW;
            case ICPS::Memory::D_SIJ:
                return dBufferSij;
            case ICPS::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _m number of points in the sets.
     *  \param[in] _c scaling factor for dealing with floating point arithmetic issues.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void ICPS<ICPSConfig::WEIGHTED>::init (unsigned int _m, float _c, Staging _staging)
    {
        m = _m; c = _c;
        bufferInFMSize = m * sizeof (cl_float4);
        bufferInWSize = m * sizeof (cl_float);
        bufferOutSize = 11 * sizeof (cl_float);
        staging = _staging;

        unsigned int n = m;
        if (n % 4) n += 4 - n % 4;
        n /= 4;

        bufferSijSize = 11 * (n * sizeof (cl_float));

        try
        {
            if (m == 0)
                throw "The array cannot have zero points";
        }
        catch (const char *error)
        {
            std::cerr << "Error[ICPS<ICPSConfig::WEIGHTED>]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        global = cl::NDRange (n);

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInDevM = nullptr;
                hPtrInDevF = nullptr;
                hPtrInW = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInDM () == nullptr)
                    hBufferInDM = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInFMSize);
                if (hBufferInDF () == nullptr)
                    hBufferInDF = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInFMSize);
                if (hBufferInW () == nullptr)
                    hBufferInW = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInWSize);

                hPtrInDevM = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInDM, CL_FALSE, CL_MAP_WRITE, 0, bufferInFMSize);
                hPtrInDevF = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInDF, CL_FALSE, CL_MAP_WRITE, 0, bufferInFMSize);
                hPtrInW = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInW, CL_FALSE, CL_MAP_WRITE, 0, bufferInWSize);
                queue.enqueueUnmapMemObject (hBufferInDM, hPtrInDevM);
                queue.enqueueUnmapMemObject (hBufferInDF, hPtrInDevF);
                queue.enqueueUnmapMemObject (hBufferInW, hPtrInW);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                hPtrOut = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io)
                {
                    hPtrInDevM = nullptr;
                    hPtrInDevF = nullptr;
                    hPtrInW = nullptr;
                }
                break;
        }
        
        // Create device buffers
        if (dBufferInDM () == nullptr)
            dBufferInDM = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInFMSize);
        if (dBufferInDF () == nullptr)
            dBufferInDF = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInFMSize);
        if (dBufferInW () == nullptr)
            dBufferInW = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInWSize);
        if (dBufferSij () == nullptr)
            dBufferSij = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSijSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferInDM);
        kernel.setArg (1, dBufferInDF);
        kernel.setArg (2, dBufferInW);
        kernel.setArg (3, dBufferSij);
        kernel.setArg (4, m);
        kernel.setArg (5, c);

        reduceSij.get (Reduce<ReduceConfig::SUM, cl_float>::Memory::D_IN) = dBufferSij;
        reduceSij.get (Reduce<ReduceConfig::SUM, cl_float>::Memory::D_OUT) = dBufferOut;
        reduceSij.init (n, 11, Staging::NONE);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void ICPS<ICPSConfig::WEIGHTED>::write (ICPS::Memory mem, void *ptr, bool block, 
                                            const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPS::Memory::D_IN_DEV_M:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + m * d, hPtrInDevM);
                    queue.enqueueWriteBuffer (dBufferInDM, block, 0, bufferInFMSize, hPtrInDevM, events, event);
                    break;
                case ICPS::Memory::D_IN_DEV_F:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + m * d, hPtrInDevF);
                    queue.enqueueWriteBuffer (dBufferInDF, block, 0, bufferInFMSize, hPtrInDevF, events, event);
                    break;
                case ICPS::Memory::D_IN_W:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + m * d, hPtrInW);
                    queue.enqueueWriteBuffer (dBufferInW, block, 0, bufferInWSize, hPtrInW, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* ICPS<ICPSConfig::WEIGHTED>::read (ICPS::Memory mem, bool block, 
                                            const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPS::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferOutSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the last kernel execution.
     */
    void ICPS<ICPSConfig::WEIGHTED>::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events);

        reduceSij.run (nullptr, event);
    }


    /*! \note The scaling factor c multiplies the points (deviations) before processing
     *        in order to deal with floating point arithmetic issues.
     *  
     *  \return The scaling factor c.
     */
    float ICPS<ICPSConfig::WEIGHTED>::getScaling ()
    {
        return c;
    }


    /*! \details Updates the kernel argument for the scaling factor c.
     *  \note The scaling factor c multiplies the points (deviations) before processing
     *        in order to deal with floating point arithmetic issues.
     *  
     *  \param[in] _c scaling factor.
     */
    void ICPS<ICPSConfig::WEIGHTED>::setScaling (float _c)
    {
        c = _c;
        kernel.setArg (4, c);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. It specifies the context, queue, etc, to be used.
     */
    ICPTransform<ICPTransformConfig::QUATERNION>::ICPTransform (
        clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "icpTransform_Quaternion"), d (8)
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& ICPTransform<ICPTransformConfig::QUATERNION>::get (ICPTransform::Memory mem)
    {
        switch (mem)
        {
            case ICPTransform::Memory::H_IN_M:
                return hBufferInM;
            case ICPTransform::Memory::H_IN_T:
                return hBufferInT;
            case ICPTransform::Memory::H_OUT:
                return hBufferOut;
            case ICPTransform::Memory::D_IN_M:
                return dBufferInM;
            case ICPTransform::Memory::D_IN_T:
                return dBufferInT;
            case ICPTransform::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _m number of points in the set.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void ICPTransform<ICPTransformConfig::QUATERNION>::init (unsigned int _m, Staging _staging)
    {
        m = _m;
        bufferInMSize = m * sizeof (cl_float8);
        bufferInTSize = 2 * sizeof (cl_float4);
        bufferOutSize = m * sizeof (cl_float8);
        staging = _staging;

        try
        {
            if (m == 0)
                throw "The set cannot have zero points";
        }
        catch (const char *error)
        {
            std::cerr << "Error[ICPTransform<ICPTransformConfig::QUATERNION>]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        global = cl::NDRange (2, m);
        
        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInM = nullptr;
                hPtrInT = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInM () == nullptr)
                    hBufferInM = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInMSize);
                if (hBufferInT () == nullptr)
                    hBufferInT = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInTSize);

                hPtrInM = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInM, CL_FALSE, CL_MAP_WRITE, 0, bufferInMSize);
                hPtrInT = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInT, CL_FALSE, CL_MAP_WRITE, 0, bufferInTSize);
                queue.enqueueUnmapMemObject (hBufferInM, hPtrInM);
                queue.enqueueUnmapMemObject (hBufferInT, hPtrInT);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                hPtrOut = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io)
                {
                    hPtrInM = nullptr;
                    hPtrInT = nullptr;
                }
                break;
        }

        // Create device buffers
        if (dBufferInM () == nullptr)
            dBufferInM = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInMSize);
        if (dBufferInT () == nullptr)
            dBufferInT = cl::Buffer (context, CL_MEM_READ_WRITE, bufferInTSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferInM);
        kernel.setArg (1, dBufferOut);
        kernel.setArg (2, dBufferInT);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void ICPTransform<ICPTransformConfig::QUATERNION>::write (
        ICPTransform::Memory mem, void *ptr, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPTransform::Memory::D_IN_M:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + m * d, hPtrInM);
                    queue.enqueueWriteBuffer (dBufferInM, block, 0, bufferInMSize, hPtrInM, events, event);
                    break;
                case ICPTransform::Memory::D_IN_T:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + 2 * 4, hPtrInT);
                    queue.enqueueWriteBuffer (dBufferInT, block, 0, bufferInTSize, hPtrInT, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* ICPTransform<ICPTransformConfig::QUATERNION>::read (
        ICPTransform::Memory mem, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPTransform::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferOutSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the kernel execution.
     */
    void ICPTransform<ICPTransformConfig::QUATERNION>::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, event);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. It specifies the context, queue, etc, to be used.
     */
    ICPTransform<ICPTransformConfig::MATRIX>::ICPTransform (
        clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "icpTransform_Matrix"), d (8)
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& ICPTransform<ICPTransformConfig::MATRIX>::get (ICPTransform::Memory mem)
    {
        switch (mem)
        {
            case ICPTransform::Memory::H_IN_M:
                return hBufferInM;
            case ICPTransform::Memory::H_IN_T:
                return hBufferInT;
            case ICPTransform::Memory::H_OUT:
                return hBufferOut;
            case ICPTransform::Memory::D_IN_M:
                return dBufferInM;
            case ICPTransform::Memory::D_IN_T:
                return dBufferInT;
            case ICPTransform::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _m number of points in the set.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void ICPTransform<ICPTransformConfig::MATRIX>::init (unsigned int _m, Staging _staging)
    {
        m = _m;
        bufferInMSize = m * sizeof (cl_float8);
        bufferInTSize = 4 * sizeof (cl_float4);
        bufferOutSize = m * sizeof (cl_float8);
        staging = _staging;

        try
        {
            if (m == 0)
                throw "The set cannot have zero points";
        }
        catch (const char *error)
        {
            std::cerr << "Error[ICPTransform<ICPTransformConfig::MATRIX>]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        global = cl::NDRange (2, m);
        
        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInM = nullptr;
                hPtrInT = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInM () == nullptr)
                    hBufferInM = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInMSize);
                if (hBufferInT () == nullptr)
                    hBufferInT = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInTSize);

                hPtrInM = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInM, CL_FALSE, CL_MAP_WRITE, 0, bufferInMSize);
                hPtrInT = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInT, CL_FALSE, CL_MAP_WRITE, 0, bufferInTSize);
                queue.enqueueUnmapMemObject (hBufferInM, hPtrInM);
                queue.enqueueUnmapMemObject (hBufferInT, hPtrInT);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                hPtrOut = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io)
                {
                    hPtrInM = nullptr;
                    hPtrInT = nullptr;
                }
                break;
        }

        // Create device buffers
        if (dBufferInM () == nullptr)
            dBufferInM = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInMSize);
        if (dBufferInT () == nullptr)
            dBufferInT = cl::Buffer (context, CL_MEM_READ_WRITE, bufferInTSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferInM);
        kernel.setArg (1, dBufferOut);
        kernel.setArg (2, dBufferInT);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void ICPTransform<ICPTransformConfig::MATRIX>::write (
        ICPTransform::Memory mem, void *ptr, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPTransform::Memory::D_IN_M:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + m * d, hPtrInM);
                    queue.enqueueWriteBuffer (dBufferInM, block, 0, bufferInMSize, hPtrInM, events, event);
                    break;
                case ICPTransform::Memory::D_IN_T:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + 16, hPtrInT);
                    queue.enqueueWriteBuffer (dBufferInT, block, 0, bufferInTSize, hPtrInT, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* ICPTransform<ICPTransformConfig::MATRIX>::read (
        ICPTransform::Memory mem, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPTransform::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferOutSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the kernel execution.
     */
    void ICPTransform<ICPTransformConfig::MATRIX>::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, event);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. It specifies the context, queue, etc, to be used.
     */
    ICPPowerMethod::ICPPowerMethod (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernel (env.getProgram (info.pgIdx), "icpPowerMethod")
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& ICPPowerMethod::get (ICPPowerMethod::Memory mem)
    {
        switch (mem)
        {
            case ICPPowerMethod::Memory::H_IN_S:
                return hBufferInS;
            case ICPPowerMethod::Memory::H_IN_MEAN:
                return hBufferInMean;
            case ICPPowerMethod::Memory::H_OUT_T_K:
                return hBufferOutTk;
            case ICPPowerMethod::Memory::D_IN_S:
                return dBufferInS;
            case ICPPowerMethod::Memory::D_IN_MEAN:
                return dBufferInMean;
            case ICPPowerMethod::Memory::D_OUT_T_K:
                return dBufferOutTk;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void ICPPowerMethod::init (Staging _staging)
    {
        bufferInSSize = 11 * sizeof (cl_float);
        bufferInMeanSize = 2 * sizeof (cl_float4);
        bufferOutTkSize = 2 * sizeof (cl_float4);
        staging = _staging;
        
        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInS = nullptr;
                hPtrInMean = nullptr;
                hPtrOutTk = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInS () == nullptr)
                    hBufferInS = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSSize);
                if (hBufferInMean () == nullptr)
                    hBufferInMean = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInMeanSize);

                hPtrInS = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInS, CL_FALSE, CL_MAP_WRITE, 0, bufferInSSize);
                hPtrInMean = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInMean, CL_FALSE, CL_MAP_WRITE, 0, bufferInMeanSize);
                queue.enqueueUnmapMemObject (hBufferInS, hPtrInS);
                queue.enqueueUnmapMemObject (hBufferInMean, hPtrInMean);

                if (!io)
                {
                    queue.finish ();
                    hPtrOutTk = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOutTk () == nullptr)
                    hBufferOutTk = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutTkSize);

                hPtrOutTk = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOutTk, CL_FALSE, CL_MAP_READ, 0, bufferOutTkSize);
                queue.enqueueUnmapMemObject (hBufferOutTk, hPtrOutTk);
                queue.finish ();

                if (!io)
                {
                    hPtrInS = nullptr;
                    hPtrInMean = nullptr;
                }
                break;
        }

        // Create device buffers
        if (dBufferInS () == nullptr)
            dBufferInS = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSSize);
        if (dBufferInMean () == nullptr)
            dBufferInMean = cl::Buffer (context, CL_MEM_READ_WRITE, bufferInMeanSize);
        if (dBufferOutTk () == nullptr)
            dBufferOutTk = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutTkSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferInS);
        kernel.setArg (1, dBufferInMean);
        kernel.setArg (2, dBufferOutTk);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void ICPPowerMethod::write (ICPPowerMethod::Memory mem, 
        void *ptr, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPPowerMethod::Memory::D_IN_S:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + 11, hPtrInS);
                    queue.enqueueWriteBuffer (dBufferInS, block, 0, bufferInSSize, hPtrInS, events, event);
                    break;
                case ICPPowerMethod::Memory::D_IN_MEAN:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + 8, hPtrInMean);
                    queue.enqueueWriteBuffer (dBufferInMean, block, 0, bufferInMeanSize, hPtrInMean, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* ICPPowerMethod::read (ICPPowerMethod::Memory mem, 
        bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPPowerMethod::Memory::H_OUT_T_K:
                    queue.enqueueReadBuffer (dBufferOutTk, block, 0, bufferOutTkSize, hPtrOutTk, events, event);
                    return hPtrOutTk;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the kernel execution.
     */
    void ICPPowerMethod::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueTask (kernel, events, event);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _infoRBC opencl configuration for the `RBC` classes. 
     *                      It specifies the context, queue, etc, to be used.
     *  \param[in] _infoICP opencl configuration for the `ICP` classes. 
     *                      It specifies the context, queue, etc, to be used.
     */
    ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::REGULAR>::ICPStep (clutils::CLEnv &_env, 
        clutils::CLEnvInfo<1> _infoRBC, clutils::CLEnvInfo<1> _infoICP) : 
        env (_env), infoRBC (_infoRBC), infoICP (_infoICP), 
        context (env.getContext (infoICP.pIdx)), 
        queue (env.getQueue (infoICP.ctxIdx, infoICP.qIdx[0])), 
        fReps (env, infoICP), rbcC (env, infoRBC), 
        transform (env, infoICP), rbcS (env, infoRBC), 
        means (env, infoICP), devs (env, infoICP), 
        matrixS (env, infoICP), d (8)
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::REGULAR>::get (ICPStep::Memory mem)
    {
        switch (mem)
        {
            case ICPStep::Memory::H_IN_F:
                return hBufferInF;
            case ICPStep::Memory::H_IN_M:
                return hBufferInM;
            case ICPStep::Memory::H_IO_T:
                return hBufferIOT;
            case ICPStep::Memory::D_IN_F:
                return dBufferInF;
            case ICPStep::Memory::D_IN_M:
                return dBufferInM;
            case ICPStep::Memory::D_IO_T:
                return dBufferIOT;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _m number of points in the sets.
     *  \param[in] _nr number of fixed set representatives.
     *  \param[in] _a factor scaling the results of the distance calculations for the 
     *                geometric \f$ x_g \f$ and photometric \f$ x_p \f$ dimensions of 
     *                the \f$ x\epsilon\mathbb{R}^8 \f$ points. That is, \f$ \|x-x'\|_2^2= 
     *                f_g(a)\|x_g-x'_g\|_2^2+f_p(a)\|x_p-x'_p\|_2^2 \f$. For more info, 
     *                look at `euclideanSquaredMetric8` in [kernels/rbc_kernels.cl]
     *                (https://random-ball-cover.nlamprian.me).
     *  \param[in] _c scaling factor for dealing with floating point arithmetic 
     *                issues when computing the `S` matrix.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::REGULAR>::init (
        unsigned int _m, unsigned int _nr, float _a, float _c, Staging _staging)
    {
        m = _m; nr = _nr; a = _a; c = _c;
        bufferFMSize = m * sizeof (cl_float8);
        bufferTSize = 2 * sizeof (cl_float4);
        staging = _staging;

        try
        {
            if (m == 0)
                throw "The sets of landmarks cannot have zero points";

            if (nr == 0)
                throw "The sets of representatives cannot have zero points";

            if (a == 0.f)
                throw "The alpha parameter cannot be equal to zero";
        }
        catch (const char *error)
        {
            std::cerr << "Error[ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::REGULAR>]: " 
                      << error << std::endl;
            exit (EXIT_FAILURE);
        }
        
        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInF = nullptr;
                hPtrInM = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInF () == nullptr)
                    hBufferInF = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferFMSize);
                if (hBufferInM () == nullptr)
                    hBufferInM = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferFMSize);

                hPtrInF = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInF, CL_FALSE, CL_MAP_WRITE, 0, bufferFMSize);
                hPtrInM = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInM, CL_FALSE, CL_MAP_WRITE, 0, bufferFMSize);
                queue.enqueueUnmapMemObject (hBufferInF, hPtrInF);
                queue.enqueueUnmapMemObject (hBufferInM, hPtrInM);

                if (!io)
                {
                    queue.finish ();
                    break;
                }

            case Staging::O:
                if (!io)
                {
                    hPtrInF = nullptr;
                    hPtrInM = nullptr;
                }
                break;
        }

        if (hBufferIOT () == nullptr)
            hBufferIOT = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferTSize);

        hPtrIOT = (cl_float *) queue.enqueueMapBuffer (
            hBufferIOT, CL_FALSE, CL_MAP_READ, 0, bufferTSize);
        queue.enqueueUnmapMemObject (hBufferIOT, hPtrIOT);
        queue.finish ();

        // Create device buffers
        if (dBufferInF () == nullptr)
            dBufferInF = cl::Buffer (context, CL_MEM_READ_ONLY, bufferFMSize);
        if (dBufferInM () == nullptr)
            dBufferInM = cl::Buffer (context, CL_MEM_READ_ONLY, bufferFMSize);
        if (dBufferIOT () == nullptr)
            dBufferIOT = cl::Buffer (context, CL_MEM_READ_WRITE, bufferTSize);

        // Load initial identity transformation
        cl_float T0[8] = { 0, 0, 0, 1, 0, 0, 0, 1 };
        std::copy (T0, T0 + 8, hPtrIOT);
        queue.enqueueWriteBuffer (dBufferIOT, CL_FALSE, 0, bufferTSize, hPtrIOT);

        R = Eigen::Matrix3f::Identity ();
        q = Eigen::Quaternionf (R);
        t.setZero ();
        s = 1.f;

        // Configure classes

        // Classes in the initialization step ==================================
        
        fReps.get (ICPReps::Memory::D_IN) = dBufferInF;
        fReps.get (ICPReps::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, nr * sizeof (cl_float8));
        fReps.init (nr, Staging::NONE);

        const RBC::KernelTypeC K1 = RBC::KernelTypeC::KINECT_R;
        const RBC::RBCPermuteConfig P1 = RBC::RBCPermuteConfig::GENERIC;

        rbcC.get (RBC::RBCConstruct<K1, P1>::Memory::D_IN_X) = dBufferInF;
        rbcC.get (RBC::RBCConstruct<K1, P1>::Memory::D_IN_R) = fReps.get (ICPReps::Memory::D_OUT);
        rbcC.init (m, nr, d, a, 0, RBC::Staging::NONE);

        // Classes in the iteration step =======================================

        const ICPTransformConfig TC = ICPTransformConfig::QUATERNION;

        transform.get (ICPTransform<TC>::Memory::D_IN_M) = dBufferInM;
        transform.get (ICPTransform<TC>::Memory::D_IN_T) = dBufferIOT;
        transform.get (ICPTransform<TC>::Memory::D_OUT) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, m * sizeof (cl_float8));
        transform.init (m, Staging::NONE);

        const RBC::KernelTypeC K2 = RBC::KernelTypeC::KINECT_R;
        const RBC::RBCPermuteConfig P2 = RBC::RBCPermuteConfig::GENERIC;
        const RBC::KernelTypeS S2 = RBC::KernelTypeS::KINECT;

        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_Q) = 
            transform.get (ICPTransform<TC>::Memory::D_OUT);
        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_R) = 
            fReps.get (ICPReps::Memory::D_OUT);
        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_X_P) = 
            rbcC.get (RBC::RBCConstruct<K1, P1>::Memory::D_OUT_X_P);
        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_O) = 
            rbcC.get (RBC::RBCConstruct<K1, P1>::Memory::D_OUT_O);
        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_N) = 
            rbcC.get (RBC::RBCConstruct<K1, P1>::Memory::D_OUT_N);
        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_NN) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, m * sizeof (cl_float8));
        rbcS.init (m, nr, m, a, RBC::Staging::NONE);

        const ICPMeanConfig MC = ICPMeanConfig::REGULAR;

        means.get (ICPMean<MC>::Memory::D_IN_F) = 
            rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_NN);
        means.get (ICPMean<MC>::Memory::D_IN_M) = 
            rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_Q_P);
        means.get (ICPMean<MC>::Memory::D_OUT) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, 2 * sizeof (cl_float4));
        means.init (m, Staging::O);

        devs.get (ICPDevs::Memory::D_IN_F) = rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_NN);
        devs.get (ICPDevs::Memory::D_IN_M) = rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_Q_P);
        devs.get (ICPDevs::Memory::D_IN_MEAN) = means.get (ICPMean<MC>::Memory::D_OUT);
        devs.get (ICPDevs::Memory::D_OUT_DEV_F) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, m * sizeof (cl_float4));
        devs.get (ICPDevs::Memory::D_OUT_DEV_M) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, m * sizeof (cl_float4));
        devs.init (m, Staging::NONE);

        const ICPSConfig SC = ICPSConfig::REGULAR;

        matrixS.get (ICPS<SC>::Memory::D_IN_DEV_M) = devs.get (ICPDevs::Memory::D_OUT_DEV_M);
        matrixS.get (ICPS<SC>::Memory::D_IN_DEV_F) = devs.get (ICPDevs::Memory::D_OUT_DEV_F);
        matrixS.init (m, c, Staging::O);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::REGULAR>::write (
        ICPStep::Memory mem, void *ptr, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPStep::Memory::D_IN_F:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + m * d, hPtrInF);
                    queue.enqueueWriteBuffer (dBufferInF, block, 0, bufferFMSize, hPtrInF, events, event);
                    break;
                case ICPStep::Memory::D_IN_M:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + m * d, hPtrInM);
                    queue.enqueueWriteBuffer (dBufferInM, block, 0, bufferFMSize, hPtrInM, events, event);
                    break;
                case ICPStep::Memory::D_IO_T:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + 8, hPtrIOT);
                    queue.enqueueWriteBuffer (dBufferIOT, block, 0, bufferTSize, hPtrIOT, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::REGULAR>::read (
        ICPStep::Memory mem, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPStep::Memory::H_IO_T:
                    queue.enqueueReadBuffer (dBufferIOT, block, 0, bufferTSize, hPtrIOT, events, event);
                    return hPtrIOT;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \note Call `buildRBC` after the `D_IN_F` buffer has been written, 
     *        and before any calls to `run` (for each registration).
     */
    void ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::REGULAR>::buildRBC (
        const std::vector<cl::Event> *events, cl::Event *event)
    {
        fReps.run (events);
        rbcC.run (nullptr, event);
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the kernel execution.
     *  \param[in] config flag. If true, configures the `RBC search` process. 
     *                    Set to true once, when the `RBC` data structure is reset.
     */
    void ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::REGULAR>::run (
        const std::vector<cl::Event> *events, cl::Event *event, bool config)
    {
        transform.run ();
        rbcS.run (nullptr, nullptr, config);
        means.run ();
        devs.run ();
        matrixS.run ();

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

        queue.enqueueWriteBuffer (dBufferIOT, CL_FALSE, 0, bufferTSize, hPtrIOT, nullptr, event);
    }


    /*! \return The scaling parameter \f$ \alpha \f$. */
    float ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::REGULAR>::getAlpha ()
    {
        return a;
    }


    /*! \details Updates the kernel arguments for the scaling parameter \f$ \alpha \f$.
     *
     *  \param[in] _a scaling parameter \f$ \alpha \f$.
     */
    void ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::REGULAR>::setAlpha (float _a)
    {
        a = _a;
        rbcC.setAlpha (a);
        rbcS.setAlpha (a);
    }


    /*! \note The scaling factor c multiplies the points (deviations) before processing
     *        in order to deal with floating point arithmetic issues.
     *  
     *  \return The scaling factor c.
     */
    float ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::REGULAR>::getScaling ()
    {
        return c;
    }


    /*! \details Updates the kernel argument for the scaling factor c.
     *  \note The scaling factor c multiplies the points (deviations) before processing
     *        in order to deal with floating point arithmetic issues.
     *  
     *  \param[in] _c scaling factor.
     */
    void ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::REGULAR>::setScaling (float _c)
    {
        c = _c;
        matrixS.setScaling (c);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _infoRBC opencl configuration for the `RBC` classes. 
     *                      It specifies the context, queue, etc, to be used.
     *  \param[in] _infoICP opencl configuration for the `ICP` classes. 
     *                      It specifies the context, queue, etc, to be used.
     */
    ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::WEIGHTED>::ICPStep (clutils::CLEnv &_env, 
        clutils::CLEnvInfo<1> _infoRBC, clutils::CLEnvInfo<1> _infoICP) : 
        env (_env), infoRBC (_infoRBC), infoICP (_infoICP), 
        context (env.getContext (infoICP.pIdx)), 
        queue (env.getQueue (infoICP.ctxIdx, infoICP.qIdx[0])), 
        fReps (env, infoICP), rbcC (env, infoRBC), 
        transform (env, infoICP), rbcS (env, infoRBC), weights (env, infoICP), 
        means (env, infoICP), devs (env, infoICP), matrixS (env, infoICP), d (8)
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::WEIGHTED>::get (ICPStep::Memory mem)
    {
        switch (mem)
        {
            case ICPStep::Memory::H_IN_F:
                return hBufferInF;
            case ICPStep::Memory::H_IN_M:
                return hBufferInM;
            case ICPStep::Memory::H_IO_T:
                return hBufferIOT;
            case ICPStep::Memory::D_IN_F:
                return dBufferInF;
            case ICPStep::Memory::D_IN_M:
                return dBufferInM;
            case ICPStep::Memory::D_IO_T:
                return dBufferIOT;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _m number of points in the sets.
     *  \param[in] _nr number of fixed set representatives.
     *  \param[in] _a factor scaling the results of the distance calculations for the 
     *                geometric \f$ x_g \f$ and photometric \f$ x_p \f$ dimensions of 
     *                the \f$ x\epsilon\mathbb{R}^8 \f$ points. That is, \f$ \|x-x'\|_2^2= 
     *                f_g(a)\|x_g-x'_g\|_2^2+f_p(a)\|x_p-x'_p\|_2^2 \f$. For more info, 
     *                look at `euclideanSquaredMetric8` in [kernels/rbc_kernels.cl]
     *                (https://random-ball-cover.nlamprian.me).
     *  \param[in] _c scaling factor for dealing with floating point arithmetic 
     *                issues when computing the `S` matrix.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::WEIGHTED>::init (
        unsigned int _m, unsigned int _nr, float _a, float _c, Staging _staging)
    {
        m = _m; nr = _nr; a = _a; c = _c;
        bufferFMSize = m * sizeof (cl_float8);
        bufferTSize = 2 * sizeof (cl_float4);
        staging = _staging;

        try
        {
            if (m == 0)
                throw "The sets of landmarks cannot have zero points";

            if (nr == 0)
                throw "The sets of representatives cannot have zero points";

            if (a == 0.f)
                throw "The alpha parameter cannot be equal to zero";
        }
        catch (const char *error)
        {
            std::cerr << "Error[ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::WEIGHTED>]: " 
                      << error << std::endl;
            exit (EXIT_FAILURE);
        }
        
        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInF = nullptr;
                hPtrInM = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInF () == nullptr)
                    hBufferInF = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferFMSize);
                if (hBufferInM () == nullptr)
                    hBufferInM = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferFMSize);

                hPtrInF = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInF, CL_FALSE, CL_MAP_WRITE, 0, bufferFMSize);
                hPtrInM = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInM, CL_FALSE, CL_MAP_WRITE, 0, bufferFMSize);
                queue.enqueueUnmapMemObject (hBufferInF, hPtrInF);
                queue.enqueueUnmapMemObject (hBufferInM, hPtrInM);

                if (!io)
                {
                    queue.finish ();
                    break;
                }

            case Staging::O:
                if (!io)
                {
                    hPtrInF = nullptr;
                    hPtrInM = nullptr;
                }
                break;
        }

        if (hBufferIOT () == nullptr)
            hBufferIOT = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferTSize);

        hPtrIOT = (cl_float *) queue.enqueueMapBuffer (
            hBufferIOT, CL_FALSE, CL_MAP_READ, 0, bufferTSize);
        queue.enqueueUnmapMemObject (hBufferIOT, hPtrIOT);
        queue.finish ();

        // Create device buffers
        if (dBufferInF () == nullptr)
            dBufferInF = cl::Buffer (context, CL_MEM_READ_ONLY, bufferFMSize);
        if (dBufferInM () == nullptr)
            dBufferInM = cl::Buffer (context, CL_MEM_READ_ONLY, bufferFMSize);
        if (dBufferIOT () == nullptr)
            dBufferIOT = cl::Buffer (context, CL_MEM_READ_WRITE, bufferTSize);

        // Load initial identity transformation
        cl_float T0[8] = { 0, 0, 0, 1, 0, 0, 0, 1 };
        std::copy (T0, T0 + 8, hPtrIOT);
        queue.enqueueWriteBuffer (dBufferIOT, CL_FALSE, 0, bufferTSize, hPtrIOT);

        R = Eigen::Matrix3f::Identity ();
        q = Eigen::Quaternionf (R);
        t.setZero ();
        s = 1.f;

        // Configure classes

        // Classes in the initialization step ==================================
        
        fReps.get (ICPReps::Memory::D_IN) = dBufferInF;
        fReps.get (ICPReps::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, nr * sizeof (cl_float8));
        fReps.init (nr, Staging::NONE);

        const RBC::KernelTypeC K1 = RBC::KernelTypeC::KINECT_R;
        const RBC::RBCPermuteConfig P1 = RBC::RBCPermuteConfig::GENERIC;

        rbcC.get (RBC::RBCConstruct<K1, P1>::Memory::D_IN_X) = dBufferInF;
        rbcC.get (RBC::RBCConstruct<K1, P1>::Memory::D_IN_R) = fReps.get (ICPReps::Memory::D_OUT);
        rbcC.init (m, nr, d, a, 0, RBC::Staging::NONE);

        // Classes in the iteration step =======================================

        const ICPTransformConfig TC = ICPTransformConfig::QUATERNION;

        transform.get (ICPTransform<TC>::Memory::D_IN_M) = dBufferInM;
        transform.get (ICPTransform<TC>::Memory::D_IN_T) = dBufferIOT;
        transform.get (ICPTransform<TC>::Memory::D_OUT) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, m * sizeof (cl_float8));
        transform.init (m, Staging::NONE);

        const RBC::KernelTypeC K2 = RBC::KernelTypeC::KINECT_R;
        const RBC::RBCPermuteConfig P2 = RBC::RBCPermuteConfig::GENERIC;
        const RBC::KernelTypeS S2 = RBC::KernelTypeS::KINECT;

        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_Q) = 
            transform.get (ICPTransform<TC>::Memory::D_OUT);
        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_R) = 
            fReps.get (ICPReps::Memory::D_OUT);
        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_X_P) = 
            rbcC.get (RBC::RBCConstruct<K1, P1>::Memory::D_OUT_X_P);
        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_O) = 
            rbcC.get (RBC::RBCConstruct<K1, P1>::Memory::D_OUT_O);
        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_N) = 
            rbcC.get (RBC::RBCConstruct<K1, P1>::Memory::D_OUT_N);
        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_NN) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, m * sizeof (cl_float8));
        rbcS.init (m, nr, m, a, RBC::Staging::NONE);

        weights.get (ICPWeights::Memory::D_IN) = 
            rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_NN_ID);
        weights.get (ICPWeights::Memory::D_OUT_W) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, m * sizeof (cl_float));
        weights.get (ICPWeights::Memory::D_OUT_SUM_W) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, sizeof (cl_double));
        weights.init (m, Staging::NONE);

        const ICPMeanConfig MC = ICPMeanConfig::WEIGHTED;

        means.get (ICPMean<MC>::Memory::D_IN_F) = 
            rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_NN);
        means.get (ICPMean<MC>::Memory::D_IN_M) = 
            rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_Q_P);
        means.get (ICPMean<MC>::Memory::D_IN_W) = 
            weights.get (ICPWeights::Memory::D_OUT_W);
        means.get (ICPMean<MC>::Memory::D_IN_SUM_W) = 
            weights.get (ICPWeights::Memory::D_OUT_SUM_W);
        means.get (ICPMean<MC>::Memory::D_OUT) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, 2 * sizeof (cl_float4));
        means.init (m, Staging::O);

        devs.get (ICPDevs::Memory::D_IN_F) = rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_NN);
        devs.get (ICPDevs::Memory::D_IN_M) = rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_Q_P);
        devs.get (ICPDevs::Memory::D_IN_MEAN) = means.get (ICPMean<MC>::Memory::D_OUT);
        devs.get (ICPDevs::Memory::D_OUT_DEV_F) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, m * sizeof (cl_float4));
        devs.get (ICPDevs::Memory::D_OUT_DEV_M) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, m * sizeof (cl_float4));
        devs.init (m, Staging::NONE);

        const ICPSConfig SC = ICPSConfig::WEIGHTED;

        matrixS.get (ICPS<SC>::Memory::D_IN_DEV_M) = devs.get (ICPDevs::Memory::D_OUT_DEV_M);
        matrixS.get (ICPS<SC>::Memory::D_IN_DEV_F) = devs.get (ICPDevs::Memory::D_OUT_DEV_F);
        matrixS.get (ICPS<SC>::Memory::D_IN_W) = weights.get (ICPWeights::Memory::D_OUT_W);
        matrixS.init (m, c, Staging::O);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::WEIGHTED>::write (
        ICPStep::Memory mem, void *ptr, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPStep::Memory::D_IN_F:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + m * d, hPtrInF);
                    queue.enqueueWriteBuffer (dBufferInF, block, 0, bufferFMSize, hPtrInF, events, event);
                    break;
                case ICPStep::Memory::D_IN_M:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + m * d, hPtrInM);
                    queue.enqueueWriteBuffer (dBufferInM, block, 0, bufferFMSize, hPtrInM, events, event);
                    break;
                case ICPStep::Memory::D_IO_T:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + 8, hPtrIOT);
                    queue.enqueueWriteBuffer (dBufferIOT, block, 0, bufferTSize, hPtrIOT, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::WEIGHTED>::read (
        ICPStep::Memory mem, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPStep::Memory::H_IO_T:
                    queue.enqueueReadBuffer (dBufferIOT, block, 0, bufferTSize, hPtrIOT, events, event);
                    return hPtrIOT;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \note Call `buildRBC` after the `D_IN_F` buffer has been written, 
     *        and before any calls to `run` (for each registration).
     */
    void ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::WEIGHTED>::buildRBC (
        const std::vector<cl::Event> *events, cl::Event *event)
    {
        fReps.run (events);
        rbcC.run (nullptr, event);
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the kernel execution.
     *  \param[in] config flag. If true, configures the `RBC search` process. 
     *                    Set to true once, when the `RBC` data structure is reset.
     */
    void ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::WEIGHTED>::run (
        const std::vector<cl::Event> *events, cl::Event *event, bool config)
    {
        transform.run ();
        rbcS.run (nullptr, nullptr, config);
        weights.run ();
        means.run ();
        devs.run ();
        matrixS.run ();

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

        queue.enqueueWriteBuffer (dBufferIOT, CL_FALSE, 0, bufferTSize, hPtrIOT, nullptr, event);
    }


    /*! \return The scaling parameter \f$ \alpha \f$. */
    float ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::WEIGHTED>::getAlpha ()
    {
        return a;
    }


    /*! \details Updates the kernel arguments for the scaling parameter \f$ \alpha \f$.
     *
     *  \param[in] _a scaling parameter \f$ \alpha \f$.
     */
    void ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::WEIGHTED>::setAlpha (float _a)
    {
        a = _a;
        rbcC.setAlpha (a);
        rbcS.setAlpha (a);
    }


    /*! \note The scaling factor c multiplies the points (deviations) before processing
     *        in order to deal with floating point arithmetic issues.
     *  
     *  \return The scaling factor c.
     */
    float ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::WEIGHTED>::getScaling ()
    {
        return c;
    }


    /*! \details Updates the kernel argument for the scaling factor c.
     *  \note The scaling factor c multiplies the points (deviations) before processing
     *        in order to deal with floating point arithmetic issues.
     *  
     *  \param[in] _c scaling factor.
     */
    void ICPStep<ICPStepConfigT::EIGEN, ICPStepConfigW::WEIGHTED>::setScaling (float _c)
    {
        c = _c;
        matrixS.setScaling (c);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _infoRBC opencl configuration for the `RBC` classes. 
     *                      It specifies the context, queue, etc, to be used.
     *  \param[in] _infoICP opencl configuration for the `ICP` classes. 
     *                      It specifies the context, queue, etc, to be used.
     */
    ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::REGULAR>::ICPStep (
        clutils::CLEnv &_env, clutils::CLEnvInfo<1> _infoRBC, clutils::CLEnvInfo<1> _infoICP) : 
        env (_env), infoRBC (_infoRBC), infoICP (_infoICP), 
        context (env.getContext (infoICP.pIdx)), 
        queue (env.getQueue (infoICP.ctxIdx, infoICP.qIdx[0])), 
        fReps (env, infoICP), rbcC (env, infoRBC), 
        transform (env, infoICP), rbcS (env, infoRBC), means (env, infoICP), 
        devs (env, infoICP), matrixS (env, infoICP), powMethod (env, infoICP), d (8)
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::REGULAR>::get (ICPStep::Memory mem)
    {
        switch (mem)
        {
            case ICPStep::Memory::H_IN_F:
                return hBufferInF;
            case ICPStep::Memory::H_IN_M:
                return hBufferInM;
            case ICPStep::Memory::H_IO_T:
                return hBufferIOT;
            case ICPStep::Memory::D_IN_F:
                return dBufferInF;
            case ICPStep::Memory::D_IN_M:
                return dBufferInM;
            case ICPStep::Memory::D_IO_T:
                return dBufferIOT;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _m number of points in the sets.
     *  \param[in] _nr number of fixed set representatives.
     *  \param[in] _a factor scaling the results of the distance calculations for the 
     *                geometric \f$ x_g \f$ and photometric \f$ x_p \f$ dimensions of 
     *                the \f$ x\epsilon\mathbb{R}^8 \f$ points. That is, \f$ \|x-x'\|_2^2= 
     *                f_g(a)\|x_g-x'_g\|_2^2+f_p(a)\|x_p-x'_p\|_2^2 \f$. For more info, 
     *                look at `euclideanSquaredMetric8` in [kernels/rbc_kernels.cl]
     *                (https://random-ball-cover.nlamprian.me).
     *  \param[in] _c scaling factor for dealing with floating point arithmetic 
     *                issues when computing the `S` matrix.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::REGULAR>::init (
        unsigned int _m, unsigned int _nr, float _a, float _c, Staging _staging)
    {
        m = _m; nr = _nr; a = _a; c = _c;
        bufferFMSize = m * sizeof (cl_float8);
        bufferTSize = 2 * sizeof (cl_float4);
        staging = _staging;

        try
        {
            if (m == 0)
                throw "The sets of landmarks cannot have zero points";

            if (nr == 0)
                throw "The sets of representatives cannot have zero points";

            if (a == 0.f)
                throw "The alpha parameter cannot be equal to zero";
        }
        catch (const char *error)
        {
            std::cerr << "Error[ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::REGULAR>]: " 
                      << error << std::endl;
            exit (EXIT_FAILURE);
        }
        
        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInF = nullptr;
                hPtrInM = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInF () == nullptr)
                    hBufferInF = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferFMSize);
                if (hBufferInM () == nullptr)
                    hBufferInM = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferFMSize);

                hPtrInF = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInF, CL_FALSE, CL_MAP_WRITE, 0, bufferFMSize);
                hPtrInM = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInM, CL_FALSE, CL_MAP_WRITE, 0, bufferFMSize);
                queue.enqueueUnmapMemObject (hBufferInF, hPtrInF);
                queue.enqueueUnmapMemObject (hBufferInM, hPtrInM);

                if (!io)
                {
                    queue.finish ();
                    break;
                }

            case Staging::O:
                if (!io)
                {
                    hPtrInF = nullptr;
                    hPtrInM = nullptr;
                }
                break;
        }

        if (hBufferIOT () == nullptr)
            hBufferIOT = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferTSize);

        hPtrIOT = (cl_float *) queue.enqueueMapBuffer (
            hBufferIOT, CL_FALSE, CL_MAP_READ, 0, bufferTSize);
        queue.enqueueUnmapMemObject (hBufferIOT, hPtrIOT);
        queue.finish ();

        // Create device buffers
        if (dBufferInF () == nullptr)
            dBufferInF = cl::Buffer (context, CL_MEM_READ_ONLY, bufferFMSize);
        if (dBufferInM () == nullptr)
            dBufferInM = cl::Buffer (context, CL_MEM_READ_ONLY, bufferFMSize);
        if (dBufferIOT () == nullptr)
            dBufferIOT = cl::Buffer (context, CL_MEM_READ_WRITE, bufferTSize);

        // Load initial identity transformation
        cl_float T0[8] = { 0, 0, 0, 1, 0, 0, 0, 1 };
        std::copy (T0, T0 + 8, hPtrIOT);
        queue.enqueueWriteBuffer (dBufferIOT, CL_FALSE, 0, bufferTSize, hPtrIOT);

        R = Eigen::Matrix3f::Identity ();
        q = Eigen::Quaternionf (R);
        t.setZero ();
        s = 1.f;

        // Configure classes

        // Classes in the initialization step ==================================
        
        fReps.get (ICPReps::Memory::D_IN) = dBufferInF;
        fReps.get (ICPReps::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, nr * sizeof (cl_float8));
        fReps.init (nr, Staging::NONE);

        const RBC::KernelTypeC K1 = RBC::KernelTypeC::KINECT_R;
        const RBC::RBCPermuteConfig P1 = RBC::RBCPermuteConfig::GENERIC;

        rbcC.get (RBC::RBCConstruct<K1, P1>::Memory::D_IN_X) = dBufferInF;
        rbcC.get (RBC::RBCConstruct<K1, P1>::Memory::D_IN_R) = fReps.get (ICPReps::Memory::D_OUT);
        rbcC.init (m, nr, d, a, 0, RBC::Staging::NONE);

        // Classes in the iteration step =======================================

        const ICPTransformConfig TC = ICPTransformConfig::QUATERNION;

        transform.get (ICPTransform<TC>::Memory::D_IN_M) = dBufferInM;
        transform.get (ICPTransform<TC>::Memory::D_IN_T) = dBufferIOT;
        transform.get (ICPTransform<TC>::Memory::D_OUT) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, m * sizeof (cl_float8));
        transform.init (m, Staging::NONE);

        const RBC::KernelTypeC K2 = RBC::KernelTypeC::KINECT_R;
        const RBC::RBCPermuteConfig P2 = RBC::RBCPermuteConfig::GENERIC;
        const RBC::KernelTypeS S2 = RBC::KernelTypeS::KINECT;

        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_Q) = 
            transform.get (ICPTransform<TC>::Memory::D_OUT);
        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_R) = 
            fReps.get (ICPReps::Memory::D_OUT);
        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_X_P) = 
            rbcC.get (RBC::RBCConstruct<K1, P1>::Memory::D_OUT_X_P);
        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_O) = 
            rbcC.get (RBC::RBCConstruct<K1, P1>::Memory::D_OUT_O);
        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_N) = 
            rbcC.get (RBC::RBCConstruct<K1, P1>::Memory::D_OUT_N);
        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_NN) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, m * sizeof (cl_float8));
        rbcS.init (m, nr, m, a, RBC::Staging::NONE);

        const ICPMeanConfig MC = ICPMeanConfig::REGULAR;

        means.get (ICPMean<MC>::Memory::D_IN_F) = 
            rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_NN);
        means.get (ICPMean<MC>::Memory::D_IN_M) = 
            rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_Q_P);
        means.get (ICPMean<MC>::Memory::D_OUT) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, 2 * sizeof (cl_float4));
        means.init (m, Staging::O);

        devs.get (ICPDevs::Memory::D_IN_F) = rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_NN);
        devs.get (ICPDevs::Memory::D_IN_M) = rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_Q_P);
        devs.get (ICPDevs::Memory::D_IN_MEAN) = means.get (ICPMean<MC>::Memory::D_OUT);
        devs.get (ICPDevs::Memory::D_OUT_DEV_F) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, m * sizeof (cl_float4));
        devs.get (ICPDevs::Memory::D_OUT_DEV_M) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, m * sizeof (cl_float4));
        devs.init (m, Staging::NONE);

        const ICPSConfig SC = ICPSConfig::REGULAR;

        matrixS.get (ICPS<SC>::Memory::D_IN_DEV_M) = devs.get (ICPDevs::Memory::D_OUT_DEV_M);
        matrixS.get (ICPS<SC>::Memory::D_IN_DEV_F) = devs.get (ICPDevs::Memory::D_OUT_DEV_F);
        matrixS.get (ICPS<SC>::Memory::D_OUT) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, 11 * sizeof (cl_float));
        matrixS.init (m, c, Staging::NONE);

        powMethod.get (ICPPowerMethod::Memory::D_IN_S) = matrixS.get (ICPS<SC>::Memory::D_OUT);
        powMethod.get (ICPPowerMethod::Memory::D_IN_MEAN) = means.get (ICPMean<MC>::Memory::D_OUT);
        powMethod.get (ICPPowerMethod::Memory::D_OUT_T_K) = dBufferIOT;
        powMethod.init (Staging::O);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::REGULAR>::write (
        ICPStep::Memory mem, void *ptr, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPStep::Memory::D_IN_F:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + m * d, hPtrInF);
                    queue.enqueueWriteBuffer (dBufferInF, block, 0, bufferFMSize, hPtrInF, events, event);
                    break;
                case ICPStep::Memory::D_IN_M:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + m * d, hPtrInM);
                    queue.enqueueWriteBuffer (dBufferInM, block, 0, bufferFMSize, hPtrInM, events, event);
                    break;
                case ICPStep::Memory::D_IO_T:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + 8, hPtrIOT);
                    queue.enqueueWriteBuffer (dBufferIOT, block, 0, bufferTSize, hPtrIOT, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::REGULAR>::read (
        ICPStep::Memory mem, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPStep::Memory::H_IO_T:
                    queue.enqueueReadBuffer (dBufferIOT, block, 0, bufferTSize, hPtrIOT, events, event);
                    return hPtrIOT;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \note Call `buildRBC` after the `D_IN_F` buffer has been written, 
     *        and before any calls to `run` (for each registration).
     */
    void ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::REGULAR>::buildRBC (
        const std::vector<cl::Event> *events, cl::Event *event)
    {
        fReps.run (events);
        rbcC.run (nullptr, event);
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the kernel execution.
     *  \param[in] config flag. If true, configures the `RBC search` process. 
     *                    Set to true once, when the `RBC` data structure is reset.
     */
    void ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::REGULAR>::run (
        const std::vector<cl::Event> *events, cl::Event *event, bool config)
    {
        transform.run ();
        rbcS.run (nullptr, nullptr, config);
        means.run ();
        devs.run ();
        matrixS.run ();
        powMethod.run ();

        Tk = (cl_float *) powMethod.read (ICPPowerMethod::Memory::H_OUT_T_K);

        qk = Eigen::Quaternionf (Tk);
        Rk = qk.toRotationMatrix ();
        tk = Eigen::Map<Eigen::Vector3f> (Tk + 4, 3);
        sk = Tk[7];

        R = Rk * R;
        q = Eigen::Quaternionf (R);
        t = sk * Rk * t + tk;
        s = sk * s;
        
        Eigen::Map<Eigen::Vector4f> (hPtrIOT, 4) = q.coeffs ();  // Quaternion
        Eigen::Map<Eigen::Vector4f> (hPtrIOT + 4, 4) = t.homogeneous ();  // Translation
        hPtrIOT[7] = s;  // Scale

        queue.enqueueWriteBuffer (dBufferIOT, CL_FALSE, 0, bufferTSize, hPtrIOT, nullptr, event);
    }


    /*! \return The scaling parameter \f$ \alpha \f$. */
    float ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::REGULAR>::getAlpha ()
    {
        return a;
    }


    /*! \details Updates the kernel arguments for the scaling parameter \f$ \alpha \f$.
     *
     *  \param[in] _a scaling parameter \f$ \alpha \f$.
     */
    void ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::REGULAR>::setAlpha (float _a)
    {
        a = _a;
        rbcC.setAlpha (a);
        rbcS.setAlpha (a);
    }


    /*! \note The scaling factor c multiplies the points (deviations) before processing
     *        in order to deal with floating point arithmetic issues.
     *  
     *  \return The scaling factor c.
     */
    float ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::REGULAR>::getScaling ()
    {
        return c;
    }


    /*! \details Updates the kernel argument for the scaling factor c.
     *  \note The scaling factor c multiplies the points (deviations) before processing
     *        in order to deal with floating point arithmetic issues.
     *  
     *  \param[in] _c scaling factor.
     */
    void ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::REGULAR>::setScaling (float _c)
    {
        c = _c;
        matrixS.setScaling (c);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _infoRBC opencl configuration for the `RBC` classes. 
     *                      It specifies the context, queue, etc, to be used.
     *  \param[in] _infoICP opencl configuration for the `ICP` classes. 
     *                      It specifies the context, queue, etc, to be used.
     */
    ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::WEIGHTED>::ICPStep (
        clutils::CLEnv &_env, clutils::CLEnvInfo<1> _infoRBC, clutils::CLEnvInfo<1> _infoICP) : 
        env (_env), infoRBC (_infoRBC), infoICP (_infoICP), 
        context (env.getContext (infoICP.pIdx)), 
        queue (env.getQueue (infoICP.ctxIdx, infoICP.qIdx[0])), 
        fReps (env, infoICP), rbcC (env, infoRBC), 
        transform (env, infoICP), rbcS (env, infoRBC), weights (env, infoICP), 
        means (env, infoICP), devs (env, infoICP), matrixS (env, infoICP), 
        powMethod (env, infoICP), d (8)
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::WEIGHTED>::get (ICPStep::Memory mem)
    {
        switch (mem)
        {
            case ICPStep::Memory::H_IN_F:
                return hBufferInF;
            case ICPStep::Memory::H_IN_M:
                return hBufferInM;
            case ICPStep::Memory::H_IO_T:
                return hBufferIOT;
            case ICPStep::Memory::D_IN_F:
                return dBufferInF;
            case ICPStep::Memory::D_IN_M:
                return dBufferInM;
            case ICPStep::Memory::D_IO_T:
                return dBufferIOT;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _m number of points in the sets.
     *  \param[in] _nr number of fixed set representatives.
     *  \param[in] _a factor scaling the results of the distance calculations for the 
     *                geometric \f$ x_g \f$ and photometric \f$ x_p \f$ dimensions of 
     *                the \f$ x\epsilon\mathbb{R}^8 \f$ points. That is, \f$ \|x-x'\|_2^2= 
     *                f_g(a)\|x_g-x'_g\|_2^2+f_p(a)\|x_p-x'_p\|_2^2 \f$. For more info, 
     *                look at `euclideanSquaredMetric8` in [kernels/rbc_kernels.cl]
     *                (https://random-ball-cover.nlamprian.me).
     *  \param[in] _c scaling factor for dealing with floating point arithmetic 
     *                issues when computing the `S` matrix.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::WEIGHTED>::init (
        unsigned int _m, unsigned int _nr, float _a, float _c, Staging _staging)
    {
        m = _m; nr = _nr; a = _a; c = _c;
        bufferFMSize = m * sizeof (cl_float8);
        bufferTSize = 2 * sizeof (cl_float4);
        staging = _staging;

        try
        {
            if (m == 0)
                throw "The sets of landmarks cannot have zero points";

            if (nr == 0)
                throw "The sets of representatives cannot have zero points";

            if (a == 0.f)
                throw "The alpha parameter cannot be equal to zero";
        }
        catch (const char *error)
        {
            std::cerr << "Error[ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::WEIGHTED>]: " 
                      << error << std::endl;
            exit (EXIT_FAILURE);
        }
        
        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInF = nullptr;
                hPtrInM = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInF () == nullptr)
                    hBufferInF = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferFMSize);
                if (hBufferInM () == nullptr)
                    hBufferInM = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferFMSize);

                hPtrInF = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInF, CL_FALSE, CL_MAP_WRITE, 0, bufferFMSize);
                hPtrInM = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInM, CL_FALSE, CL_MAP_WRITE, 0, bufferFMSize);
                queue.enqueueUnmapMemObject (hBufferInF, hPtrInF);
                queue.enqueueUnmapMemObject (hBufferInM, hPtrInM);

                if (!io)
                {
                    queue.finish ();
                    break;
                }

            case Staging::O:
                if (!io)
                {
                    hPtrInF = nullptr;
                    hPtrInM = nullptr;
                }
                break;
        }

        if (hBufferIOT () == nullptr)
            hBufferIOT = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferTSize);

        hPtrIOT = (cl_float *) queue.enqueueMapBuffer (
            hBufferIOT, CL_FALSE, CL_MAP_READ, 0, bufferTSize);
        queue.enqueueUnmapMemObject (hBufferIOT, hPtrIOT);
        queue.finish ();

        // Create device buffers
        if (dBufferInF () == nullptr)
            dBufferInF = cl::Buffer (context, CL_MEM_READ_ONLY, bufferFMSize);
        if (dBufferInM () == nullptr)
            dBufferInM = cl::Buffer (context, CL_MEM_READ_ONLY, bufferFMSize);
        if (dBufferIOT () == nullptr)
            dBufferIOT = cl::Buffer (context, CL_MEM_READ_WRITE, bufferTSize);

        // Load initial identity transformation
        cl_float T0[8] = { 0, 0, 0, 1, 0, 0, 0, 1 };
        std::copy (T0, T0 + 8, hPtrIOT);
        queue.enqueueWriteBuffer (dBufferIOT, CL_FALSE, 0, bufferTSize, hPtrIOT);

        R = Eigen::Matrix3f::Identity ();
        q = Eigen::Quaternionf (R);
        t.setZero ();
        s = 1.f;

        // Configure classes

        // Classes in the initialization step ==================================
        
        fReps.get (ICPReps::Memory::D_IN) = dBufferInF;
        fReps.get (ICPReps::Memory::D_OUT) = cl::Buffer (context, CL_MEM_READ_WRITE, nr * sizeof (cl_float8));
        fReps.init (nr, Staging::NONE);

        const RBC::KernelTypeC K1 = RBC::KernelTypeC::KINECT_R;
        const RBC::RBCPermuteConfig P1 = RBC::RBCPermuteConfig::GENERIC;

        rbcC.get (RBC::RBCConstruct<K1, P1>::Memory::D_IN_X) = dBufferInF;
        rbcC.get (RBC::RBCConstruct<K1, P1>::Memory::D_IN_R) = fReps.get (ICPReps::Memory::D_OUT);
        rbcC.init (m, nr, d, a, 0, RBC::Staging::NONE);

        // Classes in the iteration step =======================================

        const ICPTransformConfig TC = ICPTransformConfig::QUATERNION;

        transform.get (ICPTransform<TC>::Memory::D_IN_M) = dBufferInM;
        transform.get (ICPTransform<TC>::Memory::D_IN_T) = dBufferIOT;
        transform.get (ICPTransform<TC>::Memory::D_OUT) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, m * sizeof (cl_float8));
        transform.init (m, Staging::NONE);

        const RBC::KernelTypeC K2 = RBC::KernelTypeC::KINECT_R;
        const RBC::RBCPermuteConfig P2 = RBC::RBCPermuteConfig::GENERIC;
        const RBC::KernelTypeS S2 = RBC::KernelTypeS::KINECT;

        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_Q) = 
            transform.get (ICPTransform<TC>::Memory::D_OUT);
        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_R) = 
            fReps.get (ICPReps::Memory::D_OUT);
        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_X_P) = 
            rbcC.get (RBC::RBCConstruct<K1, P1>::Memory::D_OUT_X_P);
        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_O) = 
            rbcC.get (RBC::RBCConstruct<K1, P1>::Memory::D_OUT_O);
        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_N) = 
            rbcC.get (RBC::RBCConstruct<K1, P1>::Memory::D_OUT_N);
        rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_NN) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, m * sizeof (cl_float8));
        rbcS.init (m, nr, m, a, RBC::Staging::NONE);

        weights.get (ICPWeights::Memory::D_IN) = 
            rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_NN_ID);
        weights.get (ICPWeights::Memory::D_OUT_W) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, m * sizeof (cl_float));
        weights.get (ICPWeights::Memory::D_OUT_SUM_W) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, sizeof (cl_double));
        weights.init (m, Staging::NONE);

        const ICPMeanConfig MC = ICPMeanConfig::WEIGHTED;

        means.get (ICPMean<MC>::Memory::D_IN_F) = 
            rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_NN);
        means.get (ICPMean<MC>::Memory::D_IN_M) = 
            rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_Q_P);
        means.get (ICPMean<MC>::Memory::D_IN_W) = 
            weights.get (ICPWeights::Memory::D_OUT_W);
        means.get (ICPMean<MC>::Memory::D_IN_SUM_W) = 
            weights.get (ICPWeights::Memory::D_OUT_SUM_W);
        means.get (ICPMean<MC>::Memory::D_OUT) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, 2 * sizeof (cl_float4));
        means.init (m, Staging::O);

        devs.get (ICPDevs::Memory::D_IN_F) = rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_NN);
        devs.get (ICPDevs::Memory::D_IN_M) = rbcS.get (RBC::RBCSearch<K2, P2, S2>::Memory::D_OUT_Q_P);
        devs.get (ICPDevs::Memory::D_IN_MEAN) = means.get (ICPMean<MC>::Memory::D_OUT);
        devs.get (ICPDevs::Memory::D_OUT_DEV_F) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, m * sizeof (cl_float4));
        devs.get (ICPDevs::Memory::D_OUT_DEV_M) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, m * sizeof (cl_float4));
        devs.init (m, Staging::NONE);

        const ICPSConfig SC = ICPSConfig::WEIGHTED;

        matrixS.get (ICPS<SC>::Memory::D_IN_DEV_M) = devs.get (ICPDevs::Memory::D_OUT_DEV_M);
        matrixS.get (ICPS<SC>::Memory::D_IN_DEV_F) = devs.get (ICPDevs::Memory::D_OUT_DEV_F);
        matrixS.get (ICPS<SC>::Memory::D_IN_W) = weights.get (ICPWeights::Memory::D_OUT_W);
        matrixS.get (ICPS<SC>::Memory::D_OUT) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, 11 * sizeof (cl_float));
        matrixS.init (m, c, Staging::NONE);

        powMethod.get (ICPPowerMethod::Memory::D_IN_S) = matrixS.get (ICPS<SC>::Memory::D_OUT);
        powMethod.get (ICPPowerMethod::Memory::D_IN_MEAN) = means.get (ICPMean<MC>::Memory::D_OUT);
        powMethod.get (ICPPowerMethod::Memory::D_OUT_T_K) = dBufferIOT;
        powMethod.init (Staging::O);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::WEIGHTED>::write (
        ICPStep::Memory mem, void *ptr, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPStep::Memory::D_IN_F:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + m * d, hPtrInF);
                    queue.enqueueWriteBuffer (dBufferInF, block, 0, bufferFMSize, hPtrInF, events, event);
                    break;
                case ICPStep::Memory::D_IN_M:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + m * d, hPtrInM);
                    queue.enqueueWriteBuffer (dBufferInM, block, 0, bufferFMSize, hPtrInM, events, event);
                    break;
                case ICPStep::Memory::D_IO_T:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + 8, hPtrIOT);
                    queue.enqueueWriteBuffer (dBufferIOT, block, 0, bufferTSize, hPtrIOT, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::WEIGHTED>::read (
        ICPStep::Memory mem, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case ICPStep::Memory::H_IO_T:
                    queue.enqueueReadBuffer (dBufferIOT, block, 0, bufferTSize, hPtrIOT, events, event);
                    return hPtrIOT;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \note Call `buildRBC` after the `D_IN_F` buffer has been written, 
     *        and before any calls to `run` (for each registration).
     */
    void ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::WEIGHTED>::buildRBC (
        const std::vector<cl::Event> *events, cl::Event *event)
    {
        fReps.run (events);
        rbcC.run (nullptr, event);
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the kernel execution.
     *  \param[in] config flag. If true, configures the `RBC search` process. 
     *                    Set to true once, when the `RBC` data structure is reset.
     */
    void ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::WEIGHTED>::run (
        const std::vector<cl::Event> *events, cl::Event *event, bool config)
    {
        transform.run ();
        rbcS.run (nullptr, nullptr, config);
        weights.run ();
        means.run ();
        devs.run ();
        matrixS.run ();
        powMethod.run ();

        Tk = (cl_float *) powMethod.read (ICPPowerMethod::Memory::H_OUT_T_K);

        qk = Eigen::Quaternionf (Tk);
        Rk = qk.toRotationMatrix ();
        tk = Eigen::Map<Eigen::Vector3f> (Tk + 4, 3);
        sk = Tk[7];

        R = Rk * R;
        q = Eigen::Quaternionf (R);
        t = sk * Rk * t + tk;
        s = sk * s;
        
        Eigen::Map<Eigen::Vector4f> (hPtrIOT, 4) = q.coeffs ();  // Quaternion
        Eigen::Map<Eigen::Vector4f> (hPtrIOT + 4, 4) = t.homogeneous ();  // Translation
        hPtrIOT[7] = s;  // Scale

        queue.enqueueWriteBuffer (dBufferIOT, CL_FALSE, 0, bufferTSize, hPtrIOT, nullptr, event);
    }


    /*! \return The scaling parameter \f$ \alpha \f$. */
    float ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::WEIGHTED>::getAlpha ()
    {
        return a;
    }


    /*! \details Updates the kernel arguments for the scaling parameter \f$ \alpha \f$.
     *
     *  \param[in] _a scaling parameter \f$ \alpha \f$.
     */
    void ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::WEIGHTED>::setAlpha (float _a)
    {
        a = _a;
        rbcC.setAlpha (a);
        rbcS.setAlpha (a);
    }


    /*! \note The scaling factor c multiplies the points (deviations) before processing
     *        in order to deal with floating point arithmetic issues.
     *  
     *  \return The scaling factor c.
     */
    float ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::WEIGHTED>::getScaling ()
    {
        return c;
    }


    /*! \details Updates the kernel argument for the scaling factor c.
     *  \note The scaling factor c multiplies the points (deviations) before processing
     *        in order to deal with floating point arithmetic issues.
     *  
     *  \param[in] _c scaling factor.
     */
    void ICPStep<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::WEIGHTED>::setScaling (float _c)
    {
        c = _c;
        matrixS.setScaling (c);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _infoRBC opencl configuration for the `RBC` classes. 
     *                      It specifies the context, queue, etc, to be used.
     *  \param[in] _infoICP opencl configuration for the `ICP` classes. 
     *                      It specifies the context, queue, etc, to be used.
     */
    template <ICPStepConfigT CR, ICPStepConfigW CW>
    ICP<CR, CW>::ICP (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _infoRBC, clutils::CLEnvInfo<1> _infoICP) : 
        ICPStep<CR, CW>::ICPStep (_env, _infoRBC, _infoICP)
    {
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _m number of points in the sets.
     *  \param[in] _nr number of fixed set representatives.
     *  \param[in] _a factor scaling the results of the distance calculations for the 
     *                geometric \f$ x_g \f$ and photometric \f$ x_p \f$ dimensions of 
     *                the \f$ x\epsilon\mathbb{R}^8 \f$ points. That is, \f$ \|x-x'\|_2^2= 
     *                f_g(a)\|x_g-x'_g\|_2^2+f_p(a)\|x_p-x'_p\|_2^2 \f$. For more info, 
     *                look at `euclideanSquaredMetric8` in [kernels/rbc_kernels.cl]
     *                (https://random-ball-cover.nlamprian.me).
     *  \param[in] _c scaling factor for dealing with floating point arithmetic 
     *                issues when computing the `S` matrix.
     *  \param[in] _max_iterations maximum number of iterations that a registration is allowed to perform.
     *  \param[in] _angle_threshold threshold for the change in angle (in degrees) in the transformation.
     *  \param[in] _translation_threshold threshold for the change in translation (in mm) in the transformation.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    template <ICPStepConfigT CR, ICPStepConfigW CW>
    void ICP<CR, CW>::init (unsigned int _m, unsigned int _nr, float _a, float _c, unsigned int _max_iterations, 
        double _angle_threshold, double _translation_threshold, Staging _staging)
    {
        max_iterations = _max_iterations;
        angle_threshold = _angle_threshold;
        translation_threshold = _translation_threshold;

        ICPStep<CR, CW>::init (_m, _nr, _a, _c, _staging);
    }


    /*! \note Call `buildRBC` after the `D_IN_F` buffer has been written, 
     *        and before any calls to `run` (for each registration).
     */
    template <ICPStepConfigT CR, ICPStepConfigW CW>
    void ICP<CR, CW>::buildRBC (const std::vector<cl::Event> *events, cl::Event *event)
    {
        ICPStep<CR, CW>::buildRBC (events, event);
        k = 0;
    }


    /*! \details Executes the iterative ICP algorithm and estimates the 
     *           relative transformation between the two associated point clouds.
     *  \note The function call is blocking, so it doesn't need to offer an event. 
     *        It also doesn't accept events, since it's meant to be called after 
     *        `buildRBC` which is the one that will wait on the events.
     */
    template <ICPStepConfigT CR, ICPStepConfigW CW>
    void ICP<CR, CW>::run ()
    {
        ICPStep<CR, CW>::run (nullptr, nullptr, true);
        
        while (check ()) ICPStep<CR, CW>::run ();
        
        this->queue.finish ();
    }


    /*! \details Checks the change in the transformation and the number of iterations.
     *  \note Call `buildRBC` after the `D_IN_F` buffer has been written, 
     *        and before any calls to `run` (for each registration).
     *        
     *  \return `false` for convergence, `true` otherwise.
     */
    template <ICPStepConfigT CR, ICPStepConfigW CW>
    inline bool ICP<CR, CW>::check ()
    {
        k++;
        double delta_angle = 180.0 / M_PI * 2.0 * std::atan2 (this->qk.vec ().norm (), this->qk.w ());  // in degrees
        double delta_tanslation = this->tk.norm ();  // in mm

        if (k == max_iterations) return false;
        if (delta_angle < angle_threshold && delta_tanslation < translation_threshold) return false;

        return true;
    }


    /*! \return The maximum number of iterations. */
    template <ICPStepConfigT CR, ICPStepConfigW CW>
    unsigned int ICP<CR, CW>::getMaxIterations ()
    {
        return max_iterations;
    }


    /*! \details Updates the parameter for the maximum number of iterations.
     *  
     *  \param[in] _max_iterations maximum number of iterations.
     */
    template <ICPStepConfigT CR, ICPStepConfigW CW>
    void ICP<CR, CW>::setMaxIterations (unsigned int _max_iterations)
    {
        max_iterations = _max_iterations;
    }


    /*! \return The threshold for the change in angle (in degrees). */
    template <ICPStepConfigT CR, ICPStepConfigW CW>
    double ICP<CR, CW>::getAngleThreshold ()
    {
        return angle_threshold;
    }


    /*! \details Updates the parameter for the threshold for the change 
     *           in angle (in degrees) in the transformation.
     *  
     *  \param[in] _angle_threshold threshold for the change in angle (in degrees).
     */
    template <ICPStepConfigT CR, ICPStepConfigW CW>
    void ICP<CR, CW>::setAngleThreshold (double _angle_threshold)
    {
        angle_threshold = _angle_threshold;
    }


    /*! \return The threshold for the change in translation (in mm). */
    template <ICPStepConfigT CR, ICPStepConfigW CW>
    double ICP<CR, CW>::getTranslationThreshold ()
    {
        return translation_threshold;
    }


    /*! \details Updates the parameter for the threshold for the change 
     *           in translation (in mm) in the transformation.
     *  
     *  \param[in] _translation_threshold threshold for the change in translation (in mm).
     */
    template <ICPStepConfigT CR, ICPStepConfigW CW>
    void ICP<CR, CW>::setTranslationThreshold (double _translation_threshold)
    {
        translation_threshold = _translation_threshold;
    }


    /*! \brief Instantiation that uses the Eigen library to estimate the rotation, and considers regular residual errors.  */
    template class ICP<ICPStepConfigT::EIGEN, ICPStepConfigW::REGULAR>;
    /*! \brief Instantiation that uses the Eigen library to estimate the rotation, and considers weighted residual errors. */
    template class ICP<ICPStepConfigT::EIGEN, ICPStepConfigW::WEIGHTED>;
    /*! \brief Instantiation that uses the Power Method to estimate the rotation, and considers regular residual errors.   */
    template class ICP<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::REGULAR>;
    /*! \brief Instantiation that uses the Power Method to estimate the rotation, and considers weighted residual errors.  */
    template class ICP<ICPStepConfigT::POWER_METHOD, ICPStepConfigW::WEIGHTED>;

}
}
