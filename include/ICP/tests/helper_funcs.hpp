/*! \file helper_funcs.hpp
 *  \brief Declarations of helper functions for testing.
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

#ifndef ICP_HELPERFUNCS_HPP
#define ICP_HELPERFUNCS_HPP

#include <cassert>
#include <algorithm>
#include <functional>
#include <RBC/data_types.hpp>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif


/*! \brief Offers functions that are serial `CPU` implementations of 
 *         the relevant algorithms in the `%ICP` pipeline.
 */
namespace ICP
{

    /*! \brief Checks the command line arguments for the profiling flag, `--profiling`. */
    bool setProfilingFlag (int argc, char **argv);


    /*! \brief Returns the first power of 2 greater than or equal to the input.
     *
     *  \param[in] num input number.
     *  \return The first power of 2 >= num.
     */
    template <typename T>
    uint64_t nextPow2 (T num)
    {
        assert (num >= 0);

        uint64_t pow;
        for (pow = 1; pow < (uint64_t) num; pow <<= 1) ;

        return pow;
    }


    /*! \brief Prints an array of an integer type to standard output.
     *
     *  \tparam T type of the data to be printed.
     *  \param[in] title legend for the output.
     *  \param[in] ptr array that is to be displayed.
     *  \param[in] width width of the array.
     *  \param[in] height height of the array.
     */
    template <typename T>
    void printBuffer (const char *title, T *ptr, uint32_t width, uint32_t height)
    {
        std::cout << title << std::endl;

        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                std::cout << std::setw (3 * sizeof (T)) << +ptr[row * width + col] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << std::endl;
    }


    /*! \brief Prints an array of floating-point type to standard output.
     *
     *  \tparam T type of the data to be printed.
     *  \param[in] title legend for the output.
     *  \param[in] ptr array that is to be displayed.
     *  \param[in] width width of the array.
     *  \param[in] height height of the array.
     *  \param[in] prec the number of decimal places to print.
     */
    template <typename T>
    void printBufferF (const char *title, T *ptr, uint32_t width, uint32_t height, uint32_t prec)
    {
        std::ios::fmtflags f (std::cout.flags ());
        std::cout << title << std::endl;
        std::cout << std::fixed << std::setprecision (prec);

        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                std::cout << std::setw (5 + prec) << ptr[row * width + col] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << std::endl;
        std::cout.flags (f);
    }


    /*! \brief Reduces each row of an array to a single element.
     *  \details It is just a naive serial implementation.
     *
     *  \param[in] in input data.
     *  \param[out] out output (reduced) data.
     *  \param[in] cols number of columns in the input array.
     *  \param[in] rows number of rows in the input array.
     *  \param[in] func function supporting the requested operation.
     */
    template <typename T>
    void cpuReduce (T *in, T *out, uint32_t cols, uint32_t rows, std::function<bool (T, T)> func)
    {
        for (uint r = 0; r < rows; ++r)
        {
            T rec = in[r * cols];
            for (uint c = 1; c < cols; ++c)
            {
                T tmp = in[r * cols + c];
                if (func (tmp, rec)) rec = tmp;
            }
            out[r] = rec;
        }
    }


    /*! \brief Reduces each row of an array to a single element (sum).
     *  \details It is just a naive serial implementation.
     *
     *  \param[in] in input data.
     *  \param[out] out output (reduced) data.
     *  \param[in] cols number of columns in the input array.
     *  \param[in] rows number of rows in the input array.
     */
    template <typename T>
    void cpuReduceSum (T *in, T *out, uint32_t cols, uint32_t rows)
    {
        for (uint r = 0; r < rows; ++r)
            out[r] = std::accumulate (&in[r * cols], &in[r * cols] + cols, 0.f);
    }


    /*! \brief Performs an inclusive scan operation on the columns of an array.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] in input data.
     *  \param[out] out output (scan) data.
     *  \param[in] width width of the array.
     *  \param[in] height height of the array.
     */
    template <typename T>
    void cpuInScan (T *in, T *out, uint32_t width, uint32_t height)
    {
        // Initialize the first element of each row
        for (uint32_t row = 0; row < height; ++row)
            out[row * width] = in[row * width];
        // Perform the scan
        for (uint32_t row = 0; row < height; ++row)
            for (uint32_t col = 1; col < width; ++col)
                out[row * width + col] = out[row * width + col - 1] + in[row * width + col];
    }


    /*! \brief Performs an exclusive scan operation on the columns of an array.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] in input data.
     *  \param[out] out output (scan) data.
     *  \param[in] width width of the array.
     *  \param[in] height height of the array.
     */
    template <typename T>
    void cpuExScan (T *in, T *out, uint32_t width, uint32_t height)
    {
        // Initialize the first element of each row
        for (uint32_t row = 0; row < height; ++row)
            out[row * width] = 0;
        // Perform the scan
        for (uint32_t row = 0; row < height; ++row)
            for (uint32_t col = 1; col < width; ++col)
                out[row * width + col] = out[row * width + col - 1] + in[row * width + col - 1];
    }


    /*! \brief Samples a point cloud for 16384 (128x128) landmarks.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] in input data.
     *  \param[out] out output (LM) data.
     */
    template <typename T>
    void cpuICPLMs (T *in, T *out)
    {
        for (uint32_t gY = 0; gY < 128; ++gY)
        {
            uint32_t yi = gY * 3 + 1;

            for (uint32_t gX = 0; gX < 128 * 8; gX += 8)
            {
                uint32_t xi = gX * 4 + 1 * 8;

                for (uint32_t k = 0; k < 8; ++k)
                    out[gY * (128 * 8) + gX + k] = in[(48 + yi) * (640 * 8) + ((64 * 8) + xi) + k];
            }
        }
    }


    /*! \brief Samples a set of 16384 (128x128) landmarks for representatives.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] in input data.
     *  \param[out] out output (LM) data.
     *  \param[in] nr number of representatives.
     */
    template <typename T>
    void cpuICPReps (T *in, T *out, uint32_t nr)
    {
        int p = std::log2 (nr);
        uint32_t nrx = std::pow (2, p - p / 2);
        uint32_t nry = std::pow (2, p / 2);

        uint stepX = 128 / nrx;
        uint stepY = 128 / nry;

        for (uint32_t gY = 0; gY < nry; ++gY)
        {
            uint32_t yi = gY * stepY + (stepY >> 1) - 1;

            for (uint32_t gX = 0; gX < nrx * 8; gX += 8)
            {
                uint32_t xi = gX * stepX + ((stepX >> 1) - 1) * 8;

                for (uint32_t k = 0; k < 8; ++k)
                    out[gY * (nrx * 8) + gX + k] = in[yi * (128 * 8) + xi + k];
            }
        }
    }


    /*! \brief Computes weights for pairs of points in the fixed and moving sets, 
     *         and also reduces them to get their sum.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] D input data (distances).
     *  \param[out] W output data (weights).
     *  \param[out] SW output data (sum of weights).
     *  \param[in] n number of elements in the input array.
     */
    template <typename T>
    void cpuICPWeights (rbc_dist_id *D, T *W, cl_double *SW, uint32_t n)
    {
        for (uint32_t j = 0; j < n; ++j)
            W[j] = 100.f / (100.f + D[j].dist);

        *SW = std::accumulate (W, W + n, 0.0);
    }


    /*! \brief Computes the mean on the xyz dimensions of the set of 8-D points.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] F fixed set.
     *  \param[in] M moving set.
     *  \param[out] mean output data.
     *  \param[in] n number of points in the array.
     */
    template <typename T>
    void cpuICPMean (T *F, T *M, T *mean, uint32_t n)
    {
        mean[0] = mean[1] = mean[2] = mean[3] = 0.0;
        mean[4] = mean[5] = mean[6] = mean[7] = 0.0;

        for (uint32_t j = 0; j < n; ++j)
        {
            mean[0] += F[j * 8] / (T) n;
            mean[1] += F[j * 8 + 1] / (T) n;
            mean[2] += F[j * 8 + 2] / (T) n;
            
            mean[4] += M[j * 8] / (T) n;
            mean[5] += M[j * 8 + 1] / (T) n;
            mean[6] += M[j * 8 + 2] / (T) n;
        }
    }


    /*! \brief Computes the weighted mean on the xyz dimensions of the set of 8-D points.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] F fixed set.
     *  \param[in] M moving set.
     *  \param[out] MEAN output data.
     *  \param[in] W weights.
     *  \param[in] n number of points in the array.
     */
    template <typename T>
    void cpuICPMeanWeighted (T *F, T *M, T *MEAN, T *W, uint32_t n)
    {
        MEAN[0] = MEAN[1] = MEAN[2] = MEAN[3] = 0.0;
        MEAN[4] = MEAN[5] = MEAN[6] = MEAN[7] = 0.0;

        cl_double sum_w = std::accumulate (W, W + n, 0.0);

        for (uint32_t j = 0; j < n; ++j)
        {
            T w = W[j] / sum_w;

            MEAN[0] += w * F[j * 8];
            MEAN[1] += w * F[j * 8 + 1];
            MEAN[2] += w * F[j * 8 + 2];

            MEAN[4] += w * M[j * 8];
            MEAN[5] += w * M[j * 8 + 1];
            MEAN[6] += w * M[j * 8 + 2];
        }
    }


    /*! \brief Computes the deviations of a set of points from their mean.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] F fixed set of 8-D elements.
     *  \param[in] M moving set of 8-D elements.
     *  \param[out] DM array (moving set deviations) of 4-D elements.
     *  \param[out] DF array (fixed set deviations) of 4-D elements.
     *  \param[in] mean fixed and moving set means.
     *  \param[in] n number of points in the sets.
     */
    template <typename T>
    void cpuICPDevs (T *F, T *M, T *DF, T *DM, T *mean, uint32_t n)
    {
        for (uint32_t j = 0; j < n; ++j)
        {
            for (uint32_t k = 0; k < 4; ++k)
            {
                DF[j * 4 + k] = F[j * 8 + k] - mean[k];
                DM[j * 4 + k] = M[j * 8 + k] - mean[4 + k];
            }
        }
    }


    /*! \brief Calculates the S matrix and the constituents of the scale factor s.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] DM input array (moving set deviations) of 4-D elements.
     *  \param[in] DF input array (fixed set deviations) of 4-D elements.
     *  \param[out] S output (sums of products) matrix S.
     *  \param[in] m number of points in the sets.
     *  \param[in] c scaling factor.
     */
    template <typename T>
    void cpuICPS (T *DM, T *DF, T *S, uint32_t m, float c)
    {
        for (uint32_t j = 0; j < 11; ++j)
            S[j] = 0.f;

        for (uint32_t i = 0; i < m; ++i)
        {
            T mp[3] = { c * DM[i * 4], c * DM[i * 4 + 1], c * DM[i * 4 + 2] };
            T fp[3] = { c * DF[i * 4], c * DF[i * 4 + 1], c * DF[i * 4 + 2] };

            S[0] += mp[0] * fp[0];
            S[1] += mp[0] * fp[1];
            S[2] += mp[0] * fp[2];
            S[3] += mp[1] * fp[0];
            S[4] += mp[1] * fp[1];
            S[5] += mp[1] * fp[2];
            S[6] += mp[2] * fp[0];
            S[7] += mp[2] * fp[1];
            S[8] += mp[2] * fp[2];
            S[9]  += mp[0] * mp[0] + mp[1] * mp[1] + mp[2] * mp[2];
            S[10] += fp[0] * fp[0] + fp[1] * fp[1] + fp[2] * fp[2];
        }
    }


    /*! \brief Calculates the S matrix and the constituents of the scale factor s.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] M input array (moving set deviations) of 4-D elements.
     *  \param[in] F input array (fixed set deviations) of 4-D elements.
     *  \param[in] W input array (weights).
     *  \param[out] S output (sums of products) matrix S.
     *  \param[in] m number of points in the sets.
     *  \param[in] c scaling factor.
     */
    template <typename T>
    void cpuICPSw (T *M, T *F, T *W, T *S, uint32_t m, float c)
    {
        for (uint32_t j = 0; j < 11; ++j)
            S[j] = 0.f;

        for (uint32_t i = 0; i < m; ++i)
        {
            T mp[3] = { c * M[i * 4], c * M[i * 4 + 1], c * M[i * 4 + 2] };
            T fp[3] = { c * F[i * 4], c * F[i * 4 + 1], c * F[i * 4 + 2] };
            T w = W[i];

            S[0] += w * mp[0] * fp[0];
            S[1] += w * mp[0] * fp[1];
            S[2] += w * mp[0] * fp[2];
            S[3] += w * mp[1] * fp[0];
            S[4] += w * mp[1] * fp[1];
            S[5] += w * mp[1] * fp[2];
            S[6] += w * mp[2] * fp[0];
            S[7] += w * mp[2] * fp[1];
            S[8] += w * mp[2] * fp[2];
            S[9]  += w * (mp[0] * mp[0] + mp[1] * mp[1] + mp[2] * mp[2]);
            S[10] += w * (fp[0] * fp[0] + fp[1] * fp[1] + fp[2] * fp[2]);
        }
    }


    /*! \brief Performs a cross product.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] a first input vector.
     *  \param[in] b second input vector.
     *  \param[out] c output vector.
     */
    template <typename T>
    void cross_product (T *a, T *b, T *c)
    {
        c[0] = (a[1] * b[2]) - (a[2] * b[1]);
        c[1] = (a[2] * b[0]) - (a[0] * b[2]);
        c[2] = (a[0] * b[1]) - (a[1] * b[0]);
    }


    /*! \brief Performs a homogeneous transformation on a set of points using 
     *         a quaternion and a translation vector.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] M input array (set of points) of 8-D elements.
     *  \param[out] tM output array (transformed points) of 8-D elements.
     *  \param[in] D transformation parameters.
     *  \param[in] m number of points in the set.
     */
    template <typename T>
    void cpuICPTransformQ (T *M, T *tM, T *D, uint32_t m)
    {
        T q[4] = { D[0], D[1], D[2], D[3] };
        T t[3] = { D[4], D[5], D[6] };
        T s = D[7];

        for (uint32_t i = 0; i < m; ++i)
        {
            T p[3] = { M[i * 8], M[i * 8 + 1], M[i * 8 + 2] };

            T q2[3] = { 2 * q[0], 2 * q[1], 2 * q[2] };

            T qcp[3]; cross_product (q, p, qcp);
            qcp[0] = qcp[0] + q[3] * p[0];
            qcp[1] = qcp[1] + q[3] * p[1];
            qcp[2] = qcp[2] + q[3] * p[2];

            T tp[3], q2cqcp[3]; cross_product (q2, qcp, q2cqcp);
            tp[0] = s * (p[0] + q2cqcp[0]) + t[0];
            tp[1] = s * (p[1] + q2cqcp[1]) + t[1];
            tp[2] = s * (p[2] + q2cqcp[2]) + t[2];

            tM[i * 8]     = tp[0];
            tM[i * 8 + 1] = tp[1];
            tM[i * 8 + 2] = tp[2];
            tM[i * 8 + 3] = M[i * 8 + 3];
            tM[i * 8 + 4] = M[i * 8 + 4];
            tM[i * 8 + 5] = M[i * 8 + 5];
            tM[i * 8 + 6] = M[i * 8 + 6];
            tM[i * 8 + 7] = M[i * 8 + 7];
        }
    }


    /*! \brief Performs a homogeneous transformation on a set of points using 
     *         a quaternion and a translation vector.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] M input array (set of points) of 8-D elements.
     *  \param[out] tM output array (transformed points) of 8-D elements.
     *  \param[in] D transformation parameters.
     *  \param[in] m number of points in the set.
     */
    template <typename T>
    void cpuICPTransformQ2 (T *M, T *tM, T *D, uint32_t m)
    {
        T q[4] = { D[0], D[1], D[2], D[3] };
        T t[3] = { D[4], D[5], D[6] };
        T s = D[7];

        T Q[4][4] = { {  q[3], -q[2],  q[1], q[0] }, 
                      {  q[2],  q[3], -q[0], q[1] }, 
                      { -q[1],  q[0],  q[3], q[2] }, 
                      { -q[0], -q[1], -q[2], q[3] } };

        T Q_[3][4] = { {  q[3], -q[2],  q[1], -q[0] }, 
                       {  q[2],  q[3], -q[0], -q[1] }, 
                       { -q[1],  q[0],  q[3], -q[2] } };

        for (uint32_t i = 0; i < m; ++i)
        {
            T p[4] = { M[i * 8], M[i * 8 + 1], M[i * 8 + 2], 0.f };

            T p_[4] = { std::inner_product (Q[0], Q[0] + 4, p, 0.f), 
                        std::inner_product (Q[1], Q[1] + 4, p, 0.f), 
                        std::inner_product (Q[2], Q[2] + 4, p, 0.f), 
                        std::inner_product (Q[3], Q[3] + 4, p, 0.f) };

            T tp[4];
            tp[0] = s * std::inner_product (Q_[0], Q_[0] + 4, p_, 0.f) + t[0];
            tp[1] = s * std::inner_product (Q_[1], Q_[1] + 4, p_, 0.f) + t[1];
            tp[2] = s * std::inner_product (Q_[2], Q_[2] + 4, p_, 0.f) + t[2];

            tM[i * 8]     = tp[0];
            tM[i * 8 + 1] = tp[1];
            tM[i * 8 + 2] = tp[2];
            tM[i * 8 + 3] = M[i * 8 + 3];
            tM[i * 8 + 4] = M[i * 8 + 4];
            tM[i * 8 + 5] = M[i * 8 + 5];
            tM[i * 8 + 6] = M[i * 8 + 6];
            tM[i * 8 + 7] = M[i * 8 + 7];
        }
    }


    /*! \brief Performs a homogeneous transformation on a set of points using 
     *         a transformation matrix.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] M input array (set of points) of 8-D elements.
     *  \param[out] tM output array (transformed points) of 8-D elements.
     *  \param[in] D transformation parameters.
     *  \param[in] m number of points in the set.
     */
    template <typename T>
    void cpuICPTransformM (T *M, T *tM, T *D, uint32_t m)
    {
        for (uint32_t i = 0; i < m; ++i)
        {
            tM[i * 8]     = std::inner_product (&D[0], &D[4], &M[i * 8], 0.f);
            tM[i * 8 + 1] = std::inner_product (&D[4], &D[8], &M[i * 8], 0.f);
            tM[i * 8 + 2] = std::inner_product (&D[8], &D[12], &M[i * 8], 0.f);
            tM[i * 8 + 3] = M[i * 8 + 3];
            tM[i * 8 + 4] = M[i * 8 + 4];
            tM[i * 8 + 5] = M[i * 8 + 5];
            tM[i * 8 + 6] = M[i * 8 + 6];
            tM[i * 8 + 7] = M[i * 8 + 7];
        }
    }


    /*! \brief Computes the vector length (\f$\ell_2\f$ norm).
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] x 4-vector.
     *  \return The vector length.
     */
    template <typename T>
    T cpuLength (T *x)
    {
        T sum = 0.f;

        sum += x[0] * x[0];
        sum += x[1] * x[1];
        sum += x[2] * x[2];
        sum += x[3] * x[3];

        return std::sqrt (sum);
    }


    /*! \brief Computes the vector distance (\f$\ell_2\f$ norm).
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] x1 4-vector.
     *  \param[in] x2 4-vector.
     *  \return The vector distance.
     */
    template <typename T>
    T cpuDistance (T *x1, T *x2)
    {
        T sum = 0.f;

        sum += std::pow (x1[0] - x2[0], 2);
        sum += std::pow (x1[1] - x2[1], 2);
        sum += std::pow (x1[2] - x2[2], 2);
        sum += std::pow (x1[3] - x2[3], 2);

        return std::sqrt (sum);
    }


    /*! \brief Normalizes a vector.
     *
     *  \tparam T type of the data to be handled.
     *  \param x 4-vector.
     */
    template <typename T>
    void cpuNormalize (T *x)
    {
        T norm = cpuLength (x);

        x[0] /= norm;
        x[1] /= norm;
        x[2] /= norm;
        x[3] /= norm;
    }


    /*! \brief Computes a matrix-vector product, \f$ x_{new}=Nx \f$.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] N `4x4` matrix.
     *  \param[in] x 4-vector.
     *  \param[out] x_new 4-vector.
     */
    template <typename T>
    void cpuProd (T *N, T *x, T *x_new)
    {
        
        x_new[0] = std::inner_product (N     , N +  4, x, 0.f);
        x_new[1] = std::inner_product (N +  4, N +  8, x, 0.f);
        x_new[2] = std::inner_product (N +  8, N + 12, x, 0.f);
        x_new[3] = std::inner_product (N + 12, N + 16, x, 0.f);
    }


    /*! \brief Computes the quantities that represent the incremental 
     *         development in the transformation estimation in iteration `k`.
     *  \details It is just a naive serial implementation.
     *
     *  \tparam T type of the data to be handled.
     *  \param[in] Sij sums of products.
     *  \param[in] means fixed and moving set means.
     *  \param[out] Tk the **unit quaternion** \f$ \dot{q_k} = q_w + q_x i + q_y j + q_z k = 
     *                 \left[ \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$, 
     *                 and  the **translation vector** \f$ t_k=\left[ \begin{matrix} 
     *                 t_x & t_y & t_z & 1 \end{matrix} \right]^T \f$. The scale is placed 
     *                 in the last element of the translation vector. That is, \f$ t_k = 
     *                 \left[ \begin{matrix} t_x & t_y & t_z & s_k \end{matrix} \right]^T \f$.
     */
    template <typename T>
    T cpuICPPowerMethod (T *Sij, T *means, T *Tk)
    {
        T Sxx = Sij[0];
        T Sxy = Sij[1];
        T Sxz = Sij[2];
        T Syx = Sij[3];
        T Syy = Sij[4];
        T Syz = Sij[5];
        T Szx = Sij[6];
        T Szy = Sij[7];
        T Szz = Sij[8];
        
        T sk = sqrt (Sij[9] / Sij[10]);

        T N[16] = 
        {
            Sxx - Syy - Szz,         Sxy + Syx,         Szx + Sxz,       Syz - Szy, 
                  Sxy + Syx, - Sxx + Syy - Szz,         Syz + Szy,       Szx - Sxz, 
                  Szx + Sxz,         Syz + Szy, - Sxx - Syy + Szz,       Sxy - Syx, 
                  Syz - Szy,         Szx - Sxz,         Sxy - Syx, Sxx + Syy + Szz 
        };

        // Power Method ============================================================

        T x[4] = { 1.f, 1.f, 1.f, 1.f };
        T x_new[4];

        // Parameters
        uint maxIter = 1000;
        T error, error_new;

        while (true)
        {
            for (uint iter = 0; iter < maxIter; ++iter)
            {
                cpuProd (N, x, x_new);

                cpuNormalize (x_new);

                error = error_new;
                if ((error_new = cpuDistance (x, x_new)) == error) break;

                std::copy (x_new, x_new + 4, x);
            }

            T lambda = std::inner_product (N, N + 4, x_new, 0.f) / x_new[0];

            if (lambda < 0)
            {
                N[0] -= lambda;
                N[5] -= lambda;
                N[10] -= lambda;
                N[15] -= lambda;

                x[0] = x[1] = x[2] = x[3] = 1.f;
            }
            else
                break;
        }

        std::copy (x_new, x_new + 4, x);
        cpuProd (N, x, x_new);
        cpuNormalize (x_new);

        // =========================================================================

        T qk[4] = { x_new[0], x_new[1], x_new[2], x_new[3] };

        T mf[3] = { means[0], means[1], means[2] };
        T mm[3] = { means[4], means[5], means[6] };

        // tk = mf - sk * (mm + cross (2 * qk.xyz, cross (qk.xyz, mm) + qk.w * mm))
        T qk_2[3] = { 2 * qk[0], 2 * qk[1], 2 * qk[2] };
        T cp1[3]; cross_product (qk, mm, cp1);
        T mmw[3] = { qk[3] * mm[0], qk[3] * mm[1], qk[3] * mm[2] };
        T tmp1[3] = { cp1[0] + mmw[0], cp1[1] + mmw[1], cp1[2] + mmw[2] };
        T cp2[3]; cross_product (qk_2, tmp1, cp2);
        T tmp2[3] = { sk * (mm[0] + cp2[0]), sk * (mm[1] + cp2[1]), sk * (mm[2] + cp2[2]) };
        T tk[4] = { mf[0] - tmp2[0], mf[1] - tmp2[1], mf[2] - tmp2[2], sk };

        std::copy (qk, qk + 4, Tk);
        std::copy (tk, tk + 4, Tk + 4);
    }

}

#endif  // ICP_HELPERFUNCS_HPP
