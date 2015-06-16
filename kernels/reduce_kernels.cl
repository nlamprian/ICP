/*! \file reduce_kernels.cl
 *  \brief Kernels for performing the `Reduce` operation.
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


/*! \brief Performs a reduce operation on the columns of an array.
 *  \details Computes the minimum element for each row in an array.
 *  \note When there are multiple rows in the array, a reduce operation 
 *        is performed per row, in parallel.
 *  \note The number of elements, `N`, in a row of the array should be a **multiple 
 *        of 4** (the data are handled as `float4`). The **x** dimension of the 
 *        global workspace, \f$ gXdim \f$, should be greater than or equal to the number of 
 *        elements in a row of the array divided by 8. That is, \f$ \ gXdim \geq N/8 \f$. 
 *        Each work-item handles `8 float` (= `2 float4`) elements in a row of the array. 
 *        The **y** dimension of the global workspace, \f$ gYdim \f$, should be equal 
 *        to the number of rows, `M`, in the array. That is, \f$ \ gYdim = M \f$. 
 *        The local workspace should be `1` in the **y** dimension, and a 
 *        **power of 2** in the **x** dimension. It is recommended 
 *        to use one `wavefront/warp` per work-group.
 *  \note When the number of elements per row of the array is small enough to be 
 *        handled by a single work-group, the output array will contain the true 
 *        minimums. When the elements are more than that, they are partitioned 
 *        into blocks and reduced independently. In this case, the kernel 
 *        outputs the minimums from each block reduction. A reduction should then be
 *        made on those minimums for the final results. The number of work-groups
 *        in the **x** dimension, \f$ wgXdim \f$, **for the case of multiple 
 *        work-groups**, should be made a **multiple of 4**. The potential extra 
 *        work-groups are used for enforcing correctness. They write the necessary
 *        identity operands, `INFINITY`, in the output array, since in the next phase 
 *        the data are going to be handled as `float4`.
 *
 *  \param[in] in input array of `float` elements.
 *  \param[out] out (reduced) output array of `float` elements. When the kernel is
 *                  dispatched with one work-group per row, the array contains the 
 *                  final results, and its size should be \f$ rows*sizeof\ (float) \f$.
 *                  When the kernel is dispatched with more than one work-groups per row,
 *                  the array contains the results from each block reduction, and its size 
 *                  should be \f$ wgXdim*rows*sizeof\ (float) \f$.
 *  \param[in] data local buffer. Its size should be `2 float` elements for each 
 *                  work-item in a work-group. That is \f$ 2*lXdim*sizeof\ (float) \f$.
 *  \param[in] n number of elements in a row of the array divided by 4.
 */
kernel
void reduce_min_f (global float4 *in, global float *out, local float *data, uint n)
{
    // Workspace dimensions
    uint lXdim = get_local_size (0);
    uint wgXdim = get_num_groups (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    // Load 8 float elements per work-item
    uint idx = gX << 1;
    int4 flag = (uint4) (idx) < (uint4) (n);
    float4 a = select ((float4) (INFINITY), in[gY * n + idx], flag);
    flag = (uint4) (idx + 1) < (uint4) (n);
    float4 b = select ((float4) (INFINITY), in[gY * n + idx + 1], flag);

    // Perform a serial min operation
    // on the 2 float4 elements
    float2 ta = fmin (a.lo, a.hi);
    float2 tb = fmin (b.lo, b.hi);

    // Store the min of each float4 element
    data[(lX << 1)] = fmin (ta.x, ta.y);
    data[(lX << 1) + 1] = fmin (tb.x, tb.y);

    // Reduce
    for (uint d = lXdim; d > 0; d >>= 1)
    {
        barrier (CLK_LOCAL_MEM_FENCE);
        if (lX < d)
            data[lX] = fmin (data[lX], data[lX + d]);
    }

    // One work-item per work-group 
    // stores the minimum element
    if (lX == 0) 
        out[gY * wgXdim + wgX] = data[0];
}


/*! \brief Performs a reduce operation on the columns of an array.
 *  \details Computes the maximum element for each row in an array.
 *  \note When there are multiple rows in the array, a reduce operation 
 *        is performed per row, in parallel.
 *  \note The number of elements, `N`, in a row of the array should be a **multiple 
 *        of 4** (the data are handled as `uint4`). The **x** dimension of the 
 *        global workspace, \f$ gXdim \f$, should be greater than or equal to the number of 
 *        elements in a row of the array divided by 8. That is, \f$ \ gXdim \geq N/8 \f$. 
 *        Each work-item handles `8 uint` (= `2 uint4`) elements in a row of the array. 
 *        The **y** dimension of the global workspace, \f$ gYdim \f$, should be equal 
 *        to the number of rows, `M`, in the array. That is, \f$ \ gYdim = M \f$. 
 *        The local workspace should be `1` in the **y** dimension, and a 
 *        **power of 2** in the **x** dimension. It is recommended 
 *        to use one `wavefront/warp` per work-group.
 *  \note When the number of elements per row of the array is small enough to be 
 *        handled by a single work-group, the output array will contain the true 
 *        maximums. When the elements are more than that, they are partitioned 
 *        into blocks and reduced independently. In this case, the kernel 
 *        outputs the maximums from each block reduction. A reduction should then be
 *        made on those maximums for the final results. The number of work-groups
 *        in the **x** dimension, \f$ wgXdim \f$, **for the case of multiple 
 *        work-groups**, should be made a **multiple of 4**. The potential extra 
 *        work-groups are used for enforcing correctness. They write the necessary
 *        identity operands, `0`, in the output array, since in the next phase 
 *        the data are going to be handled as `uint4`.
 *
 *  \param[in] in input array of `uint` elements.
 *  \param[out] out (reduced) output array of `uint` elements. When the kernel is
 *                  dispatched with one work-group per row, the array contains the 
 *                  final results, and its size should be \f$ rows*sizeof\ (uint) \f$.
 *                  When the kernel is dispatched with more than one work-groups per row,
 *                  the array contains the results from each block reduction, and its size 
 *                  should be \f$ wgXdim*rows*sizeof\ (uint) \f$.
 *  \param[in] data local buffer. Its size should be `2 uint` elements for each 
 *                  work-item in a work-group. That is \f$ 2*lXdim*sizeof\ (uint) \f$.
 *  \param[in] n number of elements in a row of the array divided by 4.
 */
kernel
void reduce_max_ui (global uint4 *in, global uint *out, local uint *data, uint n)
{
    // Workspace dimensions
    uint lXdim = get_local_size (0);
    uint wgXdim = get_num_groups (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    // Load 8 float elements per work-item
    uint idx = gX << 1;
    int4 flag = (uint4) (idx) < (uint4) (n);
    uint4 a = select ((uint4) (0), in[gY * n + idx], flag);
    flag = (uint4) (idx + 1) < (uint4) (n);
    uint4 b = select ((uint4) (0), in[gY * n + idx + 1], flag);

    // Perform a serial max operation
    // on the 2 uint4 elements
    uint2 ta = select (a.hi, a.lo, a.lo > a.hi);
    uint2 tb = select (b.hi, b.lo, b.lo > b.hi);

    // Store the min of each uint4 element
    data[(lX << 1)] = select (ta.y, ta.x, ta.x > ta.y);
    data[(lX << 1) + 1] = select (tb.y, tb.x, tb.x > tb.y);

    // Reduce
    for (uint d = lXdim; d > 0; d >>= 1)
    {
        barrier (CLK_LOCAL_MEM_FENCE);
        if (lX < d)
            data[lX] = select (data[lX + d], data[lX], data[lX] > data[lX + d]);
    }

    // One work-item per work-group 
    // stores the maximum element
    if (lX == 0) 
        out[gY * wgXdim + wgX] = data[0];
}


/*! \brief Performs a reduce operation on the columns of an array.
 *  \details Computes the sum of the elements of each row in an array.
 *  \note When there are multiple rows in the array, a reduce operation 
 *        is performed per row, in parallel.
 *  \note The number of elements, `N`, in a row of the array should be a **multiple 
 *        of 4** (the data are handled as `float4`). The **x** dimension of the 
 *        global workspace, \f$ gXdim \f$, should be greater than or equal to the number of 
 *        elements in a row of the array divided by 8. That is, \f$ \ gXdim \geq N/8 \f$. 
 *        Each work-item handles `8 float` (= `2 float4`) elements in a row of the array. 
 *        The **y** dimension of the global workspace, \f$ gYdim \f$, should be equal 
 *        to the number of rows, `M`, in the array. That is, \f$ \ gYdim = M \f$. 
 *        The local workspace should be `1` in the **y** dimension, and a 
 *        **power of 2** in the **x** dimension. It is recommended 
 *        to use one `wavefront/warp` per work-group.
 *  \note When the number of elements per row of the array is small enough to 
 *        be handled by a single work-group, the output array will contain the 
 *        true sums. When the elements are more than that, they are partitioned 
 *        into blocks and reduced independently. In this case, the kernel 
 *        outputs the sum from each block reduction. A reduction should then be
 *        made on those sums for the final results. The number of work-groups
 *        in the **x** dimension, \f$ wgXdim \f$, **for the case of multiple 
 *        work-groups**, should be made a **multiple of 4**. The potential extra 
 *        work-groups are used for enforcing correctness. They write the necessary
 *        identity operands, `0.f`, in the output array, since in the next phase 
 *        the data are going to be handled as `float4`.
 *
 *  \param[in] in input array of `float` elements.
 *  \param[out] out (reduced) output array of `float` elements. When the kernel is
 *                  dispatched with one work-group per row, the array contains the 
 *                  final results, and its size should be \f$ rows*sizeof\ (float) \f$.
 *                  When the kernel is dispatched with more than one work-groups per row,
 *                  the array contains the results from each block reduction, and its size 
 *                  should be \f$ wgXdim*rows*sizeof\ (float) \f$.
 *  \param[in] data local buffer. Its size should be `2 float` elements for each 
 *                  work-item in a work-group. That is \f$ 2*lXdim*sizeof\ (float) \f$.
 *  \param[in] n number of elements in a row of the array divided by 4.
 */
kernel
void reduce_sum_f (global float4 *in, global float *out, local float *data, uint n)
{
    // Workspace dimensions
    uint lXdim = get_local_size (0);
    uint wgXdim = get_num_groups (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    // Load 8 float elements per work-item
    uint idx = gX << 1;
    int4 flag = (uint4) (idx) < (uint4) (n);
    float4 a = select ((float4) (0.f), in[gY * n + idx], flag);
    flag = (uint4) (idx + 1) < (uint4) (n);
    float4 b = select ((float4) (0.f), in[gY * n + idx + 1], flag);

    // Store the component sum of each float4 element
    data[(lX << 1)] = dot (a, (float4) (1.f));
    data[(lX << 1) + 1] = dot (b, (float4) (1.f));

    // Reduce
    for (uint d = lXdim; d > 0; d >>= 1)
    {
        barrier (CLK_LOCAL_MEM_FENCE);
        if (lX < d)
            data[lX] += data[lX + d];
    }

    // One work-item per work-group stores the sum
    if (lX == 0) 
        out[gY * wgXdim + wgX] = data[0];
}
