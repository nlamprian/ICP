/*! \file icp_kernels.cl
 *  \brief Kernels for the `%ICP` pipeline.
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


/*! \brief Struct holding a value and a key.
 *  \details Data structure used during the reduction phase of the distances 
 *           in the array produced by a `rbcComputeDists` kernel.
 */
typedef struct
{
    float dist;
    uint id;
} dist_id;


/*! \brief Samples a point cloud for landmarks.
 *  \details Chooses landmarks at specific intervals in the x and y dimension.
 *  \note 16384 landmarks are extracted from the point cloud \f$ (640 \times 480) \f$. 
 *        These landmarks come from a center area. 10% of the points around the 
 *        cloud are ignored.
 *  \note From the center area \f$ (512 \times 384) \f$, points are sampled `1:4` 
 *        in the x dimension and `1:3` in the y dimension. There is also an offset 
 *        `1` in the x dimension and 1 in the y dimension. This creates an array 
 *        \f$ (128 \times 128) \f$ of landmarks.
 *  \note Invalid points (zero coordinates) are going to be picked. Further 
 *        processing is needed for those points to be discraded, if necessary.
 *  \note The **x** dimension of the global workspace, \f$ gXdim \f$, should 
 *        be equal to two times the number of landmarks per row (2 work-items 
 *        per landmark). That is, \f$ \ gXdim=256 \f$. The **y** dimension of 
 *        the global workspace, \f$ gYdim \f$, should be equal to the number of 
 *        landmarks per column. That is, \f$ \ gYdim=128 \f$. There is no 
 *        requirement for the local workspace.
 *
 *  \param[in] in array (point cloud) of `float8` elements.
 *  \param[out] out array (landmarks) of `float8` elements.
 */
kernel
void getLMs (global float4 *in, global float4 *out)
{
    // Workspace dimensions
    uint gXdim = get_global_size (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);

    uint xi = (((gX >> 1) << 1) << 2) + 2;
    uint yi = gY * 3 + 1;

    out[gY * gXdim + gX] = in[(48 + yi) * 1280 + (128 + xi) + gX % 2];
}


/*! \brief Samples a set of landmarks for representatives.
 *  \details Chooses representatives at specific intervals in the x and y dimension.
 *  \note Representatives are extracted from the set of landmarks \f$ (128 \times 128) \f$. 
 *  \note Representatives are sampled \f$ 1:(128/n_{rx}) \f$ in the x dimension
 *        and \f$ 1:(128/n_{ry}) \f$ in the y dimension. There is also an offset 
 *        \f$ (128/n_{rx})/2-1 \f$ in the x dimension and \f$ (128/n_{ry})/2-1 \f$ 
 *        in the y dimension. This creates an array \f$ (n_{rx} \times n_{ry}) \f$ 
 *        of representatives.
 *  \note The **x** dimension of the global workspace, \f$ gXdim \f$, should be 
 *        equal to the number of representatives per row. That is, \f$ \ gXdim=n_{rx }\f$. 
 *        The **y** dimension of the global workspace, \f$ gYdim \f$, should be 
 *        equal to the number of representatives per column. That is, \f$ \ gYdim=n_{ry} \f$. 
 *        There is no requirement for the local workspace.
 *
 *  \param[in] in array (landmarks) of `float8` elements.
 *  \param[out] out array (representatives) of `float8` elements.
 */
kernel
void getReps (global float8 *in, global float8 *out)
{
    // Workspace dimensions
    uint gXdim = get_global_size (0);
    uint gYdim = get_global_size (1);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);

    uint stepX = 128 / gXdim;
    uint stepY = 128 / gYdim;

    uint xi = gX * stepX + (stepX >> 1) - 1;
    uint yi = gY * stepY + (stepY >> 1) - 1;

    out[gY * gXdim + gX] = in[yi * 128 + xi];
}


/*! \brief Computes a set of weights \f$ \{w_i = \frac{100}{100+\|x_i-x'_i\|_p}\} \f$, 
 *         and reduces them to get their sum, \f$ \sum^n_i{w_i} \f$.
 *  \details Takes distances between pairs of points and produces a set of weights.
 *  \note The number of elements in the input array, `n`, should be a **multiple of 2**.
 *        The global workspace should be one dimensional, and its **x** dimension, 
 *        \f$ gXdim \f$, should be greater than or equal to the number of elements 
 *        in the array, `n`, divided by 2. That is, \f$ \ gXdim \geq n/2 \f$. 
 *        Each work-item handles `2 dist_id` elements. The local workspace should 
 *        be one dimensional, and its **x** dimension should be a **power of 2**. 
 *        It is recommended to use one `wavefront/warp` per work-group.
 *  \note The kernel is aimed to be used when the number of elements in the array 
 *        is small enough to be handled by a single work-group. It promotes the 
 *        output data type to `double` so that a consistent API is maintained.
 *
 *  \param[in] in array of `dist_id` elements.
 *  \param[out] weights array with the computed weights (`float` elements).
 *  \param[out] sums (reduced) array with the sum. Its size should be \f$ sizeof\ (double) \f$.
 *  \param[in] data local buffer. Its size should be `2 float` elements for each 
 *                  work-item in a work-group. That is \f$ 2*lXdim*sizeof\ (float) \f$.
 *  \param[in] n number of elements in the array.
 */
kernel
void icpComputeReduceWeights (global dist_id *in, global float *weights, global double *sums, 
                              local float *data, uint n)
{
    // Workspace dimensions
    uint lXdim = get_local_size (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    // Load 2 dist_id elements per work-item
    float a = 0.f;
    float b = 0.f;

    // Compute weights
    uint idx = gX << 1;
    if (idx < n)
    {
        a = 100.f / (100.f + in[idx].dist);
        weights[idx] = a;

        b = 100.f / (100.f + in[idx + 1].dist);
        weights[idx + 1] = b;
    }

    // Store weights
    data[(lX << 1)] = a;
    data[(lX << 1) + 1] = b;

    // Reduce
    for (uint d = lXdim; d > 0; d >>= 1)
    {
        barrier (CLK_LOCAL_MEM_FENCE);
        if (lX < d)
            data[lX] += data[lX + d];
    }

    // One work-item per work-group stores the sum
    if (lX == 0) 
        sums[wgX] = convert_double (data[0]);
}


/*! \brief Computes a set of weights \f$ \{w_i = \frac{100}{100+\|x_i-x'_i\|_p}\} \f$, 
 *         and reduces them to get their sum, \f$ \sum^n_i{w_i} \f$.
 *  \details Takes distances between pairs of points and produces a set of weights.
 *  \note The number of elements in the input array, `n`, should be a **multiple of 2**.
 *        The global workspace should be one dimensional, and its **x** dimension, 
 *        \f$ gXdim \f$, should be greater than or equal to the number of elements 
 *        in the array, `n`, divided by 2. That is, \f$ \ gXdim \geq n/2 \f$. 
 *        Each work-item handles `2 dist_id` elements. The local workspace should 
 *        be one dimensional, and its **x** dimension should be a **power of 2**. 
 *        It is recommended to use one `wavefront/warp` per work-group.
 *  \note When the number of elements is small enough to be handled by a 
 *        single work-group, use `icpComputeReduceWeights` instead.
 *  \note When the elements are more than what a single work-group can handle, 
 *        they are partitioned into blocks and reduced independently. In this 
 *        case, the kernel outputs the sum from each block reduction. A reduction 
 *        should then be made (by `reduce_sum_fd`) on those sums for the final result. 
 *        The number of work-groups in the **x** dimension, \f$ wgXdim \f$, **for the 
 *        case of multiple work-groups**, should be made a **multiple of 4**. The 
 *        potential extra work-groups are used for enforcing correctness. They write 
 *        the necessary identity operands, `0.f`, in the output array, since in the 
 *        next phase the data are going to be handled as `float4`.
 *
 *  \param[in] in array of `dist_id` elements.
 *  \param[out] weights array with the computed weights (`float` elements).
 *  \param[out] sums (reduced) array with the sums. Its size should be \f$ wgXdim*sizeof\ (float) \f$.
 *  \param[in] data local buffer. Its size should be `2 float` elements for each 
 *                  work-item in a work-group. That is \f$ 2*lXdim*sizeof\ (float) \f$.
 *  \param[in] n number of elements in the array.
 */
kernel
void icpComputeReduceWeights_WG (global dist_id *in, global float *weights, global float *sums, 
                                 local float *data, uint n)
{
    // Workspace dimensions
    uint lXdim = get_local_size (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    // Load 2 dist_id elements per work-item
    float a = 0.f;
    float b = 0.f;

    // Compute weights
    uint idx = gX << 1;
    if (idx < n)
    {
        a = 100.f / (100.f + in[idx].dist);
        weights[idx] = a;

        b = 100.f / (100.f + in[idx + 1].dist);
        weights[idx + 1] = b;
    }

    // Store weights
    data[(lX << 1)] = a;
    data[(lX << 1) + 1] = b;

    // Reduce
    for (uint d = lXdim; d > 0; d >>= 1)
    {
        barrier (CLK_LOCAL_MEM_FENCE);
        if (lX < d)
            data[lX] += data[lX + d];
    }

    // One work-item per work-group stores the sum
    if (lX == 0) 
        sums[wgX] = data[0];
}


/*! \brief Performs a reduce operation on the columns of an array.
 *  \details Computes the sum of the elements of each row in an array.
 *  \note When there are multiple rows in the array, a reduce operation 
 *        is performed per row, in parallel.
 *  \note The number of elements, `N`, in a row of the array should be a **multiple 
 *        of 4** (the data are handled as `float4`). The **x** dimension of the 
 *        global workspace, \f$ gXdim \f$, should be greater or equal to the number of 
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
 *        identity operands, `0.0`, in the output array, since in the next phase 
 *        the data are going to be handled as `double4`.
 *
 *  \param[in] in array of `float` elements.
 *  \param[out] out (reduced) array of `double` elements. When the kernel is
 *                  dispatched with one work-group per row, the array contains the 
 *                  final results, and its size should be \f$ rows*sizeof\ (double) \f$.
 *                  When the kernel is dispatched with more than one work-groups per row,
 *                  the array contains the results from each block reduction, and its size 
 *                  should be \f$ wgXdim*rows*sizeof\ (double) \f$.
 *  \param[in] data local buffer. Its size should be `2 double` elements for each 
 *                  work-item in a work-group. That is \f$ 2*lXdim*sizeof\ (double) \f$.
 *  \param[in] n number of elements in a row of the array divided by 4.
 */
kernel
void reduce_sum_fd (global float4 *in, global double *out, local double *data, uint n)
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
    double4 a = convert_double4 (select ((float4) (0.f), in[gY * n + idx]    , flag));
    flag = (uint4) (idx + 1) < (uint4) (n);
    double4 b = convert_double4 (select ((float4) (0.f), in[gY * n + idx + 1], flag));

    // Store the component sum of each double4 element
    data[(lX << 1)]     = dot (a, (double4) (1.0));
    data[(lX << 1) + 1] = dot (b, (double4) (1.0));

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


/*! \brief Performs reduce operations on arrays of 8-D points.
 *  \details Computes the means \f$ \bar{x}_j = \sum^{n}_{i}{\frac{x_{ij}}{n}}, 
 *           j=\{0,1,2\} \f$ on the xyz dimensions of the points in the 
 *           fixed and moving sets.
 *  \note The kernel normalizes the data, and then performs the reduction.
 *  \note The number of elements, `n`, in the arrays should be a **multiple of 2** 
 *        (each work-item loads 2 points). The **x** dimension of the global 
 *        workspace, \f$ gXdim \f$, should be greater than or equal to the 
 *        number of points in the arrays divided by 2. That is, \f$ \ gXdim 
 *        \geq n/2 \f$. The **y** dimension of the global workspace, \f$ gYdim \f$, 
 *        should be equal to 2. That is, \f$ \ gYdim = 2 \f$. The local workspace 
 *        should also be one dimensional, and its **x** dimension should be a 
 *        **power of 2**. It is recommended to use one `wavefront/warp` per work-group.
 *  \note When the number of points in the arrays is small enough to be handled 
 *        by a single work-group, the output arrays will contain the true means. 
 *        When the points are more than that, they are partitioned into blocks 
 *        and reduced independently. In this case, the kernel outputs the means 
 *        from each block reduction. A reduction should then be made on those 
 *        means for the final result.
 *
 *  \param[in] F fixed set of `float8` elements. The first 3 dimensions 
 *               should contain the xyz coordinates of the points.
 *  \param[in] M moving set of `float8` elements. The first 3 dimensions 
 *               should contain the xyz coordinates of the points.
 *  \param[out] mean array of mean `float4` vectors. When the kernel is dispatched with 
 *                   just one work-group per row, the array contains one vector per set 
 *                   with the means on the xyz dimensions. Its size should be 
 *                   \f$ 2*sizeof\ (float4) \f$. The first `float4` is the mean for the 
 *                   fixed set, and the second `float4` is the mean for the moving set.
 *                   When the kernel is dispatched with more than one work-group, the 
 *                   array contains the means from each block reduction, and its size 
 *                   should be \f$ 2*(wgXdim*sizeof\ (float4)) \f$. The first row contains
 *                   the block means for the fixed set, and the second row contains the 
 *                   block means for the moving set.
 *  \param[in] data local buffer. Its size should be `6 float` elements for each work-item 
 *                  in a work-group. That is \f$ lXdim*(2*(3*sizeof\ (float))) \f$.
 *  \param[in] n number of points in the sets.
 */
kernel
void icpMean (global float4 *F, global float4 *M, global float4 *mean, local float *data, uint n)
{
    // Workspace dimensions
    uint lXdim = get_local_size (0);
    uint wgXdim = get_num_groups (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    // Choose input
    global float4 *SET[2] = { F, M };
    global float4 *in = SET[gY];

    // Fetch and normalize 2 points per work-item
    // Hold the xyz coordinates of the points
    uint idx = gX << 1;
    int3 flag = (uint3) (idx) < (uint3) (n);
    vstore3 (select ((float3) (0.f), in[ idx      << 1].xyz / (float3) (n), flag), (lX << 1)    , data);
    vstore3 (select ((float3) (0.f), in[(idx + 1) << 1].xyz / (float3) (n), flag), (lX << 1) + 1, data);

    // Reduce
    float3 a, b;
    for (uint d = lXdim; d > 0; d >>= 1)
    {
        barrier (CLK_LOCAL_MEM_FENCE);
        if (lX < d)
        {
            a = vload3 (lX, data);
            b = vload3 (lX + d, data);
            vstore3 (a + b, lX, data);
        }
    }

    // One work-item per work-group 
    // stores the mean vector
    if (lX == 0) 
        mean[gY * wgXdim + wgX] = (float4) (vload3 (0, data), 0.f);
}


/*! \brief Performs reduce operations on arrays of 8-D points.
 *  \details Computes the weighted means \f$ \bar{x}_j = \frac{\sum^{n}_{i}{w_i*x_{ij}}}
 *           {\sum^{n}_{i}{w_i}}, j=\{0,1,2\} \f$ on the xyz dimensions of the 
 *           points in the fixed and moving sets.
 *  \note The kernel normalizes the data, and then performs the reduction.
 *  \note The number of elements, `n`, in the arrays should be a **multiple of 2** 
 *        (each work-item loads 2 points). The **x** dimension of the global 
 *        workspace, \f$ gXdim \f$, should be greater than or equal to the 
 *        number of points in the arrays divided by 2. That is, \f$ \ gXdim 
 *        \geq n/2 \f$. The **y** dimension of the global workspace, \f$ gYdim \f$, 
 *        should be equal to 2. That is, \f$ \ gYdim = 2 \f$. The local workspace 
 *        should also be one dimensional, and its **x** dimension should be a 
 *        **power of 2**. It is recommended to use one `wavefront/warp` per work-group.
 *  \note When the number of points in the arrays is small enough to be handled 
 *        by a single work-group, the output arrays will contain the true means. 
 *        When the points are more than that, they are partitioned into blocks 
 *        and reduced independently. In this case, the kernel outputs the means 
 *        from each block reduction. A reduction should then be made on those 
 *        means for the final result.
 *
 *  \param[in] F fixed set of `float8` elements. The first 3 dimensions 
 *                should contain the xyz coordinates of the points.
 *  \param[in] M moving set of `float8` elements. The first 3 dimensions 
 *                should contain the xyz coordinates of the points.
 *  \param[out] MEAN array of mean `float4` vectors. When the kernel is dispatched with 
 *                   just one work-group per row, the array contains one vector per set 
 *                   with the means on the xyz dimensions. Its size should be 
 *                   \f$ 2*sizeof\ (float4) \f$. The first `float4` is the mean for the 
 *                   fixed set, and the second `float4` is the mean for the moving set.
 *                   When the kernel is dispatched with more than one work-group, the 
 *                   array contains the means from each block reduction, and its size 
 *                   should be \f$ 2*(wgXdim*sizeof\ (float4)) \f$. The first row contains
 *                   the block means for the fixed set, and the second row contains the 
 *                   block means for the moving set.
 *  \param[in] W array with the weights between the pairs of points.
 *  \param[in] sum_w sum of the weights.
 *  \param[in] data local buffer. Its size should be `6 float` elements for each work-item 
 *                  in a work-group. That is \f$ lXdim*(2*(3*sizeof\ (float))) \f$.
 *  \param[in] n number of points in the sets.
 */
kernel
void icpMean_Weighted (global float4 *F, global float4 *M, global float4 *MEAN, 
                       global float *W, constant double *sum_w, local float *data, uint n)
{
    // Workspace dimensions
    uint lXdim = get_local_size (0);
    uint wgXdim = get_num_groups (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    global float4 *SET[2] = { F, M };
    global float4 *in = SET[gY];

    // Fetch, weight and normalize 2 points per work-item
    // Hold the xyz coordinates of the points
    uint idx = gX << 1;
    int3 flag = (uint3) (idx) < (uint3) (n);
    vstore3 (select ((float3) (0.f), (float3) (W[idx]     / sum_w[0]) * in[ idx      << 1].xyz, flag), (lX << 1)    , data);
    vstore3 (select ((float3) (0.f), (float3) (W[idx + 1] / sum_w[0]) * in[(idx + 1) << 1].xyz, flag), (lX << 1) + 1, data);

    // Reduce
    float3 a, b;
    for (uint d = lXdim; d > 0; d >>= 1)
    {
        barrier (CLK_LOCAL_MEM_FENCE);
        if (lX < d)
        {
            a = vload3 (lX, data);
            b = vload3 (lX + d, data);
            vstore3 (a + b, lX, data);
        }
    }

    // One work-item per work-group 
    // stores the mean vector
    if (lX == 0)
        MEAN[gY * wgXdim + wgX] = (float4) (vload3 (0, data), 0.f);
}


/*! \brief Performs a reduce operation on an array of 4-D points.
 *  \details Computes the sums \f$ \bar{x}_j = \sum^{n}_{i}{x_{ij}}, 
 *           j=\{0,1,2\} \f$ on the xyz dimensions of the points 
 *           in the fixed and moving sets.
 *  \note This kernel is supposed to be used for the reduction of the 
 *        block means from `icpMean`.
 *  \note The global workspace should be one dimensional and its **x** dimension, 
 *        \f$ gXdim \f$, should be greater or equal to the number of points, `n`, 
 *        in the array divided by 2. That is, \f$ \ gXdim \geq n/2 \f$. The local 
 *        workspace should also be one dimensional, and its **x** dimension should be 
 *        a **power of 2**. It is recommended to use one `wavefront/warp` per work-group.
 *  \note When the number of points in the array is small enough to be handled 
 *        by a single work-group, the output array will contain the true means. 
 *        When the points are more than that, they are partitioned into blocks 
 *        and reduced independently. In this case, the kernel outputs the means 
 *        from each block reduction. A reduction should then be made on those 
 *        means for the final result.
 *
 *  \param[in] in array of `float4` elements. The first 3 dimensions 
 *                should contain the xyz coordinates of the points.
 *  \param[out] out array of mean `float4` vectors. When the kernel is 
 *                  dispatched with just one work-group, the array contains  
 *                  one vector with the means on the xyz dimensions, and its 
 *                  size should be \f$ sizeof\ (float4) \f$. When the kernel is 
 *                  dispatched with more than one work-group, the array contains 
 *                  the means from each block reduction, and its size should be 
 *                  \f$ wgXdim*sizeof\ (float4) \f$.
 *  \param[in] data local buffer. Its size should be `6 float` elements for each 
 *                  work-item in a work-group. That is \f$ lXdim*(2*(3*sizeof\ (float))) \f$.
 *  \param[in] n number of points in the array.
 */
kernel
void icpGMean (global float4 *in, global float4 *out, local float *data, uint n)
{
    // Workspace dimensions
    uint lXdim = get_local_size (0);
    uint wgXdim = get_num_groups (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    // Load 2 block means per work-item
    uint idx = gX << 1;
    int3 flag = (uint3) (idx) < (uint3) (n);
    vstore3 (select ((float3) (0.f), in[gY * n + idx].xyz,     flag), (lX << 1),     data);
    flag = (uint3) (idx + 1) < (uint3) (n);
    vstore3 (select ((float3) (0.f), in[gY * n + idx + 1].xyz, flag), (lX << 1) + 1, data);

    // Reduce
    float3 a, b;
    for (uint d = lXdim; d > 0; d >>= 1)
    {
        barrier (CLK_LOCAL_MEM_FENCE);
        if (lX < d)
        {
            a = vload3 (lX, data);
            b = vload3 (lX + d, data);
            vstore3 (a + b, lX, data);
        }
    }

    // One work-item per work-group 
    // stores the mean vector
    if (lX == 0) 
        out[gY * wgXdim + wgX] = (float4) (vload3 (0, data), 0.f);
}


/*! \brief Computes the deviations from the means of the fixed and moving sets of 8-D points.
 *  \details Subtracts the means from the 4-D geometric coordinates of the points.
 *  \note The **x** dimension of the global workspace, \f$ gXdim \f$, should be equal 
 *        to the number of points in the sets. That is, \f$ \ gXdim=n \f$. The **y** 
 *        dimension of the global workspace should be equal to 2. That is, \f$ \ gYdim=2 \f$. 
 *        There is no requirement for the local workspace.
 *
 *  \param[in] F fixed set of `float8` elements. The first 3 dimensions 
 *               should contain the xyz coordinates of the points.
 *  \param[in] M moving set of `float8` elements. The first 3 dimensions 
 *               should contain the xyz coordinates of the points.
 *  \param[out] DF array of `float4` elements (fixed set deviations from the mean). 
 *                 Only the geometric information gets transfered in the output.
 *  \param[out] DM array of `float4` elements (moving set deviations from the mean). 
 *                 Only the geometric information gets transfered in the output.
 *  \param[in] mean fixed and moving set means. The first `float4` is the fixed set 
 *                  mean, amd the second `float` is the moving set mean.
 */
kernel
void icpSubtractMean (global float4 *F, global float4 *M, global float4 *DF, global float4 *DM, 
                      constant float4 *mean)
{
    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);

    global float4 *SET[2] = { F, M };
    global float4 *DEV[2] = { DF, DM };
    global float4 *in = SET[gY];
    global float4 *out = DEV[gY];

    // Compute deviation from mean
    out[gX] = in[gX * 2] - mean[gY];
}


/*! \brief Produces the products in the Sij elements of the S matrix.
 *  \details Multiplies the deviations of corresponding points in the 
 *           fixed set \f$ \mathcal{F}_{m \times 3} \f$ and the moving set 
 *           \f$ \mathcal{M}_{m \times 3} \f$.
 *  \note Since large sums of built, there is an option to scale the points in 
 *        order to deal with floating point arithmetic issues. The resulting S 
 *        matrix will have accordingly scaled eigenvalues. The eigenvectors stay 
 *        the same. The constituents of the scaling factor s are scaled as well, 
 *        but the factor itself is not affected.
 *  \note Each work-item processes up to 4 pairs of points. So, the output 
 *        of each work-item is partial sums of products. On a next step, these 
 *        results will have to be reduced to get the final S matrix.
 *  \note The global workspace should be one dimensional and its **x** dimension, 
 *        \f$ gXdim \f$, should be greater or equal to the number of points, `m`, 
 *        in the sets. That is, \f$ \ gXdim \geq m \f$. There is no requirement 
 *        for the local workspace.
 *
 *  \param[in] M array (moving set deviations) of `float4` elements. The first 
 *               3 dimensions should contain the xyz coordinates of the points.
 *  \param[in] F array (fixed set deviations) of `float4` elements. The first 
 *               3 dimensions should contain the xyz coordinates of the points.
 *  \param[out] Sij array (partial sums of products). Its number of rows is `11`. 
 *                  Its number of columns is \f$ gXdim \f$. So, its size should be 
 *                  \f$ 11 * gXdim * sizeof\ (float) \f$.
 *  \param[in] m number of points in the sets.
 *  \param[in] c scaling factor. 
 */
kernel
void icpSijProducts (global float4 *M, global float4 *F, global float *Sij, uint m, float c)
{
    // Workspace dimensions
    uint gXdim = get_global_size (0);

    // Workspace indices
    uint gX = get_global_id (0);

    float3 A[4];
    A[0] = (float3) (0.f);  // mx * [fx fy fz]
    A[1] = (float3) (0.f);  // my * [fx fy fz]
    A[2] = (float3) (0.f);  // mz * [fx fy fz]
    A[3] = (float3) (0.f);  // [fp'*fp mp'*mp 0], used for calculating scale s

    for (uint pi = gX; pi < m; pi += gXdim)
    {
        int4 flag = (uint4) (pi) < (uint4) (m);
        float4 Mp = (float4) c * select ((float4) (0.f), M[pi], flag);
        float4 Fp = (float4) c * select ((float4) (0.f), F[pi], flag);

        A[0] += Mp.xxx * Fp.xyz;
        A[1] += Mp.yyy * Fp.xyz;
        A[2] += Mp.zzz * Fp.xyz;
        A[3] += (float3) (dot (Fp.xyz, Fp.xyz), dot (Mp.xyz, Mp.xyz), 0.f);
    }

    uint idx = gX;
    Sij[idx] = A[0].x; idx += gXdim;  // mx * fx
    Sij[idx] = A[0].y; idx += gXdim;  // mx * fy
    Sij[idx] = A[0].z; idx += gXdim;  // mx * fz
    Sij[idx] = A[1].x; idx += gXdim;  // my * fx
    Sij[idx] = A[1].y; idx += gXdim;  // my * fy
    Sij[idx] = A[1].z; idx += gXdim;  // my * fz
    Sij[idx] = A[2].x; idx += gXdim;  // mz * fx
    Sij[idx] = A[2].y; idx += gXdim;  // mz * fy
    Sij[idx] = A[2].z; idx += gXdim;  // mz * fz
    Sij[idx] = A[3].x; idx += gXdim;  // fp' * fp
    Sij[idx] = A[3].y;                // mp' * mp
}


/*! \brief Produces the weighted products in the Sij elements of the S matrix.
 *  \details Multiplies the deviations of corresponding points in the 
 *           fixed set \f$ \mathcal{F}_{m \times 3} \f$ and the moving set 
 *           \f$ \mathcal{M}_{m \times 3} \f$.
 *  \note Since large sums of built, there is an option to scale the points in 
 *        order to deal with floating point arithmetic issues. The resulting S 
 *        matrix will have accordingly scaled eigenvalues. The eigenvectors stay 
 *        the same. The constituents of the scaling factor s are scaled as well, 
 *        but the factor itself is not affected.
 *  \note Each work-item processes up to 4 pairs of points. So, the output 
 *        of each work-item is partial sums of products. On a next step, these 
 *        results will have to be reduced to get the final S matrix.
 *  \note The global workspace should be one dimensional and its **x** dimension, 
 *        \f$ gXdim \f$, should be greater or equal to the number of points, `m`, 
 *        in the sets. That is, \f$ \ gXdim \geq m \f$. There is no requirement 
 *        for the local workspace.
 *
 *  \param[in] M array (moving set deviations) of `float4` elements. The first 
 *               3 dimensions should contain the xyz coordinates of the points.
 *  \param[in] F array (fixed set deviations) of `float4` elements. The first 
 *               3 dimensions should contain the xyz coordinates of the points.
 *  \param[in] W array (weights) of `float` elements.
 *  \param[out] Sij array (partial sums of products). Its number of rows is `11`. 
 *                  Its number of columns is \f$ gXdim \f$. So, its size should be 
 *                  \f$ 11 * gXdim * sizeof\ (float) \f$.
 *  \param[in] m number of points in the sets.
 *  \param[in] c scaling factor. 
 */
kernel
void icpSijProducts_Weighted (global float4 *M, global float4 *F, global float *W, 
                              global float *Sij, uint m, float c)
{
    // Workspace dimensions
    uint gXdim = get_global_size (0);

    // Workspace indices
    uint gX = get_global_id (0);

    float3 A[4];
    A[0] = (float3) (0.f);  // mx * [fx fy fz]
    A[1] = (float3) (0.f);  // my * [fx fy fz]
    A[2] = (float3) (0.f);  // mz * [fx fy fz]
    A[3] = (float3) (0.f);  // [fp'*fp mp'*mp 0], used for calculating scale s

    for (uint pi = gX; pi < m; pi += gXdim)
    {
        int4 flag = (uint4) (pi) < (uint4) (m);
        float4 Mp = (float4) c * select ((float4) (0.f), M[pi], flag);
        float4 Fp = (float4) c * select ((float4) (0.f), F[pi], flag);
        float w = select (0.f, W[pi], flag.x);

        A[0] += (float3) w * (Mp.xxx * Fp.xyz);
        A[1] += (float3) w * (Mp.yyy * Fp.xyz);
        A[2] += (float3) w * (Mp.zzz * Fp.xyz);
        A[3] += (float3) (w * dot (Fp.xyz, Fp.xyz), w * dot (Mp.xyz, Mp.xyz), 0.f);
    }

    uint idx = gX;
    Sij[idx] = A[0].x; idx += gXdim;  // mx * fx
    Sij[idx] = A[0].y; idx += gXdim;  // mx * fy
    Sij[idx] = A[0].z; idx += gXdim;  // mx * fz
    Sij[idx] = A[1].x; idx += gXdim;  // my * fx
    Sij[idx] = A[1].y; idx += gXdim;  // my * fy
    Sij[idx] = A[1].z; idx += gXdim;  // my * fz
    Sij[idx] = A[2].x; idx += gXdim;  // mz * fx
    Sij[idx] = A[2].y; idx += gXdim;  // mz * fy
    Sij[idx] = A[2].z; idx += gXdim;  // mz * fz
    Sij[idx] = A[3].x; idx += gXdim;  // fp' * fp
    Sij[idx] = A[3].y;                // mp' * mp
}


/*! \brief Performs a homogeneous transformation on a set of points, \f$ p = 
 *         \left[ \begin{matrix} p_x & p_y & p_z & 1 \end{matrix} \right]^T \f$ 
 *         (or as a quaternion, \f$ \dot{p} = \left[ \begin{matrix} p_x & p_y & 
 *         p_z & 0 \end{matrix} \right]^T \f$), using unit quaternions \f$ \dot{q} 
 *         = (\omega, \mathcal{v}) = q_w + (q_x i + q_y j + q_z k) = \left[ 
 *         \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$, 
 *         where \f$ \dot{q}\cdot\dot{q}=1 \f$.
 *  \details Transforms each point in a set, \f$\ p' = s\dot{q}\dot{p}\dot{q}^*+t = 
 *           s(p + 2\mathcal{v} \times (\mathcal{v} \times p + \omega p)) + t \f$. 
 *  \note The **x** dimension of the global workspace, \f$ gXdim \f$, should be 
 *        equal to 2.  That is, \f$ \ gXdim = 2 \f$. The **y** dimension of the 
 *        global workspace should be equal to the number of points, `m`, in the 
 *        sets. That is, \f$ \ gYdim = m \f$. There is no requirement for the 
 *        local workspace.
 *
 *  \param[in] M array of `float8` elements. The first 4 dimensions should 
 *               contain the homogeneous coordinates of the points.
 *  \param[out] tM array of `float8` elements. The first 4 dimensions will 
 *                 contain the transformed homogeneous coordinates of the points.
 *  \param[in] data array of size \f$ 2 * sizeof\ (float4) \f$. The first `float4` 
 *                  is the **quaternion**, and the second is the **translation vector**. 
 *                  If there is a need to apply **scaling**, the factor should be available 
 *                  in the last element of the translation vector. That is, 
 *                  \f$ t = \left[ \begin{matrix} t_x & t_y & t_z & s \end{matrix} \right]^T \f$.
 */
kernel
void icpTransform_Quaternion (global float4 *M, global float4 *tM, constant float4 *data)
{
    // Workspace dimensions
    uint gXdim = get_global_size (0);
    uint gYdim = get_global_size (1);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);

    // Flatten indices
    uint idx = gY * gXdim + gX;

    // Rearrange indices
    uint gX1 = idx / gYdim;
    uint gY1 = idx % gYdim;

    float4 tp = M[gY1 * gXdim + gX1];

    if (gX1 == 0)
    {
        float4 q = data[0];
        float4 t = data[1];

        float3 p = tp.xyz;
        
        tp.xyz = t.w * (p + cross (2 * q.xyz, cross (q.xyz, p) + q.w * p)) + t.xyz;
    }

    tM[gY1 * gXdim + gX1] = tp;
}


/*! \brief Performs a homogeneous transformation on a set of points, \f$ p = 
 *         \left[ \begin{matrix} p_x & p_y & p_z & 1 \end{matrix} \right]^T \f$ 
 *         (or as a quaternion, \f$ \dot{p} = \left[ \begin{matrix} p_x & p_y & 
 *         p_z & 0 \end{matrix} \right]^T \f$), using unit quaternions \f$ \dot{q} 
 *         = q_w + q_x i + q_y j + q_z k = \left[ \begin{matrix} q_x & q_y & q_z & 
 *         q_w \end{matrix} \right]^T \f$, where \f$ \dot{q}\cdot\dot{q}=1 \f$.
 *  \details Transforms each point in a set, \f$\ p' = s\dot{q}\dot{p}\dot{q}^*+t = 
 *           s\bar{Q}^TQ\dot{p}+t = s 
 *           \left[ \begin{matrix}  q_w & q_z & -q_y & q_x \\ -q_z & q_w & q_x & q_y 
 *           \\ q_y & -q_x & q_w & q_z \\ -q_x & -q_y & -q_z & q_w \end{matrix} \right]^T
 *           \left[ \begin{matrix} q_w & -q_z & q_y & q_x \\ q_z & q_w & -q_x & q_y 
 *           \\ -q_y & q_x & q_w & q_z \\ -q_x & -q_y & -q_z & q_w \end{matrix} \right]
 *           \left[ \begin{matrix} p_x \\ p_y \\ p_z \\ 0 \end{matrix} \right] + 
 *           \left[ \begin{matrix} t_x \\ t_y \\ t_z \\ 1 \end{matrix} \right] = s
 *           \left[ \begin{matrix} 1-2q_y^2-2q_z^2 & 2(q_xq_y-q_zq_w) & 
 *           2(q_xq_z+q_yq_w) & 0 \\ 2(q_xq_y+q_zq_w) & 1-2q_x^2-2q_z^2 & 
 *           2(q_yq_z-q_xq_w) & 0 \\ 2(q_xq_z-q_yq_w) & 2(q_yq_z+q_xq_w) & 
 *           1-2q_x^2-2q_y^2 & 0 \\ 0 & 0 & 0 & 1 \end{matrix} \right] 
 *           \left[ \begin{matrix} p_x \\ p_y \\ p_z \\ 0 \end{matrix} \right] + 
 *           \left[ \begin{matrix} t_x \\ t_y \\ t_z \\ 1 \end{matrix} \right]\f$. 
 *  \note The **x** dimension of the global workspace, \f$ gXdim \f$, should be 
 *        equal to 2.  That is, \f$ \ gXdim = 2 \f$. The **y** dimension of the 
 *        global workspace should be equal to the number of points, `m`, in the 
 *        sets. That is, \f$ \ gYdim = m \f$. There is no requirement for the 
 *        local workspace.
 *
 *  \param[in] M array of `float8` elements. The first 4 dimensions should 
 *               contain the homogeneous coordinates of the points.
 *  \param[out] tM array of `float8` elements. The first 4 dimensions will 
 *                 contain the transformed homogeneous coordinates of the points.
 *  \param[in] data array of size \f$ 2 * sizeof\ (float4) \f$. The first `float4` 
 *                  is the **quaternion**, and the second is the **translation vector**. 
 *                  If there is a need to apply **scaling**, the factor should be available 
 *                  in the last element of the translation vector. That is, 
 *                  \f$ t = \left[ \begin{matrix} t_x & t_y & t_z & s \end{matrix} \right]^T \f$.
 */
kernel
void icpTransform_Quaternion_2 (global float4 *M, global float4 *tM, constant float4 *data)
{
    // Workspace dimensions
    uint gXdim = get_global_size (0);
    uint gYdim = get_global_size (1);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);

    // Flatten indices
    uint idx = gY * gXdim + gX;

    // Rearrange indices
    uint gX1 = idx / gYdim;
    uint gY1 = idx % gYdim;

    float4 tp = M[gY1 * gXdim + gX1];

    if (gX1 == 0)
    {
        float4 q = data[0];
        float4 t = data[1];

        float4 p = (float4) (tp.xyz, 0.f);
        
        float4 p_ = (float4) (dot ((float4) ( q.w, -q.z,  q.y, q.x), p), 
                              dot ((float4) ( q.z,  q.w, -q.x, q.y), p), 
                              dot ((float4) (-q.y,  q.x,  q.w, q.z), p), 
                              dot ((float4) (-q.x, -q.y, -q.z, q.w), p));

        tp.x = t.w * dot ((float4) ( q.w, -q.z,  q.y, -q.x), p_) + t.x;
        tp.y = t.w * dot ((float4) ( q.z,  q.w, -q.x, -q.y), p_) + t.y;
        tp.z = t.w * dot ((float4) (-q.y,  q.x,  q.w, -q.z), p_) + t.z;
    }

    tM[gY1 * gXdim + gX1] = tp;
}


/*! \brief Performs a homogeneous transformation on a set of points.
 *  \details Transforms each point in a set, \f$ p' = Tp = \left[ \begin{matrix} 
 *           R' & t \\ 0 & 1 \end{matrix} \right]\left[ \begin{matrix} 
 *           p \\ 1 \end{matrix} \right] = \left[ \begin{matrix} 
 *           sR & t \\ 0 & 1 \end{matrix} \right]\left[ \begin{matrix} 
 *           p \\ 1 \end{matrix} \right] = sRp+t \f$.
 *  \note It there is a need for **scaling**, the scaling factor should already 
 *        be incorporated in the **rotation matrix**.
 *  \note The **x** dimension of the global workspace, \f$ gXdim \f$, should be 
 *        equal to 2.  That is, \f$ \ gXdim = 2 \f$. The **y** dimension of the 
 *        global workspace should be equal to the number of points, `m`, in the 
 *        sets. That is, \f$ \ gYdim = m \f$. There is no requirement for the 
 *        local workspace.
 *
 *  \param[in] M array of `float8` elements. The first 4 dimensions should 
 *               contain the homogeneous coordinates of the points.
 *  \param[out] tM array of `float8` elements. The first 4 dimensions will 
 *                 contain the transformed homogeneous coordinates of the points.
 *  \param[in] T the transformation matrix of size \f$ 16 * sizeof\ (float) \f$. 
 *               The elements should be laid out in row major order.
 */
kernel
void icpTransform_Matrix (global float4 *M, global float4 *tM, constant float4 *T)
{
    // Workspace dimensions
    uint gXdim = get_global_size (0);
    uint gYdim = get_global_size (1);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);

    // Flatten indices
    uint idx = gY * gXdim + gX;

    // Rearrange indices
    uint gX1 = idx / gYdim;
    uint gY1 = idx % gYdim;

    float4 tp = M[gY1 * gXdim + gX1];

    if (gX1 == 0)
    {
        float4 p = tp;

        tp.x = dot (T[0], p);
        tp.y = dot (T[1], p);
        tp.z = dot (T[2], p);
    }

    tM[gY1 * gXdim + gX1] = tp;
}


/*! \brief Computes a matrix-vector product, \f$ x_{new}=Nx \f$.
 *
 *  \param[in] N `4x4` matrix (`4xfloat4` elements).
 *  \param[in] x vector (`float4` element).
 *  \param[out] x_new vector (`float4` element).
 */
inline
void prod (float4 *N, float4 *x, float4 *x_new)
{
    (*x_new).x = dot (N[0], *x);
    (*x_new).y = dot (N[1], *x);
    (*x_new).z = dot (N[2], *x);
    (*x_new).w = dot (N[3], *x);
}


/*! \brief Computes the quantities that represent the incremental development 
 *         in the transformation estimation in iteration `k`.
 *  \details Uses the `Power Method` to estimate the unit quaternion \f$ q_k \f$ 
 *           that represents the rotation, and then computes the scale \f$ s_k \f$
 *           and translation \f$ t_k \f$.
 *  \note The Power Method is performed on \f$N\f$ and computes the eigenvector corresponding 
 *        to the maximum magnitude eigenvalue \f$ \mu \f$. The eigenvector of interest is the 
 *        one corresponding to the most positive eigenvalue \f$\lambda\f$. If \f$ \mu<0 \f$, 
 *        the algorithm is executed again on \f$ N'=N + |\lambda| I \f$. Then, the eigenvalue 
 *        \f$ \lambda \f$ is \f$ \mu' - \mu \f$. The corresponding eigenvector doesn't change.
 *  \note The kernel should be dispatched as a task (1 work-item).
 *
 *  \param[in] Sij array (sums of products) of size \f$11*sizeof\ (float)\f$.
 *                 The first `9` elements (in row major order) are the \f$S_k\f$ matrix, and 
 *                 the next `2` are the numerator and denominator of the scale \f$s_k\f$.
 *  \param[in] means array (fixed and moving set means) of size \f$2*sizeof\ (float4)\f$.
 *  \param[out] Tk array of size \f$ 2 * sizeof\ (float4) \f$. The first `float4` 
 *                 is the **unit quaternion** \f$ \dot{q_k} = q_w + q_x i + q_y j + q_z k = 
 *                 \left[ \begin{matrix} q_x & q_y & q_z & q_w \end{matrix} \right]^T \f$, 
 *                 and the second one is the **translation vector** \f$ t_k=\left[ \begin{matrix} 
 *                 t_x & t_y & t_z & 1 \end{matrix} \right]^T \f$. The scale is placed 
 *                 in the last element of the translation vector. That is, \f$ t_k = 
 *                 \left[ \begin{matrix} t_x & t_y & t_z & s_k \end{matrix} \right]^T \f$.
 */
kernel
void icpPowerMethod (global float *Sij, global float4 *means, global float4 *Tk)
{
    float Sxx = Sij[0];
    float Sxy = Sij[1];
    float Sxz = Sij[2];
    float Syx = Sij[3];
    float Syy = Sij[4];
    float Syz = Sij[5];
    float Szx = Sij[6];
    float Szy = Sij[7];
    float Szz = Sij[8];
    
    float sk = sqrt (Sij[9] / Sij[10]);

    prefetch (means, 2);

    float4 N[4] = 
    {
        (float4) (Sxx - Syy - Szz,         Sxy + Syx,         Szx + Sxz,       Syz - Szy), 
        (float4) (      Sxy + Syx, - Sxx + Syy - Szz,         Syz + Szy,       Szx - Sxz), 
        (float4) (      Szx + Sxz,         Syz + Szy, - Sxx - Syy + Szz,       Sxy - Syx), 
        (float4) (      Syz - Szy,         Szx - Sxz,         Sxy - Syx, Sxx + Syy + Szz) 
    };

    // Power Method ============================================================

    float4 x = (float4) (1.f);
    float4 x_new;

    // Parameters
    uint maxIter = 1000;
    float error, error_new;

    while (true)
    {
        for (uint iter = 0; iter < maxIter; ++iter)
        {
            prod (N, &x, &x_new);

            x_new = fast_normalize (x_new);

            error = error_new;
            if ((error_new = fast_distance (x, x_new)) == error) break;

            x = x_new;
        }

        float lambda = dot (N[0], x_new) / x_new.x;

        if (lambda < 0)
        {
            N[0].x -= lambda;
            N[1].y -= lambda;
            N[2].z -= lambda;
            N[3].w -= lambda;

            x = (float4) (1.f);
        }
        else
            break;
    }

    x = x_new;
    prod (N, &x, &x_new);
    x_new = normalize (x_new);

    // Transformation ==========================================================

    float4 qk = x_new;

    float3 mf = means[0].xyz;
    float3 mm = means[1].xyz;

    float3 tk = mf - sk * (mm + cross (2 * qk.xyz, cross (qk.xyz, mm) + qk.w * mm));

    Tk[0] = qk;
    Tk[1] = (float4) (tk, sk);
}
