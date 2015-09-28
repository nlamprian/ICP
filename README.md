# ICP
`ICP` is an implementation of the **Photogeometric Iterative Closest Point (ICP)** algorithm in OpenCL. **ICP** performs real-time frame-to-frame 3-D registration, utilizing both the **geometry** and the **texture** of a scene. This way, the geometric features can guide the registration process in situations where there are faintly textured regions, and the color can be used in those places with non-salient surface topology. The implemented framework is based on the paper [Real-time RGB-D mapping and 3-D modeling on the GPU using the random ball cover data structure](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=6130381&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D6130381) by Neumann et al.

![cover](http://i76.photobucket.com/albums/j16/paign10/icp_step_by_step_zps7pvd5zhn.gif)

Currently, there are two options for the rotation estimation step. One that uses **rotation matrices** and estimates the rotation by performing **Singular Value Decomposition** on the `CPU`. The other uses **unit quaternions** and estimates the rotation based on the **Power Method**. The rest of the computational load is executed exclusively on the `GPU`. Both resulting pipelines are able to perform one **ICP iteration** in about `1.1 millisecond`, for input sets of `|F|=|M|=16384` landmarks and `|R|=256` representative points.

# Note
The project was developed and tested on `Ubuntu 14.04.2`, on a system with an `AMD R9 270X` GPU.

The complete `documentation` is available [here](http://icp.paign10.me).

For more details on the implemented algorithms, take a look at the project's [wiki](https://github.com/pAIgn10/ICP/wiki/Algorithms).

# Dependencies
The project has dependencies on the [CLUtils](https://github.com/pAIgn10/CLUtils), [GuidedFilter](https://github.com/pAIgn10/GuidedFilter), [RandomBallCover](https://github.com/pAIgn10/RandomBallCover), and [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) libraries.

All these dependencies are automatically downloaded by cmake, if they are not available on your system. Please note that you'll have to [configure Mercurial](http://eigen.tuxfamily.org/index.php?title=Mercurial) before Eigen is downloaded.

# Examples
There are three applications. `kinect_frame_grabber` that is able to capture point clouds from Kinect (you'll have to install [libfreenect](https://github.com/OpenKinect/libfreenect) to build it). `icp_step_by_step` visualizes the process of frame-to-frame registration, step by step. `icp_registration` presents how to set up the ICP pipeline.

# Compilation

```bash
git clone https://github.com/pAIgn10/ICP.git
cd ICP

mkdir build
cd build

cmake -DBUILD_EXAMPLES=ON ..
# or to build the tests too
cmake -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON ..

make

# optionally, capture a pair of point clouds
./bin/kinect_frame_grabber -f -s left
./bin/kinect_frame_grabber -f -s right
# to run the examples (from the build directory!)
./bin/icp_step_by_step
# or if you recorded your own pair of point clouds
./bin/icp_step_by_step kg_pc8d_left kg_pc8d_right

# to run the tests (e.g.)
./bin/icp_tests_icp
# or with profiling information
./bin/icp_tests_icp --profiling

# to install the libraries
sudo make install
# you'll need to copy manually the kernel 
# files into your own projects

# to build the docs
make doxygen
firefox docs/html/index.html
```
