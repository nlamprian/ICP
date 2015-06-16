include_directories ( ${CLUtils_INCLUDE_DIR} 
                      ${RBC_INCLUDE_DIR}
                      ${EIGEN_INCLUDE_DIR} )

add_library ( ICPAlgorithms STATIC ICP/algorithms.cpp )
add_library ( ICPHelperFuncs STATIC ICP/tests/helper_funcs.cpp )

add_dependencies ( ICPAlgorithms  CLUtils RBC Eigen )
add_dependencies ( ICPHelperFuncs CLUtils RBC )

target_link_libraries ( ICPAlgorithms ${RBC_LIBRARIES} )

target_include_directories ( 
    ICPAlgorithms PUBLIC 
    ${COMMON_INCLUDES} 
)

target_include_directories ( 
    ICPHelperFuncs PUBLIC 
    ${COMMON_INCLUDES} 
)

install ( DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include )
install ( DIRECTORY ${PROJECT_BINARY_DIR}/lib/ DESTINATION lib/ICP )

# ==============================================================================

find_package ( Freenect QUIET )

if ( FREENECT_FOUND )

    find_package ( GLUT REQUIRED )
    find_package ( GLEW REQUIRED )
    find_package ( libusb-1.0 REQUIRED )
    find_package ( Threads REQUIRED )

    include_directories ( ${OPENGL_INCLUDE_DIRS} 
                          ${GLUT_INCLUDE_DIRS} 
                          ${GLEW_INCLUDE_DIRS} 
                          ${LIBUSB_1_INCLUDE_DIRS} 
                          ${FREENECT_INCLUDE_DIR} 
                          ${GuidedFilter_INCLUDE_DIR} )

    add_executable ( kinect_frame_grabber kinect_frame_grabber.cpp )

    add_dependencies ( kinect_frame_grabber CLUtils GuidedFilter )

    target_link_libraries ( kinect_frame_grabber ${OPENGL_LIBRARIES} 
                                                 ${GLUT_LIBRARY} 
                                                 ${GLEW_LIBRARIES} 
                                                 ${FREENECT_LIBRARIES} 
                                                 ${OPENCL_LIBRARIES} 
                                                 ${CMAKE_THREAD_LIBS_INIT} 
                                                 ${CLUtils_LIBRARIES} 
                                                 ${GuidedFilter_LIBRARIES} )

endif ( FREENECT_FOUND )
