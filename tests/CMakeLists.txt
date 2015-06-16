if ( BUILD_TESTS )

    find_package ( Threads REQUIRED )

    include_directories ( ${CLUtils_INCLUDE_DIR} 
                          ${GTEST_INCLUDE_DIRS}
                          ${OPENGL_INCLUDE_DIRS} )

    add_executable ( ${FNAME}_tests_reduce testsReduce.cpp )
    add_executable ( ${FNAME}_tests_scan testsScan.cpp )
    add_executable ( ${FNAME}_tests_icp testsICP.cpp )

    add_dependencies ( ${FNAME}_tests_reduce CLUtils googletest )
    add_dependencies ( ${FNAME}_tests_scan CLUtils googletest )
    add_dependencies ( ${FNAME}_tests_icp CLUtils googletest )

    target_link_libraries ( ${FNAME}_tests_reduce LINK_PUBLIC ${CLUtils_LIBRARIES} 
                                                              ICPHelperFuncs 
                                                              ICPAlgorithms
                                                              ${OPENGL_LIBRARIES}
                                                              ${OPENCL_LIBRARIES}
                                                              ${GTEST_BOTH_LIBRARIES}
                                                              ${CMAKE_THREAD_LIBS_INIT} )

    target_link_libraries ( ${FNAME}_tests_scan LINK_PUBLIC ${CLUtils_LIBRARIES} 
                                                            ICPHelperFuncs 
                                                            ICPAlgorithms
                                                            ${OPENGL_LIBRARIES}
                                                            ${OPENCL_LIBRARIES}
                                                            ${GTEST_BOTH_LIBRARIES}
                                                            ${CMAKE_THREAD_LIBS_INIT} )

    target_link_libraries ( ${FNAME}_tests_icp LINK_PUBLIC ${CLUtils_LIBRARIES} 
                                                           ICPHelperFuncs 
                                                           ICPAlgorithms
                                                           ${OPENGL_LIBRARIES}
                                                           ${OPENCL_LIBRARIES}
                                                           ${GTEST_BOTH_LIBRARIES}
                                                           ${CMAKE_THREAD_LIBS_INIT} )

    add_test ( NAME ${FNAME}_tests_reduce 
               COMMAND ${EXECUTABLE_OUTPUT_PATH}/${FNAME}_tests_reduce 
               WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )

    add_test ( NAME ${FNAME}_tests_scan 
               COMMAND ${EXECUTABLE_OUTPUT_PATH}/${FNAME}_tests_scan 
               WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )

    add_test ( NAME ${FNAME}_tests_icp 
               COMMAND ${EXECUTABLE_OUTPUT_PATH}/${FNAME}_tests_icp 
               WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )


    add_custom_target ( check COMMAND ${CMAKE_CTEST_COMMAND} --verbose )

endif ( BUILD_TESTS )
