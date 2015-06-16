find_package ( GLUT REQUIRED )
find_package ( GLEW REQUIRED )

include_directories ( 
    ${OPENGL_INCLUDE_DIRS} 
    ${GLUT_INCLUDE_DIRS} 
    ${GLEW_INCLUDE_DIRS} 
    ${CLUtils_INCLUDE_DIR} 
    ${EIGEN_INCLUDE_DIR} 
)

add_executable ( 
    ${FNAME}_step_by_step 
    step_by_step.cpp 
    ${PROJECT_SOURCE_DIR}/src/step_by_step/ocl_icp_sbs.cpp 
)

add_dependencies ( 
    ${FNAME}_step_by_step 
    CLUtils Eigen 
)

target_link_libraries ( 
    ${FNAME}_step_by_step 
    ${OPENGL_LIBRARIES} 
    ${GLUT_LIBRARY} 
    ${GLEW_LIBRARIES} 
    ${OPENCL_LIBRARIES} 
    ${CLUtils_LIBRARIES} 
    ICPAlgorithms 
)
