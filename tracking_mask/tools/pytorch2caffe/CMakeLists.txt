cmake_minimum_required (VERSION 3.5.1)
 
project (write_model)
 
add_executable(write_model wm.cpp)
 
include_directories ( /home/omnisky/programfiles/objectdetection/detector_ssd/include
    /usr/local/include
    /usr/local/cuda/include
    /usr/include )
 
target_link_libraries(write_model
    /home/omnisky/programfiles/objectdetection/detector_ssd/build/lib/libcaffe.so
    /usr/lib/x86_64-linux-gnu/libglog.so
    /usr/lib/x86_64-linux-gnu/libboost_system.so
    )

