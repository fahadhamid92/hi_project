#find_library(MGL mgl PATHS /usr/local/lib)

ADD_EXECUTABLE( Hi_Project main.cpp findEyeCenter.cpp findEyeCenter.h findEyeCorner.cpp findEyeCorner.h constants.h helpers.cpp helpers.h )
TARGET_LINK_LIBRARIES( Hi_Project ${OpenCV_LIBS} )
#target_link_libraries( Hi_Project ${MGL})
