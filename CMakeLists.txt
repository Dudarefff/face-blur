cmake_minimum_required(VERSION 3.16)
project(obs-face-blur VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(libobs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn)
if(OpenCV_VERSION VERSION_LESS 4.5.4)
  message(FATAL_ERROR "Нужен OpenCV >= 4.5.4 (модуль dnn, NMSBoxes), найден ${OpenCV_VERSION}")
endif()

add_library(${PROJECT_NAME} MODULE
  src/plugin-main.c
  src/face-blur-filter.cpp
  src/face-tracker.hpp
  src/yunet-detector.hpp)

set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE OBS::libobs ${OpenCV_LIBS})

if(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE /W3 /utf-8)
else()
  target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
endif()

# Готовая папка в формате пользовательских плагинов OBS:
#   dist/obs-face-blur/bin/64bit/<плагин>
#   dist/obs-face-blur/data/{effects,locale,models}
set(DIST_DIR "${CMAKE_BINARY_DIR}/dist/${PROJECT_NAME}")
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DIST_DIR}/bin/64bit"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:${PROJECT_NAME}>" "${DIST_DIR}/bin/64bit/"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/data" "${DIST_DIR}/data"
  COMMENT "Собираю папку плагина: ${DIST_DIR}")

# Windows: рядом с плагином нужны DLL OpenCV (но не obs.dll — она уже есть в OBS)
if(WIN32)
  set(_opencv_dlls "")
  foreach(_lib ${OpenCV_LIBS})
    if(TARGET ${_lib})
      get_target_property(_type ${_lib} TYPE)
      if(_type STREQUAL "SHARED_LIBRARY")
        list(APPEND _opencv_dlls "$<TARGET_FILE:${_lib}>")
      endif()
    endif()
  endforeach()
  if(_opencv_dlls)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_opencv_dlls} "${DIST_DIR}/bin/64bit/"
      COMMAND_EXPAND_LISTS
      COMMENT "Копирую DLL OpenCV")
  endif()
endif()
