# Install script for directory: C:/Users/gauta/Code/CMPE256-Group10/implicit-0.7.2/implicit

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Users/gauta/Code/CMPE256-Group10/implicit-0.7.2/_skbuild/win-amd64-3.14/cmake-install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/implicit" TYPE MODULE FILES "C:/Users/gauta/Code/CMPE256-Group10/implicit-0.7.2/_skbuild/win-amd64-3.14/cmake-build/implicit/Debug/_nearest_neighbours.cp314-win_amd64.pyd")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/implicit" TYPE MODULE FILES "C:/Users/gauta/Code/CMPE256-Group10/implicit-0.7.2/_skbuild/win-amd64-3.14/cmake-build/implicit/Release/_nearest_neighbours.cp314-win_amd64.pyd")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/implicit" TYPE MODULE FILES "C:/Users/gauta/Code/CMPE256-Group10/implicit-0.7.2/_skbuild/win-amd64-3.14/cmake-build/implicit/MinSizeRel/_nearest_neighbours.cp314-win_amd64.pyd")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/implicit" TYPE MODULE FILES "C:/Users/gauta/Code/CMPE256-Group10/implicit-0.7.2/_skbuild/win-amd64-3.14/cmake-build/implicit/RelWithDebInfo/_nearest_neighbours.cp314-win_amd64.pyd")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/implicit" TYPE MODULE FILES "C:/Users/gauta/Code/CMPE256-Group10/implicit-0.7.2/_skbuild/win-amd64-3.14/cmake-build/implicit/Debug/evaluation.cp314-win_amd64.pyd")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/implicit" TYPE MODULE FILES "C:/Users/gauta/Code/CMPE256-Group10/implicit-0.7.2/_skbuild/win-amd64-3.14/cmake-build/implicit/Release/evaluation.cp314-win_amd64.pyd")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/implicit" TYPE MODULE FILES "C:/Users/gauta/Code/CMPE256-Group10/implicit-0.7.2/_skbuild/win-amd64-3.14/cmake-build/implicit/MinSizeRel/evaluation.cp314-win_amd64.pyd")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/implicit" TYPE MODULE FILES "C:/Users/gauta/Code/CMPE256-Group10/implicit-0.7.2/_skbuild/win-amd64-3.14/cmake-build/implicit/RelWithDebInfo/evaluation.cp314-win_amd64.pyd")
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/gauta/Code/CMPE256-Group10/implicit-0.7.2/_skbuild/win-amd64-3.14/cmake-build/implicit/cpu/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/gauta/Code/CMPE256-Group10/implicit-0.7.2/_skbuild/win-amd64-3.14/cmake-build/implicit/gpu/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/implicit" TYPE FILE FILES
    "C:/Users/gauta/Code/CMPE256-Group10/implicit-0.7.2/implicit/__init__.py"
    "C:/Users/gauta/Code/CMPE256-Group10/implicit-0.7.2/implicit/als.py"
    "C:/Users/gauta/Code/CMPE256-Group10/implicit-0.7.2/implicit/approximate_als.py"
    "C:/Users/gauta/Code/CMPE256-Group10/implicit-0.7.2/implicit/bpr.py"
    "C:/Users/gauta/Code/CMPE256-Group10/implicit-0.7.2/implicit/lmf.py"
    "C:/Users/gauta/Code/CMPE256-Group10/implicit-0.7.2/implicit/nearest_neighbours.py"
    "C:/Users/gauta/Code/CMPE256-Group10/implicit-0.7.2/implicit/recommender_base.py"
    "C:/Users/gauta/Code/CMPE256-Group10/implicit-0.7.2/implicit/utils.py"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/gauta/Code/CMPE256-Group10/implicit-0.7.2/_skbuild/win-amd64-3.14/cmake-build/implicit/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
