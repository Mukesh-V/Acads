FIND_PACKAGE(deal.II 9.4.0 REQUIRED
  HINTS
    /usr/local/bin ../ ../../ $ENV{DEAL_II_DIR}
    # You can specify additional hints for search paths here, e.g.
    # $ENV{HOME}/workspace/deal.II
)

# Set the name of the project and target:
SET(TARGET "first")

# Declare all source files the target consists of:
SET(TARGET_SRC
  first.cc
  # You can specify additional files here!
)

CMAKE_MINIMUM_REQUIRED(VERSION 3.3.0)
DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET} CXX)
DEAL_II_INVOKE_AUTOPILOT()