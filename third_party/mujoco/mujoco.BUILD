package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mujoco_lib",
    srcs = ["build/lib/libmujoco.so"],
    hdrs = glob(["include/mujoco/*.h"]),
    includes = [
        "include",
        "include/mujoco",
    ],
    linkopts = ["-Wl,-rpath,'$$ORIGIN'"],
    linkstatic = 0,
)

filegroup(
    name = "mujoco_so",
    srcs = ["build/lib/libmujoco.so"],
)
