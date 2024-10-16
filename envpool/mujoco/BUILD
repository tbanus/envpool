# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@pip_requirements//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

genrule(
    name = "gen_mujoco_gym_xml",
    srcs = ["@mujoco_gym_xml"],
    outs = ["assets_gym"],
    cmd = "mkdir -p $(OUTS) && cp -r $(SRCS) $(OUTS)",
)

genrule(
    name = "gen_mujoco_dmc_xml",
    srcs = ["@mujoco_dmc_xml"],
    outs = ["assets_dmc"],
    cmd = "mkdir -p $(OUTS) && cp -r $(SRCS) $(OUTS)",
)

genrule(
    name = "gen_mujoco_so",
    srcs = ["@mujoco//:mujoco_so"],
    outs = ["libmujoco.so.2.2.1"],
    cmd = "cp $< $@",
)

cc_library(
    name = "mujoco_gym_env",
    hdrs = glob([
        "opy-software/**/*.h",
        "opy-software/**/*.hh",
        "opy-software/**/*.hpp",
        "opy-software/**/*.cpp",
        "opy-software/**/*.ipp",
        "gym/ant.h",
        "gym/half_cheetah.h",
        "gym/hopper.h",
        "gym/humanoid.h",
        "gym/humanoid_standup.h",
        "gym/inverted_double_pendulum.h",
        "gym/inverted_pendulum.h",
        "gym/mujoco_env.h",
        "gym/pusher.h",
        "gym/reacher.h",
        "gym/swimmer.h",
        "gym/walker2d.h",
        ]),
    includes = [
        "opy-software",
        "opy-software/include",
        "opy-software/include/FSM",
        "opy-software/include/Utilities",
        "opy-software/include/Dynamics",
        "opy-software/include/Controllers",
        "opy-software/include/convexMPC",
        "opy-software/include/motor_driver",
        "opy-software/third_party",
        # "opy-software/third_party/osqp",
        # "opy-software/third_party/osqp/include",
        # "opy-software/third_party/osqp/lin_sys",
        # "opy-software/third_party/osqp/lin_sys/direct",
        # "opy-software/third_party/osqp/lin_sys/direct/qdldl",
        # "opy-software/third_party/osqp/tests",
        # "opy-software/third_party/osqp/lin_sys/direct/qdldl/amd/include",
        # "opy-software/third_party/osqp/lin_sys/direct/qdldl/qdldl_sources/include",
        # "opy-software/third_party/osqp/lin_sys/direct/pardiso",
        "opy-software/third_party/qpOASES/include/qpOASES",
        "opy-software/third_party/qpOASES/include",
        "opy-software/third_party/qpOASES",
        "opy-software/third_party/GoldFarb_Optimizer",
        "opy-software/third_party/ParamHandler",
        "opy-software/third_party/JCQP",
        "opy-software/third_party/vn100/include",
        "opy-software/third_party/JCQP/amd/include",
        "opy-software/lcm_types/cpp",
        "opy-software/third_party/lcm",
        
    ],
    srcs = glob([  
    "opy-software/src/**/*.cpp",
    "opy-software/src/FSM/*.cpp",
    "opy-software/src/*.cpp",
    "opy-software/src/Utilities/*.cpp",

    "opy-software/third_party/ParamHandler/**/*.cpp",
    "opy-software/third_party/qpOASES/**/*.hpp",
    "opy-software/third_party/qpOASES/**/*.ipp",
    "opy-software/third_party/Goldfarb_Optimizer/**/*.cpp",
    "opy-software/third_party/Goldfarb_Optimizer/**/*.cc",
        # "opy-software/third_party/vn100/**/*.cpp",

    "opy-software/third_party/ParamHandler/**/*.cpp",
    "opy-software/third_party/qpOASES/**/*.cpp",
    "opy-software/third_party/Goldfarb_Optimizer/**/*.cpp",
    "opy-software/third_party/Goldfarb_Optimizer/**/*.cc",

    "opy-software/third_party/ParamHandler/*.cpp",
    "opy-software/third_party/qpOASES/*.cpp",
    "opy-software/third_party/Goldfarb_Optimizer/*.cpp",
    "opy-software/third_party/Goldfarb_Optimizer/*.cc",
    # "opy-software/third_party/osqp/**/*.cpp",
    # "opy-software/third_party/osqp/**/*.c",
    # "opy-software/third_party/osqp/**/*.h",
    "opy-software/third_party/JCQP/**/*.cpp",
    "opy-software/third_party/JCQP/**/*.c",
            "opy-software/src/MotorDriverNew.cpp",
              "opy-software/src/PeriodicTask.cpp",
        #    "opy-software/src/CANInterface.cpp",
            #  "opy-software/src/asel_reader.c",
            #  "opy-software/src/IMUTest_helper.c",
             "opy-software/src/MotorCMD.cpp",
             #"opy-software/src/SwingHelper.cpp",
             #"opy-software/src/IMUreader.cpp",
            #  "opy-software/src/IMUreader_vn100.cpp",
             "opy-software/src/Controllers/LegController.cpp",
             "opy-software/src/Dynamics/Quadruped.cpp",
             "opy-software/src/Dynamics/FloatingBaseModel.cpp",
             "opy-software/src/Controllers/SingleLegController.cpp",
             "opy-software/src/Utilities/utilities.cpp",
             "opy-software/src/Controllers/DesiredStateCommand.cpp",
             "opy-software/src/Estimators/OrientationEstimator.cpp",
             "opy-software/src/Estimators/ContactEstimator.cpp",
             "opy-software/src/Estimators/PositionVelocityEstimator.cpp",
             "opy-software/src/Controllers/WBIC.cpp",
             "opy-software/src/Controllers/WBC_Ctrl.cpp",
             "opy-software/src/Controllers/KinWBC.cpp",
             "opy-software/src/FSM/Stand_FSM.cpp",
             "opy-software/src/FSM/FSM_State.cpp",
             "opy-software/src/FSM/ControlFSM.cpp",
             "opy-software/src/FSM/FSM_State_BalanceStand.cpp",
             "opy-software/src/FSM/FSM_State_Locomotion.cpp",
             "opy-software/src/FSM/FSM_State_Passive.cpp",
             "opy-software/src/FSM/FSM_State_Recovery.cpp",
             "opy-software/src/FSM/FSM_State_eDamp.cpp",
             "opy-software/src/MIT_Controller.cpp",
             "opy-software/src/RobotRunner.cpp",
             "opy-software/src/Bridge/Bridge.cpp",
             "opy-software/src/Controllers/ControlParameters/ControlParameterInterface.cpp",
             "opy-software/src/Controllers/ControlParameters/ControlParameters.cpp",
             "opy-software/src/Controllers/FootstepPlanner/GraphSearch.cpp",
             "opy-software/src/Controllers/convexMPC_interface.cpp",
             "opy-software/src/Controllers/SolverMPC.cpp",
             "opy-software/src/Controllers/RobotState.cpp",
             "opy-software/src/Controllers/FootSwingTrajectory.cpp",
             "opy-software/src/Controllers/ContactSet/FixedBodyContact.cpp",
             "opy-software/src/Controllers/ContactSet/SingleContact.cpp",
             "opy-software/src/Controllers/TaskSet/BodyOriTask.cpp",
             "opy-software/src/Controllers/TaskSet/BodyPosTask.cpp",
             "opy-software/src/Controllers/TaskSet/BodyPostureTask.cpp",
             "opy-software/src/Controllers/TaskSet/BodyRyRzTask.cpp",
             "opy-software/src/Controllers/TaskSet/JPosTask.cpp",
             "opy-software/src/Controllers/TaskSet/LinkPosTask.cpp",
             "opy-software/src/Controllers/TaskSet/LocalHeadPosTask.cpp",
             "opy-software/src/Controllers/TaskSet/LocalPosTask.cpp",
             "opy-software/src/Controllers/TaskSet/LocalRollTask.cpp",
             "opy-software/src/Controllers/TaskSet/LocalTailPosTask.cpp",
             "opy-software/src/Controllers/LocomotionCtrl.cpp",
             "opy-software/src/Controllers/ConvexMPCLocomotion.cpp",
             "opy-software/src/Controllers/SparseCMPC.cpp",
             "opy-software/src/Controllers/SparseCMPC_Math.cpp",
             "opy-software/src/Controllers/OsqpTriples.cpp",
             "opy-software/src/Controllers/GaitScheduler.cpp",
             "opy-software/src/Controllers/Gait.cpp",
             "opy-software/src/Controllers/ContactKalman.cpp",
             "opy-software/src/Solo12MB_Ctrl.cpp",
            # "opy-software/include/third_party/Goldfarb_Optimizer/QuadProg++.cc"
            
        ]),
    
    data = [
        ":gen_mujoco_gym_xml",
    ],
    deps = [
        "//envpool/core:async_envpool",
        "@mujoco//:mujoco_lib",

    ],
    linkopts = ["-ltinyxml2 -usr/local/lib/libosqp.so -usr/local/lib/libqdldl.so -usr/local/lib/libqpOASES.so"], 
    copts = ["-DHW_ENABLE=ON -D OPYv05=1 -O3 -ggdb -w  -D__NO_COPYRIGHT__ -Wno-error  -Wall -pedantic -Wfloat-equal -Wshadow -DLINUX -finline-functions"],
)
# -no-pie 
pybind_extension(
    name = "mujoco_gym_envpool",
    srcs = [
        "gym/mujoco_envpool.cc",
    ],
    deps = [
        ":mujoco_gym_env",
        "//envpool/core:py_envpool",
    ],
)

cc_library(
    name = "mujoco_dmc_env",
    srcs = [
        "dmc/mujoco_env.cc",
        "dmc/utils.cc",
    ],
    hdrs = [
        "dmc/acrobot.h",
        "dmc/ball_in_cup.h",
        "dmc/cartpole.h",
        "dmc/cheetah.h",
        "dmc/finger.h",
        "dmc/fish.h",
        "dmc/hopper.h",
        "dmc/humanoid.h",
        "dmc/humanoid_CMU.h",
        "dmc/manipulator.h",
        "dmc/mujoco_env.h",
        "dmc/pendulum.h",
        "dmc/point_mass.h",
        "dmc/reacher.h",
        "dmc/swimmer.h",
        "dmc/utils.h",
        "dmc/walker.h",
    ],
    data = [":gen_mujoco_dmc_xml"],
    deps = [
        "//envpool/core:async_envpool",
        "@mujoco//:mujoco_lib",
        "@pugixml",
    ],
)

pybind_extension(
    name = "mujoco_dmc_envpool",
    srcs = [
        "dmc/mujoco_envpool.cc",
    ],
    deps = [
        ":mujoco_dmc_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "mujoco_dmc",
    srcs = ["dmc/__init__.py"],
    data = [
        ":gen_mujoco_dmc_xml",
        ":gen_mujoco_so",
        ":mujoco_dmc_envpool.so",
    ],
    deps = ["//envpool/python:api"],
)

py_library(
    name = "mujoco_gym",
    srcs = ["gym/__init__.py"],
    data = [
        ":gen_mujoco_gym_xml",
        ":gen_mujoco_so",
        ":mujoco_gym_envpool.so",
    ],
    deps = ["//envpool/python:api"],
)

cc_test(
    name = "mujoco_envpool_test",
    size = "enormous",
    srcs = ["gym/mujoco_gym_envpool_test.cc"],
    deps = [
        ":mujoco_gym_env",
        "@com_google_googletest//:gtest_main",
    ],
)

py_library(
    name = "mujoco_dmc_registration",
    srcs = ["dmc/registration.py"],
    deps = ["//envpool:registration"],
)

py_library(
    name = "mujoco_gym_registration",
    srcs = ["gym/registration.py"],
    deps = ["//envpool:registration"],
)

py_test(
    name = "mujoco_gym_deterministic_test",
    size = "enormous",
    srcs = ["gym/mujoco_gym_deterministic_test.py"],
    deps = [
        ":mujoco_gym",
        ":mujoco_gym_registration",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("gym"),
    ],
)

py_test(
    name = "mujoco_gym_align_test",
    size = "enormous",
    srcs = ["gym/mujoco_gym_align_test.py"],
    deps = [
        ":mujoco_gym",
        ":mujoco_gym_registration",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("gym"),
        requirement("mujoco"),
    ],
)

py_test(
    name = "mujoco_dmc_suite_deterministic_test",
    size = "enormous",
    srcs = ["dmc/mujoco_dmc_suite_deterministic_test.py"],
    deps = [
        ":mujoco_dmc",
        ":mujoco_dmc_registration",
        requirement("numpy"),
        requirement("absl-py"),
    ],
)

py_test(
    name = "mujoco_dmc_suite_align_test",
    size = "enormous",
    srcs = ["dmc/mujoco_dmc_suite_align_test.py"],
    deps = [
        ":mujoco_dmc",
        ":mujoco_dmc_registration",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("mujoco"),
        requirement("dm-env"),
        requirement("dm-control"),
    ],
)

py_test(
    name = "mujoco_dmc_suite_ext_deterministic_test",
    size = "enormous",
    srcs = ["dmc/mujoco_dmc_suite_ext_deterministic_test.py"],
    deps = [
        ":mujoco_dmc",
        ":mujoco_dmc_registration",
        requirement("numpy"),
        requirement("absl-py"),
    ],
)

py_test(
    name = "mujoco_dmc_suite_ext_align_test",
    size = "enormous",
    srcs = ["dmc/mujoco_dmc_suite_ext_align_test.py"],
    deps = [
        ":mujoco_dmc",
        ":mujoco_dmc_registration",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("mujoco"),
        requirement("dm-env"),
        requirement("dm-control"),
    ],
)
