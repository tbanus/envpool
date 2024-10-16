/*
 * Copyright 2022 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENVPOOL_MUJOCO_GYM_HUMANOID_H_
#define ENVPOOL_MUJOCO_GYM_HUMANOID_H_

#include <algorithm>
#include <fstream>
#include <limits>
#include <memory>
#include <iostream>
#include <string>
#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/gym/mujoco_env.h"
#include <RobotRunner.h>
#include <RobotController.h>
#include <MIT_Controller.hpp>
#include <MotorCMD.h>
#include <eigen3/Eigen/Dense>

namespace mujoco_gym {

class HumanoidEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "frame_skip"_.Bind(5), "post_constraint"_.Bind(true),
        "use_contact_force"_.Bind(false), "forward_reward_weight"_.Bind(1.25),
        "terminate_when_unhealthy"_.Bind(true),
        "exclude_current_positions_from_observation"_.Bind(true),
        "ctrl_cost_weight"_.Bind(0.0), "healthy_reward"_.Bind(5.0),
        "healthy_z_min"_.Bind(0.05), "healthy_z_max"_.Bind(0.45),
        "contact_cost_weight"_.Bind(5e-7), "contact_cost_max"_.Bind(10.0),
        "reset_noise_scale"_.Bind(0));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    bool no_pos = conf["exclude_current_positions_from_observation"_];
    return MakeDict(
        "obs"_.Bind(Spec<mjtNum>({361}, {-inf, inf})),
#ifdef ENVPOOL_TEST
        "info:qpos0"_.Bind(Spec<mjtNum>({24})),
        "info:qvel0"_.Bind(Spec<mjtNum>({23})),
#endif
        "info:reward_linvel"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_quadctrl"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_alive"_.Bind(Spec<mjtNum>({-1})),
        "info:reward_impact"_.Bind(Spec<mjtNum>({-1})),
        "info:x_position"_.Bind(Spec<mjtNum>({-1})),
        "info:y_position"_.Bind(Spec<mjtNum>({-1})),
        "info:distance_from_origin"_.Bind(Spec<mjtNum>({-1})),
        "info:x_velocity"_.Bind(Spec<mjtNum>({-1})),
        "info:y_velocity"_.Bind(Spec<mjtNum>({-1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 12}, {-40, 40})));
  }
};

using HumanoidEnvSpec = EnvSpec<HumanoidEnvFns>;

class HumanoidEnv : public Env<HumanoidEnvSpec>, public MujocoEnv {
 protected:
    bool terminate_when_unhealthy_, no_pos_, use_contact_force_;
    mjtNum ctrl_cost_weight_, forward_reward_weight_, healthy_reward_;
    mjtNum healthy_z_min_, healthy_z_max_;
    mjtNum contact_cost_weight_, contact_cost_max_;
    std::uniform_real_distribution<> dist_;
    RobotRunner* _robotRunner;
      // MIT_Controller robot_ctrl;

    CANcommand* _Command;
    CANData _Feedback;
    IMUData _ImuData;
    GamepadCommand _GamepadCommand;
    RobotController* ctrl;
    RobotControlParameters _robotParams;
    std::ofstream outputFile;
  PeriodicTaskManager taskManager;

 public:
  HumanoidEnv(const Spec& spec, int env_id)
      : Env<HumanoidEnvSpec>(spec, env_id),
        MujocoEnv(std::string("/home/banus/thesis-project/legged-sim/resource/opy_v05/opy_v05.xml"),
                  spec.config["frame_skip"_], spec.config["post_constraint"_],
                  spec.config["max_episode_steps"_]),
        terminate_when_unhealthy_(spec.config["terminate_when_unhealthy"_]),
        no_pos_(spec.config["exclude_current_positions_from_observation"_]),
        use_contact_force_(spec.config["use_contact_force"_]),
        ctrl_cost_weight_(spec.config["ctrl_cost_weight"_]),
        forward_reward_weight_(spec.config["forward_reward_weight"_]),
        healthy_reward_(spec.config["healthy_reward"_]),
        healthy_z_min_(spec.config["healthy_z_min"_]),
        healthy_z_max_(spec.config["healthy_z_max"_]),
        contact_cost_weight_(spec.config["contact_cost_weight"_]),
        contact_cost_max_(spec.config["contact_cost_max"_]),
        dist_(-spec.config["reset_noise_scale"_],
              spec.config["reset_noise_scale"_])
         {
    MIT_Controller robot_ctrl;
    std::cout << "HumanoidEnv" << std::endl;
    _robotRunner= new RobotRunner(&robot_ctrl, &taskManager, 0.002, "robot-control", 30);
    std::cout << "Initialized MIT_Controller\n";
  
    ctrl = &robot_ctrl;
    std::cout << "Assigned MIT_Controller to ctrl\n";

    _robotRunner->driverCommand = &_GamepadCommand;
    std::cout << "Assigned GamepadCommand to RobotRunner\n";

    _robotRunner->vectorNavData = &_ImuData;
    std::cout << "Assigned IMUData to RobotRunner\n";

    _robotRunner->setCANData((&_Feedback));


    std::cout << "Assigned CANData to RobotRunner\n";

    _robotRunner->_CANcommand = _Command;
    std::cout << "Assigned CANcommand to RobotRunner\n";

    _robotRunner->controlParameters = &_robotParams;
    std::cout << "Assigned RobotControlParameters to RobotRunner\n";
    // printf("CANData: %p\n", _robotRunner->_CANData);

    _robotRunner->init();
    std::cout << "Initialized RobotRunner\n";
    ctrl->getControlFSM()->data.controlParameters->control_mode=1;
    // WriteState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    std::string fname;
    fname = "/home/banus/thesis-project/envpool/logs/" + std::to_string(env_id_) + "_log.csv";
    outputFile.open(fname.c_str());
  }

  void MujocoResetModel() override {
    printf("mjResetModel\n");
    for (int i = 0; i < model_->nq; ++i) {
      data_->qpos[i] = init_qpos_[i] + dist_(gen_);
    }
    for (int i = 0; i < model_->nv; ++i) {
      data_->qvel[i] = init_qvel_[i] + dist_(gen_);
    }
    int kSideSign_[4] = {-1, 1, -1, 1};
    model_->opt.timestep = 0.002;
    for (int leg = 0; leg < 4; leg++) {
      data_->qpos[(leg)*3 + 0 + 7] = 1 * (M_PI / 180) * kSideSign_[leg];  // Add 7 to skip the first 7 dofs from body. (Position + Quaternion)
      data_->qpos[(leg)*3 + 1 + 7] = -90 * (M_PI / 180);  // *kDirSign_[leg];
      data_->qpos[(leg)*3 + 2 + 7] = 173 * (M_PI / 180);  // *kDirSign_[leg];
    }

#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_, data_->qpos, sizeof(mjtNum) * model_->nq);
    std::memcpy(qvel0_, data_->qvel, sizeof(mjtNum) * model_->nv);
#endif
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    printf("Reset\n");
    MujocoReset();
    std::cout << "Reset" << std::endl;
    WriteState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    // MIT_Controller robot_ctrl;
    // ctrl=&robot_ctrl;
    // std::cout << "Initialized MIT_Controller\n";

    // std::cout << "WriteState" << std::endl;
    // _robotRunner = new RobotRunner(ctrl, &taskManager, 0.002, "robot-control", 30);
    // std::cout << "Allocated RobotRunner\n";

    // _robotRunner->driverCommand = &_GamepadCommand;
    // std::cout << "Assigned GamepadCommand to RobotRunner\n";

    // _robotRunner->vectorNavData = &_ImuData;
    // std::cout << "Assigned IMUData to RobotRunner\n";

    // printf("[Humanoid.h] [HumanoidEnv::Reset] CANData: %p\n", &_Feedback);
    // _robotRunner->setCANData(&_Feedback);
    //     std::cout << "Assigned CANData to RobotRunner\n";

    // _robotRunner->_CANcommand = &_Command;
    // std::cout << "Assigned CANcommand to RobotRunner\n";

    // _robotRunner->controlParameters = &_robotParams;
    // std::cout << "Assigned RobotControlParameters to RobotRunner\n";

    // _robotRunner->init();
    // std::cout << "Initialized RobotRunner\n";

    done_ = false;
    elapsed_step_ = 0;
  }

   void Step(const Action& action) override {
      int debug = 1; // Set this to true or false to enable/disable debug prints
     if (debug) {
          std::cout << "[DEBUG] [humanoid.h] [HumanoidEnv::Step] Called Step" << std::endl;
     }
      _robotRunner->run();
      if (debug) {
          std::cout << "[DEBUG] [humanoid.h] [HumanoidEnv::Step] Called _robotRunner->run()" << std::endl;
      }
  
      mjtNum* act = static_cast<mjtNum*>(action["action"_].Data());
      if (debug) {
          std::cout << "[DEBUG] [humanoid.h] [HumanoidEnv::Step] Retrieved action data" << std::endl;
      }
  
      mjtNum motor_commands[12];
      // for (int leg = 0; leg < 4; leg++) {
      //     motor_commands[leg * 3 + 0 + 0] = _Command->tau_abad_ff[leg] + _Command->kp_abad[leg] * (_Command->q_des_abad[leg] - _Feedback.q_abad[leg]) +
      //                                       _Command->kd_abad[leg] * (_Command->qd_des_abad[leg] - _Feedback.qd_abad[leg]);  // Torque
      //     motor_commands[leg * 3 + 1 + 0] = _Command->tau_hip_ff[leg] + _Command->kp_hip[leg] * (_Command->q_des_hip[leg] - _Feedback.q_hip[leg]) +
      //                                       _Command->kd_hip[leg] * (_Command->qd_des_hip[leg] - _Feedback.qd_hip[leg]);  // Torque
      //     motor_commands[leg * 3 + 2 + 0] = _Command->tau_knee_ff[leg] + _Command->kp_knee[leg] * (_Command->q_des_knee[leg] - _Feedback.q_knee[leg]) +
      //                                       _Command->kd_knee[leg] * (_Command->qd_des_knee[leg] - _Feedback.qd_knee[leg]);  // Torque
      //     if (debug) {
      //         std::cout << "[DEBUG] [humanoid.h] [HumanoidEnv::Step] Calculated motor commands for leg " << leg << std::endl;
      //     }
      //     for(int i = 0; i < 3; i++) {
      //       printf(" Humanoid (_Command->q_des_abad[leg] %f\n",_Command->q_des_abad[i]);
      //         // printf("motor_commands[leg * 3 + 0 + 0] %f\n",motor_commands[i]);
      //     }
          
      // }
  
      const auto& before = GetMassCenter();
      if (debug) {
          std::cout << "[DEBUG] [humanoid.h] [HumanoidEnv::Step] Retrieved mass center before MujocoStep" << std::endl;
      }
  
      MujocoStep(motor_commands);
      if (debug) {
          std::cout << "[DEBUG] [humanoid.h] [HumanoidEnv::Step] Called MujocoStep with motor commands" << std::endl;
      }
  
      const auto& after = GetMassCenter();
      if (debug) {
          std::cout << "[DEBUG] [humanoid.h] [HumanoidEnv::Step] Retrieved mass center after MujocoStep" << std::endl;
      }
  
      // ctrl_cost
      mjtNum ctrl_cost = 0.0;
      for (int i = 0; i < model_->nu; ++i) {
          ctrl_cost += ctrl_cost_weight_ * act[i] * act[i];
      }
      if (debug) {
          std::cout << "[DEBUG] [humanoid.h] [HumanoidEnv::Step] Calculated control cost" << std::endl;
      }
  
      // xv and yv
      mjtNum dt = frame_skip_ * model_->opt.timestep;
      mjtNum xv = (after[0] - before[0]) / dt;
      mjtNum yv = (after[1] - before[1]) / dt;
      if (debug) {
          std::cout << "[DEBUG] [humanoid.h] [HumanoidEnv::Step] Calculated xv and yv" << std::endl;
      }
  
      // contact cost
      mjtNum contact_cost = 0.0;
      if (use_contact_force_) {
          for (int i = 0; i < 6 * model_->nbody; ++i) {
              mjtNum x = data_->cfrc_ext[i];
              contact_cost += contact_cost_weight_ * x * x;
          }
          contact_cost = std::min(contact_cost, contact_cost_max_);
      }
      if (debug) {
          std::cout << "[DEBUG] [humanoid.h] [HumanoidEnv::Step] Calculated contact cost" << std::endl;
      }
  
      // reward and done
      mjtNum healthy_reward = terminate_when_unhealthy_ || IsHealthy() ? healthy_reward_ : 0.0;
      auto reward = static_cast<float>(xv * forward_reward_weight_ + healthy_reward - ctrl_cost - contact_cost);
      ++elapsed_step_;
      done_ = (terminate_when_unhealthy_ ? !IsHealthy() : false) || (elapsed_step_ >= max_episode_steps_);
      if (debug) {
          std::cout << "[DEBUG] [humanoid.h] [HumanoidEnv::Step] Calculated reward and updated done status" << std::endl;
      }
  
      WriteState(reward, xv, yv, ctrl_cost, contact_cost, after[0], after[1], healthy_reward);
      if (debug) {
          std::cout << "[DEBUG] [humanoid.h] [HumanoidEnv::Step] Wrote state with reward and other metrics" << std::endl;
      }
  }

 private:
  bool IsHealthy() {
    return healthy_z_min_ < data_->qpos[2] && data_->qpos[2] < healthy_z_max_;
  }

  std::array<mjtNum, 2> GetMassCenter() {
    mjtNum mass_sum = 0.0;
    mjtNum mass_x = 0.0;
    mjtNum mass_y = 0.0;
    for (int i = 0; i < model_->nbody; ++i) {
      mjtNum mass = model_->body_mass[i];
      mass_sum += mass;
      mass_x += mass * data_->xipos[i * 3 + 0];
      mass_y += mass * data_->xipos[i * 3 + 1];
    }
    return {mass_x / mass_sum, mass_y / mass_sum};
  }

  void WriteState(float reward, mjtNum xv, mjtNum yv, mjtNum ctrl_cost,
                  mjtNum contact_cost, mjtNum x_after, mjtNum y_after,
                  mjtNum healthy_reward) {
    State state = Allocate();
    state["reward"_] = reward;
    mjtNum* obs = static_cast<mjtNum*>(state["obs"_].Data());
    for (int i = no_pos_ ? 2 : 0; i < model_->nq; ++i) {
      *(obs++) = data_->qpos[i];
    }
    for (int i = 0; i < model_->nv; ++i) {
      *(obs++) = data_->qvel[i];
    }
    for (int i = 0; i < 10 * model_->nbody; ++i) {
      *(obs++) = data_->cinert[i];
    }
    for (int i = 0; i < 6 * model_->nbody; ++i) {
      *(obs++) = data_->cvel[i];
    }
    for (int i = 0; i < model_->nv; ++i) {
      *(obs++) = data_->qfrc_actuator[i];
    }
    for (int i = 0; i < 6 * model_->nbody; ++i) {
      *(obs++) = data_->cfrc_ext[i];
    }

    state["info:reward_linvel"_] = xv * forward_reward_weight_;
    state["info:reward_quadctrl"_] = -ctrl_cost;
    state["info:reward_alive"_] = healthy_reward;
    state["info:reward_impact"_] = -contact_cost;
    state["info:x_position"_] = x_after;
    state["info:y_position"_] = y_after;
    state["info:distance_from_origin"_] =
        std::sqrt(x_after * x_after + y_after * y_after);
    state["info:x_velocity"_] = xv;
    state["info:y_velocity"_] = yv;

    for (int leg = 0; leg < 4; leg++) {
      _Feedback.q_abad[leg] = data_->qpos[(leg)*3 + 0 + 7];  // Add 7 to skip the first 7 dofs from body. (Position + Quaternion)
      _Feedback.q_hip[leg] = data_->qpos[(leg)*3 + 1 + 7];
      _Feedback.q_knee[leg] = data_->qpos[(leg)*3 + 2 + 7];
      _Feedback.qd_abad[leg] = data_->qvel[(leg)*3 + 0 + 6];
      _Feedback.qd_hip[leg] = data_->qvel[(leg)*3 + 1 + 6];
      _Feedback.qd_knee[leg] = data_->qvel[(leg)*3 + 2 + 6];
    }
    _ImuData.acc_x = data_->sensordata[0];
    _ImuData.acc_y = data_->sensordata[1];
    _ImuData.acc_z = data_->sensordata[2];

    _ImuData.heave = data_->qvel[0];
    _ImuData.heave_dt = data_->qvel[1];
    _ImuData.heave_ddt = data_->qvel[2];

    _ImuData.gyr_x = data_->qvel[3];
    _ImuData.gyr_y = data_->qvel[4];
    _ImuData.gyr_z = data_->qvel[5];

    _ImuData.quat[0] = data_->qpos[3];
    _ImuData.quat[1] = data_->qpos[4];
    _ImuData.quat[2] = data_->qpos[5];
    _ImuData.quat[3] = data_->qpos[6];

    _ImuData.pos_x = data_->qpos[0];
    _ImuData.pos_y = data_->qpos[1];
    _ImuData.pos_z = data_->qpos[2];

    for (int i = 0; i < 19; i++) {
      outputFile << data_->qpos[i] << ",";
    }
    outputFile << std::endl;

#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_, model_->nq);
    state["info:qvel0"_].Assign(qvel0_, model_->nv);
#endif
  }
};

using HumanoidEnvPool = AsyncEnvPool<HumanoidEnv>;

}  // namespace mujoco_gym

#endif  // ENVPOOL_MUJOCO_GYM_HUMANOID_H_