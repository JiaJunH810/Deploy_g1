from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch
import copy
# import onnx
import onnxruntime
import datetime
import os
import json
import pickle

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config
import sys
sys.path.append("/home/zy/Deploy_g1/scripts")
from scripts.joint_select_reorder import extend_joint, obs_match


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        # 构建数据字典
        self.data_dict = {
            'root_trans_offset': [],
            'base_lin_vel': [],
            'pose_aa': [],
            'dof': [],
            'dof_vel': [],
            'root_rot': [],
            'base_quat': [],
            'base_ang_vel': [],
            'smpl_joints': [],
            'fps': 30,
        }
        # Initialize the policy network
        # pytorch script
        # self.policy = torch.jit.load(config.policy_path)

        self.recorded_data = {
            #'link_angular_acceleration': [],
            'base_angular_vel': [],
            'projected_gravity': [],
            'dof_pos': [],
            'dof_vel': [],
            'dof_angular_acceleration':[],
            'torque': []
        }
        self.is_recording = False # Flag to control recording

        # onnx
        self.ort_session = onnxruntime.InferenceSession(config.policy_path)
        self.input_name = self.ort_session.get_inputs()[0].name


        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        # self.target_dof_pos = config.default_angles.copy()
        # self.obs = np.zeros(config.num_obs, dtype=np.float32)
        # self.cmd = np.array([0.0, 0, 0])
        self.counter = 0
        self.policy_path = config.policy_path


        #self.target_dof_pos = config.default_angles.copy()
        self.target_dof_pos = config.default_start_angles.copy()

        self.ref_motion_phase = 0
        self.ang_vel_buf = np.zeros(3 * config.history_length, dtype=np.float32)
        self.proj_g_buf = np.zeros(3 * config.history_length, dtype=np.float32)
        self.dof_pos_buf = np.zeros(config.num_actions * config.history_length, dtype=np.float32)
        self.dof_vel_buf = np.zeros(config.num_actions * config.history_length, dtype=np.float32)
        self.action_buf = np.zeros(config.num_actions * config.history_length, dtype=np.float32)
        self.ref_motion_phase_buf = np.zeros(1 * config.history_length, dtype=np.float32)
        self.base_lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.prev_dqj = np.zeros(config.num_actions, dtype=np.float32)

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 1

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)
        self.resort_sub()

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
            

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.action_joint2motor_idx + self.config.fixed_joint2motor_idx
        kps = self.config.kps_start + self.config.fixed_kps
        kds = self.config.kds_start + self.config.fixed_kds
        default_pos = np.concatenate((self.config.default_start_angles, self.config.fixed_target), axis=0)
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].mode = 1
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                # self.low_cmd.motor_cmd[motor_idx].q = target_pos * (1 - alpha)
                self.low_cmd.motor_cmd[motor_idx].dq = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps_start[j]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds_start[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.resort_pub()
            # for i in range(35):
            #     print(f"low_cmd.motor_cmd: {i}  ", self.low_cmd.motor_cmd[i])
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.action_joint2motor_idx)):
                motor_idx = self.config.action_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].mode = 1
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_start_angles[i]
                self.low_cmd.motor_cmd[motor_idx].dq = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps_start[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds_start[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.fixed_joint2motor_idx)):
                motor_idx = self.config.fixed_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.fixed_target[i]
                self.low_cmd.motor_cmd[motor_idx].dq = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.fixed_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.fixed_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            
            self.resort_pub()
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
            
            self.is_recording = True
            print('%'*20)
            print('Start data recording')
            print('%'*20)
        
    def resort_sub(self):
        # 将索引15~26的数据向前移两格
        for i in range(15, 27, 1):
            self.low_state.motor_state[i - 2] = copy.deepcopy(self.low_state.motor_state[i])
        # 将索引20~24的数据再向前移两格子（移动后）
        for i in range(20, 25, 1):
            self.low_state.motor_state[i - 2] = copy.deepcopy(self.low_state.motor_state[i])
        # 将后面的6个数据全部设置为空
        for i in range(23, 29, 1):
            self.low_state.motor_state[i].mode = 0
    
    def resort_pub(self):
        
        # 索引22~13
        for i in range(22, 12, -1):
            self.low_cmd.motor_cmd[i + 2] = copy.deepcopy(self.low_cmd.motor_cmd[i])
        
        # 索引13数据设置为空
        self.low_cmd.motor_cmd[13].mode = 0
        # 索引14数据设置为空
        self.low_cmd.motor_cmd[14].mode = 0
        # 将索引20~24数据再往后移两格
        for i in range (24, 19, -1):
            self.low_cmd.motor_cmd[i + 2] = copy.deepcopy(self.low_cmd.motor_cmd[i])
        self.low_cmd.motor_cmd[20].mode = 0
        self.low_cmd.motor_cmd[21].mode = 0
        # 再将第28和第29的数据设置为空，扩张到29个电机
        for i in range(27, 35):
            self.low_cmd.motor_cmd[27].mode = 0

    def run(self):
        self.counter += 1

        self.prev_dqj = self.dqj.copy() 

        # Get the current joint position and velocity
        for i in range(len(self.config.action_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.action_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.action_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.fixed_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.fixed_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        
        qj = self.qj.copy()    
        dqj = self.dqj.copy()   
    


        projected_gravity = get_gravity_orientation(quat)
        dof_pos = qj * self.config.dof_pos_scale
        dof_vel = dqj * self.config.dof_vel_scale
        base_ang_vel = ang_vel* self.config.ang_vel_scale
        
        self.ref_motion_phase += self.config.ref_motion_phase
        num_actions = self.config.num_actions


        print("Shapes of arrays to concatenate:")
        print(f"self.action shape: {np.array(self.action).shape}")
        print(f"base_ang_vel shape: {np.array(base_ang_vel).shape}")
        print(f"dof_pos shape: {np.array(dof_pos).shape}")
        print(f"dof_vel shape: {np.array(dof_vel).shape}")
        # print(f"history_obs_buf shape: {np.array(history_obs_buf).shape}")
        print(f"projected_gravity shape: {np.array(projected_gravity).shape}")
        print(f"[self.ref_motion_phase] shape: {np.array([self.ref_motion_phase]).shape}")
        

        history_obs_buf = np.concatenate((self.action_buf, self.ang_vel_buf, self.dof_pos_buf, self.dof_vel_buf, self.proj_g_buf, self.ref_motion_phase_buf), axis=-1, dtype=np.float32)
        
        print(f"history_obs_buf shape: {np.array(history_obs_buf).shape}")

        try:
            obs_buf = np.concatenate((self.action, base_ang_vel.flatten(), dof_pos, dof_vel, history_obs_buf, projected_gravity, [self.ref_motion_phase]), axis=-1, dtype=np.float32)
        except ValueError as e:
            print(f"Concatenation failed with error: {e}")
            print("Please ensure all arrays have the same number of dimensions (either all 1D or all 2D)")
            raise
        # obs_buf = np.concatenate((self.action, base_ang_vel, dof_pos, dof_vel, history_obs_buf, projected_gravity, [self.ref_motion_phase]), axis=-1, dtype=np.float32)
            


        # history_obs_buf = np.concatenate((self.action_buf, self.ang_vel_buf, self.dof_pos_buf, self.dof_vel_buf, self.proj_g_buf, self.ref_motion_phase_buf), axis=-1, dtype=np.float32)
                        
        # obs_buf = np.concatenate((self.action, base_ang_vel, dof_pos, dof_vel, history_obs_buf, projected_gravity, [self.ref_motion_phase]), axis=-1, dtype=np.float32)
            

        # update history
        self.ang_vel_buf = np.concatenate((base_ang_vel.flatten(), self.ang_vel_buf[:-3]), axis=-1, dtype=np.float32)
        
        self.proj_g_buf = np.concatenate((projected_gravity, self.proj_g_buf[:-3] ), axis=-1, dtype=np.float32)
        self.dof_pos_buf = np.concatenate((dof_pos, self.dof_pos_buf[:-num_actions] ), axis=-1, dtype=np.float32)
        self.dof_vel_buf = np.concatenate((dof_vel, self.dof_vel_buf[:-num_actions] ), axis=-1, dtype=np.float32)
        self.action_buf = np.concatenate((self.action, self.action_buf[:-num_actions] ), axis=-1, dtype=np.float32)
        self.ref_motion_phase_buf = np.concatenate(([self.ref_motion_phase], self.ref_motion_phase_buf[:-1] ), axis=-1, dtype=np.float32)                
        
        
        self.read_data()
        obs_tensor = torch.from_numpy(obs_buf).unsqueeze(0).cpu().numpy()
        self.action = np.squeeze(self.ort_session.run(None, {self.input_name: obs_tensor})[0])
        
        print(f"left ankle pitch: {self.action[4]}")
        print(f"left ankle roll: {self.action[5]}")
        print(f"right ankle pitch: {self.action[10]}")
        print(f"right ankle roll: {self.action[11]}")

        # self.action[4] = 0
        # self.action[5] = 0
        # self.action[10] = 0
        # self.action[11] = 0
        # self.action[17] = 0
        # self.action[22] = 0


        # self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        
        # transform action to target_dof_pos
        target_dof_pos = self.config.default_waist1_angles + self.action * self.config.action_scale
        # target_dof_pos = extend_joint(target_dof_pos)
        # print("#" * 100)
        # print(target_dof_pos.shape)
        # target_dof_pos[4] = 0
        target_dof_pos[5] = 0
        # target_dof_pos[10] = 0
        target_dof_pos[11] = 0
        target_dof_pos[17] = 0
        target_dof_pos[22] = 0


        dof_angular_acceleration = (self.dqj - self.prev_dqj) / self.config.control_dt

        commanded_torques = np.zeros(len(all_motor_indices), dtype=np.float32)

        # Build low cmd
        for i in range(len(self.config.action_joint2motor_idx)):
            motor_idx = self.config.action_joint2motor_idx[i]

#possible problem
            q_current = self.low_state.motor_state[motor_idx].q
            dq_current = self.low_state.motor_state[motor_idx].dq

            self.low_cmd.motor_cmd[motor_idx].mode = 1
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            #self.low_cmd.motor_cmd[motor_idx].dq = 0

            #replaced dq with dq
            self.low_cmd.motor_cmd[motor_idx].dq = 0

            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps_start[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds_start[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

            tau_val = all_kps[i] * (target_dof_pos[i] - q_current) - all_kds[i] * dq_current
            
            commanded_torques[i] = tau_val

        for i in range(len(self.config.fixed_joint2motor_idx)):
            motor_idx = self.config.fixed_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.fixed_target[i]
            #self.low_cmd.motor_cmd[motor_idx].dq = 0

            #replaced dq with dq
            self.low_cmd.motor_cmd[motor_idx].dq = 0

            self.low_cmd.motor_cmd[motor_idx].kp = self.config.fixed_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.fixed_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        self.resort_pub()

        if self.is_recording:
            # base_angular_vel 是经过转换和缩放的 ang_vel
            # dof_pos 是原始的 qj
            # dof_vel 是原始的 dqj
            self.recorded_data['base_angular_vel'].append(base_ang_vel.flatten().tolist())
            self.recorded_data['projected_gravity'].append(projected_gravity.tolist())
            self.recorded_data['dof_pos'].append(self.qj.tolist()) # 原始关节位置
            self.recorded_data['dof_vel'].append(self.dqj.tolist()) # 原始关节速度
            self.recorded_data['dof_angular_acceleration'].append(dof_angular_acceleration.tolist())
            self.recorded_data['torque'].append(commanded_torques.tolist())


        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt)
        
    def read_data(self):
        # 基向量位移root_trams_offset(可以没有)
        # 连杆位姿 pose_aa(可以没有)
        # dof(就是dof_pos就是关节位置)(qj)
        # 基向量的旋转四元组root_rot(quat)
        # fps是频率
        # dof_vel是关节速度(dqj)
        # base_lin_vel
        # base_ang_vel
        root_trans_offset = [0.0, 0.0, 0.0]
        pose_aa = [0.0, 0.0, 0.0]
        dof_pos = self.qj.copy() * self.config.dof_pos_scale
        quat = self.low_state.imu_state.quaternion
        dof_vel = dqj = self.dqj.copy() * self.config.dof_vel_scale
        base_lin_acc = np.array([self.low_state.imu_state.accelerometer], dtype=np.float32)
        self.base_lin_vel = self.base_lin_vel + base_lin_acc * self.config.control_dt
        base_ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32) * self.config.ang_vel_scale
        root_rot = [0.0, 0.0, 0.0]
        fps = int(1 / self.config.control_dt)
        smpl_joints = np.zeros((24, 3), dtype=np.float32)
        # 生成带时间戳的文件名（精确到分钟）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        policy_path = self.policy_path.split("/")[-1].split(".")[0]
        filename = f"/home/zy/桌面/{policy_path}/{policy_path}_{timestamp}.pkl"
        try:
            os.makedirs(f"/home/zy/桌面/{policy_path}")
        except Exception as e:
            pass
        self.data_dict['root_trans_offset'].append(root_trans_offset)
        self.data_dict['base_lin_vel'].append(self.base_lin_vel.tolist())
        self.data_dict['pose_aa'].append(pose_aa)
        self.data_dict['dof'].append(dof_pos.tolist())
        self.data_dict['dof_vel'].append(dof_vel.tolist())
        self.data_dict['root_rot'].append(root_rot)
        self.data_dict['base_quat'].append(quat)
        self.data_dict['base_ang_vel'].append(base_ang_vel.tolist())
        self.data_dict['smpl_joints'].append(smpl_joints.tolist())
        with open(filename, 'wb') as f:
            pickle.dump(self.data_dict, f)



     def save_recorded_data(self):
        print("正在保存记录的数据...")
        # 从policy_path中提取策略名称 (例如 "3Waist_RoundHouseKick_28000")
        policy_name = os.path.basename(self.config.policy_path).split('.')[0]
        # 获取精确到时分秒的时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 定义保存数据的目录: /home/zy/Deploy_g1/legged_gym/data/
        # 这里假设 LEGGED_GYM_ROOT_DIR 是 /home/zy/Deploy_g1/legged_gym
        save_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "data")
        os.makedirs(save_dir, exist_ok=True) # 如果目录不存在则创建

        # 构建文件名，例如: 3Waist_RoundHouseKick_28000_20250709_184730.npy
        filename = os.path.join(save_dir, f"{policy_name}_{timestamp}.npy")

        # 将所有列表数据转换为NumPy数组以便高效保存
        for key, value in self.recorded_data.items():
            self.recorded_data[key] = np.array(value, dtype=np.float32)

        np.save(filename, self.recorded_data)
        print(f"数据已成功保存至：{filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="/home/zy/Deploy_g1/deploy_real_edpsw/configs/g1_real_waist1.yaml")
    args = parser.parse_args()

    # Load config
    config_path = "/home/zy/Deploy_g1/deploy_real_edpsw/configs/g1_real_waist1.yaml" #args.config
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()
    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    # while True:
    #     try:
    #         controller.run()
    #         # Press the select key to exit
    #         if controller.remote_controller.button[KeyMap.select] == 1:
    #             break
    #     except KeyboardInterrupt:
    #         break
    # # Enter the damping state
    # create_damping_cmd(controller.low_cmd)
    # controller.send_cmd(controller.low_cmd)
    # print("Exit")
    try:
        while True:
            controller.run()
            # 按下选择键 (Select) 退出
            if controller.remote_controller.button[KeyMap.select] == 1:
                print("检测到Select键按下。正在退出...")
                break
    except KeyboardInterrupt:
        print("检测到键盘中断 (Ctrl+C)。正在退出...")
    finally:
        # 确保在程序退出前保存记录的数据
        if controller.is_recording: # 仅当 recording 确实开始时才保存
            controller.save_recorded_data()
        # 进入阻尼状态
        create_damping_cmd(controller.low_cmd)
        controller.send_cmd(controller.low_cmd)
        print("程序已退出。")


