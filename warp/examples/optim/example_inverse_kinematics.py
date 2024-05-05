# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid Kinematics
#
# Tests rigid body forward and backwards kinematics through the
# wp.sim.eval_ik() and wp.sim.eval_fk() methods.
#
###########################################################################

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render
import warp.examples

wp.init()

TARGET = wp.constant(wp.vec3(2.0, 1.0, 0.0))


@wp.kernel
def compute_loss(body_q: wp.array(dtype=wp.transform), body_index: int, loss: wp.array(dtype=float)):
    x = wp.transform_get_translation(body_q[body_index])

    delta = x - TARGET
    loss[0] = wp.dot(delta, delta)


@wp.kernel
def step_kernel(x: wp.array(dtype=float), grad: wp.array(dtype=float), alpha: float):
    tid = wp.tid()

    # gradient descent step
    x[tid] = x[tid] - grad[tid] * alpha


class Example:
    def __init__(self, stage, device=None, verbose=False):
        self.verbose = verbose
        if device is None:
            self.device = wp.get_device()
        else:
            self.device = device

        self.frame_dt = 1.0 / 60.0
        self.render_time = 0.0

        articulation_builder = wp.sim.ModelBuilder()

        ROBOT_MODELS = {
            "pr2": "mj_pr2/mj_pr2.mjcf",
            "allegro": "allegro/allegro_right.mjcf",
            "panda": "panda/panda.mjcf",
            "panda_robotiq": "panda_robotiq/panda_robotiq.mjcf",
            "ur5e": "ur5e/ur5e.xml",
            "ur10e": "ur10e/ur10e.xml",
            "kuka_iiwa_14": "kuka_iiwa_14/iiwa14.xml",
            "jaco2": "jaco2/jaco2.xml",
            "shadow_left": "shadow_hand/left_hand.xml",
            "shadow_right": "shadow_hand/right_hand.xml",
        }

        import os
        wp.sim.parse_mjcf(
            os.path.join(warp.examples.get_asset_directory(), ROBOT_MODELS["ur5e"]),
            articulation_builder,
            xform=wp.transform_identity(),
            stiffness=0.0,
            damping=1.0,
            armature=0.1,
            contact_ke=1.0e4,
            contact_kd=1.0e2,
            contact_kf=1.0e2,
            contact_mu=0.75,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
            up_axis="y"
        )
        builder = wp.sim.ModelBuilder()
        builder.add_builder(articulation_builder)

        # finalize model
        self.model = builder.finalize(self.device)
        self.model.ground = False

        self.state = self.model.state()

        self.renderer = None
        if stage:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=50.0)

        # optimization variables
        self.loss = wp.zeros(1, dtype=float, device=self.device)

        self.model.joint_q.requires_grad = True
        self.state.body_q.requires_grad = True
        self.loss.requires_grad = True

        self.train_rate = 0.01

    def forward(self):
        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)

        wp.launch(
            compute_loss,
            dim=1,
            inputs=[self.state.body_q, len(self.state.body_q) - 1, self.loss],
            device=self.device,
        )

    def step(self):
        tape = wp.Tape()
        with tape:
            self.forward()
        tape.backward(loss=self.loss)

        if self.verbose:
            print(f"loss: {self.loss}")
            print(f"joint_grad: {tape.gradients[self.model.joint_q]}")

        # gradient descent
        wp.launch(
            step_kernel,
            dim=len(self.model.joint_q),
            inputs=[self.model.joint_q, tape.gradients[self.model.joint_q], self.train_rate],
            device=self.device,
        )

        # zero gradients
        tape.zero()

    def render(self):
        if self.renderer is None:
            return

        self.renderer.begin_frame(self.render_time)
        self.renderer.render(self.state)
        self.renderer.render_sphere(name="target", pos=TARGET, rot=wp.quat_identity(), radius=0.1, color=(1.0, 0.0, 0.0))
        self.renderer.end_frame()
        self.render_time += self.frame_dt


if __name__ == "__main__":
    stage_path = "example_inverse_kinematics.usd"

    example = Example(stage_path, device=wp.get_preferred_device(), verbose=True)

    train_iters = 512

    for _ in range(train_iters):
        example.step()
        example.render()

    if example.renderer:
        example.renderer.save()
