"""Real-robot backend for the Guava harness.

The physical robot is reached through an HTTP bridge (see ``pixi_fer_ros2/guava_bridge``)
so this package needs no ROS installation. ``RealEnv`` and ``RealRobotArm`` expose the
subset of ``guava.sim.env.RoboEnv`` / ``guava.robot.arm.RobotArm`` used by the nine
harness tools, so ``guava.tools.tools`` runs unchanged on hardware.
"""
