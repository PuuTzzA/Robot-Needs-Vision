#!/usr/bin/env python3
"""Named Nav2 goals for ROS 2 Humble. No motion until GO is entered."""
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import queue
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, PoseStamped
from nav2_msgs.action import NavigateToPose
import yaml


def map_identity(path):
    config = yaml.safe_load(path.read_text())
    image = Path(config['image'])
    if not image.is_absolute():
        image = path.parent / image
    return hashlib.sha256(path.read_bytes() + image.read_bytes()).hexdigest()


def parse_route(words, points):
    names = [word.upper() for word in words]
    if not names or any(name not in points for name in names):
        raise ValueError('Please save all waypoints in the queue first: ' + ' '.join(names))
    return names


class PointConsole(Node):
    def __init__(self, path, identity):
        super().__init__('named_point_console')
        self.path, self.identity = path, identity
        self.points = {}
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get('map_sha256') != identity:
                raise ValueError('Waypoint file does not match this map; please specify a new --points file.')
            self.points = saved['points']
            for point in self.points.values():
                if not all(math.isfinite(float(point[k])) for k in ('x', 'y', 'yaw_degrees')):
                    raise ValueError('Waypoint contains invalid coordinates')
        self.clicked = None
        self.clicked_yaw = None
        self.route = deque()
        self.current = None
        self.goal_handle = None
        self.cancel_requested = False
        self.cancel_sent = False
        self.client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.create_subscription(PointStamped, '/clicked_point', self.receive, 10)
        self.create_subscription(PoseStamped, '/waypoint_pose', self.receive_pose, 10)

    def say(self, message):
        print(message, flush=True)

    def receive(self, msg):
        if msg.header.frame_id != 'map':
            self.say('Please set RViz Fixed Frame to map and select the point again.')
            return
        self.clicked_yaw = None
        self.clicked = (msg.point.x, msg.point.y)
        self.say(f'Selected point {self.clicked}. SAVE A 0 to save; UPDATE A 0 to update.')

    def receive_pose(self, msg):
        # A dedicated topic prevents waypoint selection from sending a Nav2 goal.
        self.clicked = self.clicked_yaw = None
        if msg.header.frame_id != 'map':
            self.say('Please set RViz Fixed Frame to map and drag the arrow again.')
            return
        q = msg.pose.orientation
        values = (msg.pose.position.x, msg.pose.position.y, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(v) for v in values):
            self.say('Pose contains invalid values, please reselect.')
            return
        norm = math.hypot(q.x, q.y, q.z, q.w)
        if norm < 1e-9:
            self.say('Invalid orientation, please drag the arrow again.')
            return
        x, y, z, w = (v / norm for v in (q.x, q.y, q.z, q.w))
        self.clicked_yaw = math.degrees(math.atan2(2 * (w*z + x*y),
                                                   1 - 2 * (y*y + z*z)))
        self.clicked = (msg.pose.position.x, msg.pose.position.y)
        self.say(f'Selected position {self.clicked}, orientation {self.clicked_yaw:.1f}°. '
                 'Enter SAVE B or UPDATE B to save; this action will not navigate.')

    def persist(self, points):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(map_sha256=self.identity, points=points), indent=2) + '\n')
        temporary.replace(self.path)
        self.points = points

    def cancel(self):
        self.route.clear()
        self.cancel_requested = True
        if self.goal_handle is not None and not self.cancel_sent:
            self.cancel_sent = True
            self.goal_handle.cancel_goal_async().add_done_callback(self.canceled)
        if self.current is None:
            self.say('Queue cleared, no active goal currently executing.')

    def canceled(self, future):
        try:
            if not future.result().goals_canceling:
                self.say('Cancellation not confirmed; please check the robot, queue cleared.')
            else:
                self.say('Cancellation request accepted, waiting for task to end.')
        except Exception as error:
            self.say(f'Cancellation communication failed: {error}; please check the robot.')

    def start_next(self):
        if self.current is not None or not self.route:
            return
        if not self.client.server_is_ready():
            self.route.clear()
            self.say('Nav2 is not ready, queue cleared. Please check localization and Nav2.')
            return
        self.current = self.route.popleft()
        self.cancel_requested = self.cancel_sent = False
        point = self.points[self.current]
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(point['x'])
        goal.pose.pose.position.y = float(point['y'])
        angle = math.radians(float(point['yaw_degrees'])) / 2
        goal.pose.pose.orientation.z = math.sin(angle)
        goal.pose.pose.orientation.w = math.cos(angle)
        try:
            self.client.send_goal_async(goal).add_done_callback(self.accepted)
            self.say('Sending goal ' + self.current)
        except Exception as error:
            self.finish(False, f'Failed to send: {error}')

    def accepted(self, future):
        try:
            self.goal_handle = future.result()
            if not self.goal_handle.accepted:
                self.finish(False, 'Goal rejected')
                return
            self.goal_handle.get_result_async().add_done_callback(self.completed)
            if self.cancel_requested:
                self.cancel()
            else:
                self.say('Navigating to ' + self.current)
        except Exception as error:
            self.finish(False, f'Goal response failed: {error}')

    def completed(self, future):
        try:
            status = future.result().status
            success = status == GoalStatus.STATUS_SUCCEEDED
            message = {GoalStatus.STATUS_SUCCEEDED: 'Reached destination', GoalStatus.STATUS_CANCELED: 'Canceled',
                       GoalStatus.STATUS_ABORTED: 'Navigation failed'}.get(status, f'Status {status}')
            self.finish(success, message)
        except Exception as error:
            self.finish(False, f'Result communication failed: {error}')

    def finish(self, success, message):
        self.say(f'{self.current}: {message}')
        if not success or self.cancel_requested:
            self.route.clear()
        self.current = self.goal_handle = None
        if not self.route:
            self.say('Queue finished.')

    def command(self, line):
        words = line.replace(',', ' ').replace('，', ' ').split()
        if not words:
            return
        op = words[0].upper()
        if op == 'CANCEL':
            self.cancel()
        elif op == 'LIST':
            self.say(json.dumps(self.points, indent=2, ensure_ascii=False))
        elif op == 'STATUS':
            self.say(f'Current: {self.current}; Pending: {list(self.route)}')
        elif op in ('SAVE', 'UPDATE') and len(words) in (2, 3):
            if self.current or self.route:
                raise ValueError('Cannot modify waypoints during navigation; please CANCEL first and wait for it to stop.')
            if self.clicked is None:
                raise ValueError('Please select a point first using the dedicated arrow tool or Publish Point.')
            name = words[1].upper()
            yaw = float(words[2]) if len(words) == 3 else self.clicked_yaw
            if yaw is None:
                raise ValueError('Publish Point does not include orientation; please enter SAVE <NAME> <ANGLE>, or use the dedicated arrow tool.')
            if not all(math.isfinite(v) for v in (*self.clicked, yaw)):
                raise ValueError('Invalid coordinates or angle')
            if (op == 'SAVE' and name in self.points) or (op == 'UPDATE' and name not in self.points):
                raise ValueError('Use SAVE for new waypoints, and UPDATE for existing waypoints.')
            updated = dict(self.points)
            updated[name] = dict(x=self.clicked[0], y=self.clicked[1], yaw_degrees=yaw)
            self.persist(updated)
            self.clicked = self.clicked_yaw = None
            self.say(f'Saved {name}. You must re-click the map before saving the next point.')
        elif op == 'GO':
            if self.current or self.route:
                raise ValueError('Task already running; please CANCEL first and wait for it to stop.')
            self.route.extend(parse_route(words[1:], self.points))
            self.say('Execution order: ' + ' -> '.join(self.route))
        else:
            self.say('SAVE A [angle] / UPDATE A [angle] / LIST / GO A B C / STATUS / CANCEL / QUIT')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', required=True, help='Same map as loaded by localization')
    parser.add_argument('--points', required=True)
    args = parser.parse_args()
    identity = map_identity(Path(args.map).expanduser().resolve())
    rclpy.init()
    node = PointConsole(Path(args.points).expanduser().resolve(), identity)
    commands = queue.Queue()

    def read_input():
        while True:
            try:
                line = input()
            except EOFError:
                commands.put('QUIT')
                return
            commands.put(line)
            if line.strip().upper() == 'QUIT':
                return

    threading.Thread(target=read_input, daemon=True).start()
    node.say('First confirm localization uses the specified map, and use 2D Pose Estimate to complete localization.')
    node.say('SAVE A [angle] / UPDATE A [angle] / LIST / GO A B C / STATUS / CANCEL / QUIT')
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            try:
                line = commands.get_nowait()
            except queue.Empty:
                line = ''
            if line.strip().upper() == 'QUIT':
                break
            try:
                node.command(line)
            except (ValueError, KeyError, OSError) as error:
                node.say(f'Not executed: {error}')
            node.start_next()
    except KeyboardInterrupt:
        pass
    finally:
        if node.current and rclpy.ok():
            node.cancel()
            deadline = time.monotonic() + 5
            while node.current and rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
            if node.current:
                node.say('Cancellation not confirmed, please check the robot and use manual/physical stop if needed.')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
