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
        raise ValueError('请先保存队列中的所有点位：' + ' '.join(names))
    return names


class PointConsole(Node):
    def __init__(self, path, identity):
        super().__init__('named_point_console')
        self.path, self.identity = path, identity
        self.points = {}
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get('map_sha256') != identity:
                raise ValueError('点位文件不属于这张地图；请指定新的 --points 文件。')
            self.points = saved['points']
            for point in self.points.values():
                if not all(math.isfinite(float(point[k])) for k in ('x', 'y', 'yaw_degrees')):
                    raise ValueError('点位包含无效坐标')
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
            self.say('请将 RViz Fixed Frame 设置为 map 后重新选点。')
            return
        self.clicked_yaw = None
        self.clicked = (msg.point.x, msg.point.y)
        self.say(f'选点 {self.clicked}。SAVE A 0 保存；UPDATE A 0 更新。')

    def receive_pose(self, msg):
        # A dedicated topic prevents waypoint selection from sending a Nav2 goal.
        self.clicked = self.clicked_yaw = None
        if msg.header.frame_id != 'map':
            self.say('请将 RViz Fixed Frame 设置为 map 后重新拖箭头。')
            return
        q = msg.pose.orientation
        values = (msg.pose.position.x, msg.pose.position.y, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(v) for v in values):
            self.say('位姿包含无效数值，请重新选择。')
            return
        norm = math.hypot(q.x, q.y, q.z, q.w)
        if norm < 1e-9:
            self.say('朝向无效，请重新拖箭头。')
            return
        x, y, z, w = (v / norm for v in (q.x, q.y, q.z, q.w))
        self.clicked_yaw = math.degrees(math.atan2(2 * (w*z + x*y),
                                                   1 - 2 * (y*y + z*z)))
        self.clicked = (msg.pose.position.x, msg.pose.position.y)
        self.say(f'已选位置 {self.clicked}，朝向 {self.clicked_yaw:.1f}°。'
                 '输入 SAVE B 或 UPDATE B 保存；此操作不会导航。')

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
            self.say('队列已清空，没有正在执行的目标。')

    def canceled(self, future):
        try:
            if not future.result().goals_canceling:
                self.say('取消未被确认；请检查机器人，队列已清空。')
            else:
                self.say('取消请求已接受，等待任务结束。')
        except Exception as error:
            self.say(f'取消通信失败：{error}；请检查机器人。')

    def start_next(self):
        if self.current is not None or not self.route:
            return
        if not self.client.server_is_ready():
            self.route.clear()
            self.say('Nav2 未就绪，队列已清空。请检查定位和 Nav2。')
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
            self.say('发送目标 ' + self.current)
        except Exception as error:
            self.finish(False, f'发送失败：{error}')

    def accepted(self, future):
        try:
            self.goal_handle = future.result()
            if not self.goal_handle.accepted:
                self.finish(False, '目标被拒绝')
                return
            self.goal_handle.get_result_async().add_done_callback(self.completed)
            if self.cancel_requested:
                self.cancel()
            else:
                self.say('正在前往 ' + self.current)
        except Exception as error:
            self.finish(False, f'目标应答失败：{error}')

    def completed(self, future):
        try:
            status = future.result().status
            success = status == GoalStatus.STATUS_SUCCEEDED
            message = {GoalStatus.STATUS_SUCCEEDED: '已到达', GoalStatus.STATUS_CANCELED: '已取消',
                       GoalStatus.STATUS_ABORTED: '导航失败'}.get(status, f'状态 {status}')
            self.finish(success, message)
        except Exception as error:
            self.finish(False, f'结果通信失败：{error}')

    def finish(self, success, message):
        self.say(f'{self.current}: {message}')
        if not success or self.cancel_requested:
            self.route.clear()
        self.current = self.goal_handle = None
        if not self.route:
            self.say('队列结束。')

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
            self.say(f'当前：{self.current}；待执行：{list(self.route)}')
        elif op in ('SAVE', 'UPDATE') and len(words) in (2, 3):
            if self.current or self.route:
                raise ValueError('导航中不能修改点位，请先 CANCEL 并等待结束。')
            if self.clicked is None:
                raise ValueError('请先用专用箭头工具或 Publish Point 重新选点。')
            name = words[1].upper()
            yaw = float(words[2]) if len(words) == 3 else self.clicked_yaw
            if yaw is None:
                raise ValueError('Publish Point 不含方向，请输入 SAVE 名称 角度，或用专用箭头工具。')
            if not all(math.isfinite(v) for v in (*self.clicked, yaw)):
                raise ValueError('坐标或角度无效')
            if (op == 'SAVE' and name in self.points) or (op == 'UPDATE' and name not in self.points):
                raise ValueError('新点用 SAVE，已有点用 UPDATE。')
            updated = dict(self.points)
            updated[name] = dict(x=self.clicked[0], y=self.clicked[1], yaw_degrees=yaw)
            self.persist(updated)
            self.clicked = self.clicked_yaw = None
            self.say(f'已保存 {name}。保存下一个点前必须重新点击地图。')
        elif op == 'GO':
            if self.current or self.route:
                raise ValueError('已有任务，请先 CANCEL 并等待结束。')
            self.route.extend(parse_route(words[1:], self.points))
            self.say('执行顺序：' + ' → '.join(self.route))
        else:
            self.say('SAVE A [角度] / UPDATE A [角度] / LIST / GO A B C / STATUS / CANCEL / QUIT')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', required=True, help='与 localization 加载的地图相同')
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
    node.say('先确认 localization 使用指定地图，并用 2D Pose Estimate 完成定位。')
    node.say('SAVE A [角度] / UPDATE A [角度] / LIST / GO A B C / STATUS / CANCEL / QUIT')
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
                node.say(f'未执行：{error}')
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
                node.say('未确认取消成功，请检查机器人并使用现场停止手段。')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

