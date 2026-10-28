"""Mission Control Tkinter Graphical User Interface.

Faithfully implements the layout specified in the project mockup:
- Top Header: Status banner and Level indicator
- Upper Left: Left camera live view
- Upper Center: Right camera live view
- Lower Left: Last processed frame (photo or stereo size measurement)
- Lower Center: Live occupancy map with robot pose, heading, and keypoints
- Right Sidebar: Command cheat-sheet, text command input, and live console output
- Bottom Banner: Multi-line Result display for current level
"""

from __future__ import annotations

from collections import deque
import io
import queue
import threading
import time
import tkinter as tk
from tkinter import font as tkfont, ttk
from typing import TYPE_CHECKING, Optional
import cv2
import numpy as np
from PIL import Image, ImageTk

if TYPE_CHECKING:
    from .commands import CommandDispatcher
    from .map_view import MapRenderer
    from .mission import MissionManager

from shared.protocol import RobotStatus


class MissionControlGUI:
    """Tkinter Desktop Application for Stage 2 Autonomous Mission Control."""

    def __init__(
        self,
        root: tk.Tk,
        mission: MissionManager,
        dispatcher: CommandDispatcher,
        map_renderer: MapRenderer,
    ) -> None:
        self.root = root
        self.m = mission
        self.dispatcher = dispatcher
        self.map_renderer = map_renderer

        self.root.title("TurtleBot4 - Stage 2 Autonomous Mission Control")
        self.root.geometry("1480x880")
        self.root.minsize(1200, 720)
        self.root.configure(bg="#22252a")

        # Command history buffer
        self.history: list[str] = []
        self.history_idx: int = -1

        # Image cache to prevent garbage collection
        self._img_left: Optional[ImageTk.PhotoImage] = None
        self._img_right: Optional[ImageTk.PhotoImage] = None
        self._img_processed: Optional[ImageTk.PhotoImage] = None
        self._img_map: Optional[ImageTk.PhotoImage] = None

        # Build UI layout
        self._build_styles()
        self._build_widgets()

        # Connect callbacks
        self.m.add_listener(self._on_mission_update)
        self.m.robot.add_status_callback(self._on_robot_status)

        # Bind hotkeys
        self.root.bind("<Key>", self._on_key_press)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Periodic GUI update loop
        self._update_loop()

    def _build_styles(self) -> None:
        self.font_title = tkfont.Font(family="Segoe UI", size=22, weight="bold")
        self.font_subtitle = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        self.font_header = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        self.font_body = tkfont.Font(family="Segoe UI", size=10)
        self.font_console = tkfont.Font(family="Consolas", size=9)

    def _build_widgets(self) -> None:
        # Main grid: 2 columns (Left: 72% content, Right: 28% commands/console)
        self.root.columnconfigure(0, weight=7, uniform="cols")
        self.root.columnconfigure(1, weight=3, uniform="cols")
        self.root.rowconfigure(0, weight=1)

        # Left Container
        left_container = tk.Frame(self.root, bg="#1a1c20")
        left_container.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        left_container.rowconfigure(0, weight=0)  # Top banner
        left_container.rowconfigure(1, weight=1, uniform="rows")  # Upper live cams
        left_container.rowconfigure(2, weight=1, uniform="rows")  # Lower processed & map
        left_container.rowconfigure(3, weight=0)  # Bottom results banner
        left_container.columnconfigure(0, weight=1, uniform="left_cols")
        left_container.columnconfigure(1, weight=1, uniform="left_cols")

        # ---------------------------------------------------------------------
        # 1. Top Status Banner (Spans left container cols 0 and 1)
        # ---------------------------------------------------------------------
        top_banner = tk.Frame(left_container, bg="#9ca3af", relief="solid", bd=2)
        top_banner.grid(row=0, column=0, columnspan=2, sticky="ew", padx=3, pady=3)
        top_banner.columnconfigure(0, weight=7)
        top_banner.columnconfigure(1, weight=3)

        self.lbl_status = tk.Label(
            top_banner,
            text="Status: Idle (Ready)",
            font=self.font_title,
            bg="#9ca3af",
            fg="#111827",
            anchor="w",
            padx=12,
            pady=6,
        )
        self.lbl_status.grid(row=0, column=0, sticky="w")

        right_info_box = tk.Frame(top_banner, bg="#9ca3af")
        right_info_box.grid(row=0, column=1, sticky="e", padx=12)

        self.lbl_level = tk.Label(
            right_info_box,
            text="Level: -",
            font=self.font_subtitle,
            bg="#9ca3af",
            fg="#111827",
            anchor="e",
        )
        self.lbl_level.pack(anchor="e")

        self.lbl_extra_info = tk.Label(
            right_info_box,
            text="Possible more info",
            font=self.font_body,
            bg="#9ca3af",
            fg="#374151",
            anchor="e",
        )
        self.lbl_extra_info.pack(anchor="e")

        # ---------------------------------------------------------------------
        # 2. Upper Cameras: Left Camera and Right Camera Live Views
        # ---------------------------------------------------------------------
        # Left Cam
        panel_left_cam = tk.Frame(left_container, bg="#9ca3af", relief="solid", bd=2)
        panel_left_cam.grid(row=1, column=0, sticky="nsew", padx=3, pady=3)
        self.lbl_left_hdr = tk.Label(
            panel_left_cam,
            text="Left camera : -- fps, 1280x720",
            font=self.font_header,
            bg="#374151",
            fg="#f3f4f6",
            anchor="w",
            padx=6,
        )
        self.lbl_left_hdr.pack(fill="x")
        self.canvas_left_cam = tk.Label(panel_left_cam, bg="#111827", text="Waiting for Left Camera...")
        self.canvas_left_cam.pack(fill="both", expand=True)

        # Right Cam
        panel_right_cam = tk.Frame(left_container, bg="#9ca3af", relief="solid", bd=2)
        panel_right_cam.grid(row=1, column=1, sticky="nsew", padx=3, pady=3)
        self.lbl_right_hdr = tk.Label(
            panel_right_cam,
            text="Right camera : -- fps, 1280x720",
            font=self.font_header,
            bg="#374151",
            fg="#f3f4f6",
            anchor="w",
            padx=6,
        )
        self.lbl_right_hdr.pack(fill="x")
        self.canvas_right_cam = tk.Label(panel_right_cam, bg="#111827", text="Waiting for Right Camera...")
        self.canvas_right_cam.pack(fill="both", expand=True)

        # ---------------------------------------------------------------------
        # 3. Lower Panels: Last Processed Frame & Robot Map Pose
        # ---------------------------------------------------------------------
        # Last Processed Frame
        panel_proc = tk.Frame(left_container, bg="#9ca3af", relief="solid", bd=2)
        panel_proc.grid(row=2, column=0, sticky="nsew", padx=3, pady=3)
        self.lbl_proc_hdr = tk.Label(
            panel_proc,
            text="Last Processed Frame: none",
            font=self.font_header,
            bg="#374151",
            fg="#f3f4f6",
            anchor="w",
            padx=6,
        )
        self.lbl_proc_hdr.pack(fill="x")
        self.canvas_proc = tk.Label(
            panel_proc,
            bg="#111827",
            text="Shows the last screenshot or\nlast received depth map / size bounding box",
            fg="#9ca3af",
            font=self.font_subtitle,
        )
        self.canvas_proc.pack(fill="both", expand=True)

        # Robot Location on Map
        panel_map = tk.Frame(left_container, bg="#9ca3af", relief="solid", bd=2)
        panel_map.grid(row=2, column=1, sticky="nsew", padx=3, pady=3)
        map_top_bar = tk.Frame(panel_map, bg="#374151")
        map_top_bar.pack(fill="x")

        tk.Label(map_top_bar, text="Robot: ", font=self.font_header, bg="#374151", fg="#f3f4f6").pack(side="left", padx=4)
        self.lbl_robot_conn = tk.Label(
            map_top_bar, text="Disconnected", font=self.font_header, bg="#374151", fg="#ef4444"
        )
        self.lbl_robot_conn.pack(side="left")

        self.canvas_map = tk.Label(panel_map, bg="#111827", text="Shows the robot's location on the map", fg="#9ca3af")
        self.canvas_map.pack(fill="both", expand=True)

        # ---------------------------------------------------------------------
        # 4. Bottom Results Banner
        # ---------------------------------------------------------------------
        panel_result = tk.Frame(left_container, bg="#9ca3af", relief="solid", bd=2)
        panel_result.grid(row=3, column=0, columnspan=2, sticky="ew", padx=3, pady=3)
        self.lbl_result = tk.Label(
            panel_result,
            text="Result:\n(Awaiting Level execution)",
            font=self.font_subtitle,
            bg="#9ca3af",
            fg="#111827",
            anchor="w",
            justify="left",
            padx=12,
            pady=8,
        )
        self.lbl_result.pack(fill="x")

        # ---------------------------------------------------------------------
        # 5. Right Sidebar: Commands, Input, and Console Out
        # ---------------------------------------------------------------------
        right_container = tk.Frame(self.root, bg="#1a1c20")
        right_container.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        right_container.rowconfigure(0, weight=0)  # Commands cheat-sheet
        right_container.rowconfigure(1, weight=0)  # Command input box
        right_container.rowconfigure(2, weight=1)  # Console Out
        right_container.columnconfigure(0, weight=1)

        # Commands cheat sheet box
        cmd_box = tk.Frame(right_container, bg="#272a30", relief="solid", bd=2)
        cmd_box.grid(row=0, column=0, sticky="ew", padx=3, pady=3)

        cheat_text = (
            "Possible Commands:\n"
            "Level <Level>\n"
            "Restart <Level>\n"
            "Goto <Point>\n"
            "Ping <VM | SSH | LLM>\n"
            "W / A / S / D\n"
            "Photo\n"
            "Measure [hint]\n"
            "Classify\n"
            "Points / Status / Stop"
        )
        lbl_cheat = tk.Label(
            cmd_box,
            text=cheat_text,
            font=self.font_body,
            bg="#272a30",
            fg="#e5e7eb",
            anchor="w",
            justify="left",
            padx=10,
            pady=6,
        )
        lbl_cheat.pack(fill="x")

        # Command input
        input_frame = tk.Frame(right_container, bg="#272a30", relief="solid", bd=2)
        input_frame.grid(row=1, column=0, sticky="ew", padx=3, pady=3)
        tk.Label(input_frame, text="Input:", font=self.font_header, bg="#272a30", fg="#f3f4f6").pack(
            side="left", padx=6, pady=8
        )
        self.entry_cmd = tk.Entry(
            input_frame, font=self.font_body, bg="#111827", fg="#ffffff", insertbackground="#ffffff"
        )
        self.entry_cmd.pack(side="left", fill="x", expand=True, padx=6, pady=6)
        self.entry_cmd.bind("<Return>", self._on_enter_command)
        self.entry_cmd.bind("<Up>", self._on_history_up)
        self.entry_cmd.bind("<Down>", self._on_history_down)

        btn_send = tk.Button(
            input_frame,
            text="Run",
            font=self.font_header,
            bg="#3b82f6",
            fg="#ffffff",
            activebackground="#2563eb",
            relief="flat",
            padx=10,
            command=lambda: self._on_enter_command(None),
        )
        btn_send.pack(side="right", padx=6, pady=6)

        # Console Out
        console_frame = tk.Frame(right_container, bg="#111827", relief="solid", bd=2)
        console_frame.grid(row=2, column=0, sticky="nsew", padx=3, pady=3)
        console_frame.rowconfigure(1, weight=1)
        console_frame.columnconfigure(0, weight=1)

        tk.Label(
            console_frame,
            text="Console Out",
            font=self.font_header,
            bg="#272a30",
            fg="#f3f4f6",
            anchor="w",
            padx=8,
            pady=4,
        ).grid(row=0, column=0, sticky="ew")

        self.txt_console = tk.Text(
            console_frame,
            bg="#111827",
            fg="#d1d5db",
            font=self.font_console,
            wrap="word",
            state="disabled",
            bd=0,
            padx=8,
            pady=6,
        )
        self.txt_console.grid(row=1, column=0, sticky="nsew")

        scroll = ttk.Scrollbar(console_frame, command=self.txt_console.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.txt_console.config(yscrollcommand=scroll.set)

    # -------------------------------------------------------------------------
    # Updates & Event Handlers
    # -------------------------------------------------------------------------

    def _on_key_press(self, event: tk.Event) -> None:
        # Ignore teleop keys if user is typing in the entry input box
        if self.root.focus_get() == self.entry_cmd:
            return

        char = (event.char or "").lower()
        if char == "w":
            self.m.robot.teleop("w")
        elif char == "s":
            self.m.robot.teleop("s")
        elif char == "a":
            self.m.robot.teleop("a")
        elif char == "d":
            self.m.robot.teleop("d")
        elif event.keysym == "space":
            self.m.robot.teleop("space")

    def _on_enter_command(self, _event) -> None:
        cmd_text = self.entry_cmd.get().strip()
        if not cmd_text:
            return
        self.entry_cmd.delete(0, tk.END)
        self.history.append(cmd_text)
        self.history_idx = len(self.history)

        print(f"> {cmd_text}")
        out = self.dispatcher.execute(cmd_text)
        if out:
            print(out)

    def _on_history_up(self, _event) -> None:
        if self.history and self.history_idx > 0:
            self.history_idx -= 1
            self.entry_cmd.delete(0, tk.END)
            self.entry_cmd.insert(0, self.history[self.history_idx])

    def _on_history_down(self, _event) -> None:
        if self.history and self.history_idx < len(self.history) - 1:
            self.history_idx += 1
            self.entry_cmd.delete(0, tk.END)
            self.entry_cmd.insert(0, self.history[self.history_idx])
        else:
            self.history_idx = len(self.history)
            self.entry_cmd.delete(0, tk.END)

    def _on_mission_update(self) -> None:
        """Triggered when mission status, level, or processed frame updates."""
        pass  # Data polled safely inside _update_loop

    def _on_robot_status(self, _status: RobotStatus) -> None:
        pass

    def _update_loop(self) -> None:
        """Main Tkinter 20 Hz update loop."""
        # 1. Update Status & Headers
        self.lbl_status.config(text=f"Status: {self.m.status_text}")
        lvl_str = f"Level: {self.m.current_level}" if self.m.current_level > 0 else "Level: -"
        self.lbl_level.config(text=lvl_str)
        self.lbl_extra_info.config(text=self.m.extra_info or "Standing by")

        # 2. Update Robot Connection Banner
        if self.m.robot.is_connected:
            self.lbl_robot_conn.config(text=f"Connected ({self.m.robot.active_vm_ip})", fg="#22c55e")
        else:
            self.lbl_robot_conn.config(text="Disconnected (Searching...)", fg="#ef4444")

        # 3. Update Result Banner
        cur_lvl = self.m.current_level or 1
        res_text = self.m.level_results.get(cur_lvl, "")
        if res_text:
            self.lbl_result.config(text=f"Result, Level {cur_lvl}:\n{res_text}")
        else:
            self.lbl_result.config(text=f"Result, Level {cur_lvl}:\n(Awaiting task completion)")

        # 4. Update Cameras Live Feed
        left_f, right_f, fps = self.m.camera.get_latest_pair()
        self.lbl_left_hdr.config(text=f"Left camera : {fps:.1f} fps, {self.m.camera.width}x{self.m.camera.height}")
        self.lbl_right_hdr.config(text=f"Right camera : {fps:.1f} fps, {self.m.camera.width}x{self.m.camera.height}")

        self._render_frame_to_label(left_f, self.canvas_left_cam, "_img_left", (440, 260))
        self._render_frame_to_label(right_f, self.canvas_right_cam, "_img_right", (440, 260))

        # 5. Update Last Processed Frame
        if self.m.last_processed_frame is not None:
            self.lbl_proc_hdr.config(text=f"Last Processed Frame: {self.m.last_processed_name}")
            self._render_frame_to_label(self.m.last_processed_frame, self.canvas_proc, "_img_processed", (440, 260))

        # 6. Update Map View
        map_img = self.map_renderer.render(
            keypoints=self.m.keypoints.get_all(),
            robot_status=self.m.robot.latest_status,
            target_keypoint_id=self.m.target_keypoint_id,
            target_size=(440, 260),
        )
        self._img_map = ImageTk.PhotoImage(map_img)
        self.canvas_map.config(image=self._img_map, text="")

        # 7. Drain console text lines
        import sys
        if hasattr(sys.stdout, "get_recent_lines"):
            lines = sys.stdout.get_recent_lines(50)  # type: ignore
            full_text = "\n".join(lines)
            self.txt_console.config(state="normal")
            self.txt_console.delete("1.0", tk.END)
            self.txt_console.insert(tk.END, full_text)
            self.txt_console.see(tk.END)
            self.txt_console.config(state="disabled")

        self.root.after(50, self._update_loop)

    def _render_frame_to_label(
        self,
        frame: Optional[np.ndarray],
        label: tk.Label,
        attr_name: str,
        target_size: tuple[int, int],
    ) -> None:
        if frame is None:
            return
        try:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img.thumbnail(target_size, Image.Resampling.BILINEAR)

            canvas = Image.new("RGB", target_size, (15, 17, 23))
            offset_x = (target_size[0] - img.width) // 2
            offset_y = (target_size[1] - img.height) // 2
            canvas.paste(img, (offset_x, offset_y))

            tk_img = ImageTk.PhotoImage(canvas)
            setattr(self, attr_name, tk_img)
            label.config(image=tk_img, text="")
        except Exception:
            pass

    def _on_close(self) -> None:
        self.m.stop()
        self.m.camera.stop()
        self.m.robot.stop()
        self.root.destroy()

