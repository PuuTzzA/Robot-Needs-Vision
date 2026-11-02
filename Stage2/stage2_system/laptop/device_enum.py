"""Hardware video capture device enumeration and virtual camera detection for Windows.

Probes DirectShow video capture devices via Windows COM interfaces, identifying friendly
names (e.g. 'H264 USB Camera', 'Integrated Camera', 'OBS Virtual Camera') and unique
hardware USB paths. Automatically detects and flags virtual cameras so they cannot hijack
stereo vision channels or shift device index assignments.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import List, Optional

VIRTUAL_NAME_KEYWORDS = (
    "obs",
    "virtual",
    "xsplit",
    "manycam",
    "vysor",
    "droidcam",
    "iriun",
    "snap camera",
    "ipwebcam",
    "unity",
    "epoccam",
)

INTEGRATED_NAME_KEYWORDS = (
    "integrated",
    "internal",
    "ir camera",
    "facetime",
    "front camera",
    "rear camera",
    "built-in",
    "webcam internal",
)


@dataclass
class VideoDevice:
    """Represents a discovered video capture device."""

    index: int
    name: str
    path: str = ""
    is_virtual: bool = False
    is_integrated: bool = False

    @property
    def label(self) -> str:
        """Formatted descriptive tag for CLI / UI displays."""
        tags = []
        if self.is_virtual:
            tags.append("VIRTUAL")
        elif self.is_integrated:
            tags.append("INTERNAL")
        else:
            tags.append("USB")
        return f"{self.name} [{', '.join(tags)}]"

    def short_path(self) -> str:
        """Return a shortened USB port identifier from the device path if available."""
        if not self.path:
            return ""
        # Example path: \\?\usb#vid_32e4&pid_9422&mi_00#8&35b9755f&0&0000#{...}
        parts = self.path.split("#")
        if len(parts) >= 3:
            return parts[2]
        return self.path[-12:]


def is_virtual_camera(name: str, path: str = "") -> bool:
    """Check if the given device name or path indicates a virtual camera filter."""
    lower_name = name.lower()
    lower_path = path.lower()
    for kw in VIRTUAL_NAME_KEYWORDS:
        if kw in lower_name or kw in lower_path:
            return True
    return False


def is_integrated_camera(name: str) -> bool:
    """Check if the given device name indicates an internal laptop webcam."""
    lower_name = name.lower()
    for kw in INTEGRATED_NAME_KEYWORDS:
        if kw in lower_name:
            return True
    return False


_CACHE_DEVICES: Optional[List[VideoDevice]] = None
_CACHE_TIMESTAMP: float = 0.0
_CACHE_TTL_S: float = 2.0

_PS_ENUM_SCRIPT = r"""
$src = @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;

namespace DShowEnum {
    [ComImport, Guid("62BE5D10-60EB-11d0-BD3B-00A0C911CE86")]
    public class CreateDevEnum {}

    [ComImport, Guid("29840822-5B84-11D0-BD3B-00A0C911CE86"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface ICreateDevEnum {
        [PreserveSig]
        int CreateClassEnumerator([In] ref Guid pType, [Out] out IEnumMoniker ppEnumMoniker, [In] int dwFlags);
    }

    [ComImport, Guid("55272A00-42CB-11CE-8135-00AA004BB851"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IPropertyBag {
        [PreserveSig]
        int Read([In, MarshalAs(UnmanagedType.LPWStr)] string pszPropName, [In, Out, MarshalAs(UnmanagedType.Struct)] ref object pVar, [In] IntPtr pErrorLog);
        [PreserveSig]
        int Write([In, MarshalAs(UnmanagedType.LPWStr)] string pszPropName, [In, MarshalAs(UnmanagedType.Struct)] ref object pVar);
    }

    public class DeviceInfo {
        public int Index { get; set; }
        public string Name { get; set; }
        public string Path { get; set; }
    }

    public class EnumDevices {
        public static List<DeviceInfo> GetVideoDevices() {
            var list = new List<DeviceInfo>();
            Guid videoInput = new Guid("860BB310-5D01-11d0-BD3B-00A0C911CE86");
            ICreateDevEnum devEnum = (ICreateDevEnum)new CreateDevEnum();
            IEnumMoniker enumMoniker;
            int hr = devEnum.CreateClassEnumerator(ref videoInput, out enumMoniker, 0);
            if (hr != 0 || enumMoniker == null) {
                return list;
            }

            IMoniker[] monikers = new IMoniker[1];
            IntPtr fetched = IntPtr.Zero;
            int index = 0;
            Guid bagGuid = new Guid("55272A00-42CB-11CE-8135-00AA004BB851");

            while (enumMoniker.Next(1, monikers, fetched) == 0 && monikers[0] != null) {
                object bagObj;
                monikers[0].BindToStorage(null, null, ref bagGuid, out bagObj);
                IPropertyBag bag = bagObj as IPropertyBag;
                string friendlyName = "(unknown)";
                string devicePath = "";
                if (bag != null) {
                    object val = null;
                    if (bag.Read("FriendlyName", ref val, IntPtr.Zero) == 0 && val != null) {
                        friendlyName = val.ToString();
                    }
                    val = null;
                    if (bag.Read("DevicePath", ref val, IntPtr.Zero) == 0 && val != null) {
                        devicePath = val.ToString();
                    }
                }
                list.Add(new DeviceInfo { Index = index, Name = friendlyName, Path = devicePath });
                Marshal.ReleaseComObject(monikers[0]);
                if (bag != null) Marshal.ReleaseComObject(bag);
                index++;
            }
            Marshal.ReleaseComObject(enumMoniker);
            Marshal.ReleaseComObject(devEnum);
            return list;
        }
    }
}
"@

Add-Type -TypeDefinition $src -ErrorAction SilentlyContinue
$devices = [DShowEnum.EnumDevices]::GetVideoDevices()
if ($devices.Count -gt 0) {
    @($devices) | ConvertTo-Json -Compress
} else {
    "[]"
}
"""


def _query_windows_directshow_devices() -> List[VideoDevice]:
    """Execute PowerShell script to query DirectShow video capture devices."""
    try:
        startupinfo = None
        if platform.system() == "Windows":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                _PS_ENUM_SCRIPT,
            ],
            capture_output=True,
            text=True,
            timeout=8.0,
            startupinfo=startupinfo,
        )

        out = proc.stdout.strip()
        if not out:
            return []

        # Find JSON payload in stdout (ignoring any profile warnings on the first lines)
        json_start = out.find("[")
        if json_start == -1:
            json_start = out.find("{")
        if json_start != -1:
            raw_json = out[json_start:]
            data = json.loads(raw_json)
            if isinstance(data, dict):
                data = [data]
            devices: List[VideoDevice] = []
            for item in data:
                idx = int(item.get("Index", len(devices)))
                name = str(item.get("Name", f"Camera {idx}"))
                path = str(item.get("Path", ""))
                is_virt = is_virtual_camera(name, path)
                is_integ = is_integrated_camera(name)
                devices.append(
                    VideoDevice(
                        index=idx,
                        name=name,
                        path=path,
                        is_virtual=is_virt,
                        is_integrated=is_integ,
                    )
                )
            return devices
    except Exception as err:
        pass
    return []


def enumerate_capture_devices(force_refresh: bool = False) -> List[VideoDevice]:
    """Enumerate system video capture devices with friendly names and categories."""
    global _CACHE_DEVICES, _CACHE_TIMESTAMP

    now = time.monotonic()
    if (
        not force_refresh
        and _CACHE_DEVICES is not None
        and (now - _CACHE_TIMESTAMP) < _CACHE_TTL_S
    ):
        return list(_CACHE_DEVICES)

    if platform.system() == "Windows":
        devices = _query_windows_directshow_devices()
    else:
        # Fallback for Linux/macOS
        devices = []
        for i in range(8):
            devices.append(
                VideoDevice(
                    index=i,
                    name=f"Camera {i}",
                    path=f"/dev/video{i}",
                    is_virtual=False,
                    is_integrated=(i == 0),
                )
            )

    _CACHE_DEVICES = list(devices)
    _CACHE_TIMESTAMP = now
    return list(devices)


def get_device_by_index(index: int, devices: Optional[List[VideoDevice]] = None) -> Optional[VideoDevice]:
    """Lookup a VideoDevice by its DirectShow index."""
    if devices is None:
        devices = enumerate_capture_devices()
    for d in devices:
        if d.index == index:
            return d
    return None


def find_matching_device_index(
    target_path: Optional[str] = None,
    target_name: Optional[str] = None,
    fallback_index: Optional[int] = None,
    exclude_indices: Optional[set[int]] = None,
    devices: Optional[List[VideoDevice]] = None,
) -> Optional[int]:
    """Find the best matching camera index in the current device list.

    Matching strategy:
    1. Exact hardware path match (most stable across OBS opening/closing and port shifts).
    2. Name match (skipping virtual cameras and excluded indices).
    3. Fallback index (if valid and not virtual).
    """
    if devices is None:
        devices = enumerate_capture_devices()

    if exclude_indices is None:
        exclude_indices = set()

    # 1. Exact hardware path match
    if target_path:
        for d in devices:
            if d.index not in exclude_indices and d.path and d.path.lower() == target_path.lower():
                return d.index

    # 2. Friendly name match (excluding virtual cameras)
    if target_name:
        for d in devices:
            if d.index not in exclude_indices and not d.is_virtual:
                if d.name.lower() == target_name.lower():
                    return d.index

    # 3. Fallback index if still valid and not virtual
    if fallback_index is not None and fallback_index not in exclude_indices:
        for d in devices:
            if d.index == fallback_index:
                if not d.is_virtual:
                    return fallback_index

    return None

