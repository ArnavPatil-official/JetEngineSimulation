"""Portable process ownership, power metadata, and sleep inhibition.

Importing this module does not launch work or alter the power policy.
"""
from __future__ import annotations

import ctypes
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path


def host_identity():
    return socket.gethostname()


def process_birth(pid):
    if not isinstance(pid, int) or pid <= 0:
        return None
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return "<dead>" if ctypes.get_last_error() == 87 else None
        created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
        try:
            exit_code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle,ctypes.byref(exit_code)):
                return None
            if exit_code.value != 259:
                return "<dead>"
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                          ctypes.byref(system), ctypes.byref(user)):
                return None
            return str((created.dwHighDateTime << 32) | created.dwLowDateTime)
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "<dead>"
    except PermissionError:
        pass
    except OSError:
        return None
    if platform.system() == "Linux":
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            # Fields after the final ')' begin with field 3 (state).
            return Path("/proc/sys/kernel/random/boot_id").read_text().strip() + ":" + stat.rsplit(")", 1)[1].split()[19]
        except FileNotFoundError:
            return "<dead>"
        except (OSError, IndexError):
            return None
    result = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True)
    return " ".join(result.stdout.split()) or None


def liveness(pid, birth, *, host=None):
    if host is not None and host != host_identity():
        return "foreign_host"
    current = process_birth(pid)
    if current is None:
        return "unknown"
    if current == "<dead>":
        return "dead"
    return "alive" if current == birth else "reused"


def power_status():
    system = platform.system()
    if system == "Darwin":
        result = subprocess.run(["pmset", "-g", "batt"], check=True, capture_output=True, text=True)
        return {"platform": system, "on_ac": "AC Power" in result.stdout,
                "raw": result.stdout, "source": "pmset -g batt"}
    if system == "Windows":
        class Status(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
        status = Status()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            raise RuntimeError("GetSystemPowerStatus failed")
        no_battery = status.BatteryFlag == 128
        return {"platform": system, "battery_present": not no_battery,
                "on_ac": status.ACLineStatus == 1 or no_battery,
                "ac_line_status": int(status.ACLineStatus), "battery_flag": int(status.BatteryFlag),
                "source": "GetSystemPowerStatus"}
    if system == "Linux":
        from scripts.phase8.pc_runtime import read_power_linux
        return {"platform": system, **read_power_linux(), "source": "Linux power_supply sysfs"}
    raise RuntimeError(f"Unsupported execution platform: {system}")


def require_ac():
    power = power_status()
    if not power["on_ac"]:
        raise RuntimeError("Scientific work requires connected external power")
    return power


class SleepInhibitor:
    def __init__(self):
        self.child = None
        self.metadata = {"platform": platform.system(), "active": False}

    def __enter__(self):
        system = platform.system()
        if system == "Darwin":
            self.child = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
            self.metadata.update(active=True, method="caffeinate -i", pid=self.child.pid)
        elif system == "Windows":
            result = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            if not result:
                raise RuntimeError("Windows sleep inhibition failed")
            self.metadata.update(active=True, method="SetThreadExecutionState")
        elif system == "Linux":
            import shutil
            if shutil.which("systemd-inhibit"):
                self.child = subprocess.Popen(["systemd-inhibit", "--what=sleep", "--mode=block",
                    "--who=catjet-pc", "--why=Phase8 scientific pipeline", sys.executable, "-c",
                    "import time; time.sleep(31536000)"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    self.child.wait(timeout=.2)
                except subprocess.TimeoutExpired:
                    self.metadata.update(active=True, method="systemd-inhibit", pid=self.child.pid)
                else:
                    self.metadata.update(method="systemd-inhibit unavailable in this session", exit_code=self.child.returncode)
            else:
                self.metadata["method"] = "no systemd-inhibit available"
        return self

    def __exit__(self, *_):
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
            self.child.wait()
        if platform.system() == "Windows" and self.metadata["active"]:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
