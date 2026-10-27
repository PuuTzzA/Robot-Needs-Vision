#!/usr/bin/env python3
"""Preview USB cameras and diagnose negotiated formats on Linux, Windows or macOS.

Ubuntu/VMware examples (use modes listed by v4l2-ctl --list-formats-ext):
    python3 Stage1/camera_test.py --cameras 0 --format MJPG
    python3 Stage1/camera_test.py --cameras 0 --format YUYV --fps 10
"""
import argparse
import math
import platform


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cameras', type=int, nargs='+', default=[0],
                        help='Camera indices; start with one camera (default: 0)')
    parser.add_argument('--format', choices=['MJPG', 'YUYV', 'auto'], default='MJPG',
                        help='Requested format; auto leaves the driver default unchanged')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=float, default=15)
    args = parser.parse_args()
    if (args.width <= 0 or args.height <= 0 or not math.isfinite(args.fps)
            or args.fps <= 0 or any(idx < 0 for idx in args.cameras)):
        parser.error('Resolution and FPS must be positive; camera indices must be nonnegative.')
    return args


def fourcc_name(value):
    if not math.isfinite(value) or value <= 0:
        return 'unknown'
    code = int(value)
    return ''.join(chr((code >> (8 * i)) & 255) for i in range(4))


def main():
    args = parse_args()
    import cv2

    system = platform.system()
    backend = {'Windows': cv2.CAP_DSHOW, 'Darwin': cv2.CAP_AVFOUNDATION,
               'Linux': cv2.CAP_V4L2}.get(system, cv2.CAP_ANY)
    print(f'Running inside {system}; selecting its camera backend (not the host OS).')
    print(f'Requested: {args.format}, {args.width}x{args.height}, {args.fps:g} FPS')
    print('Settings are requests, not guarantees. Compare the reported mode and actual frames.')
    cameras, failures, reported = {}, {}, set()
    try:
        for idx in dict.fromkeys(args.cameras):
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                print(f'[FAIL] Camera {idx}: cannot open. Check index, USB passthrough and permissions.')
                cap.release()
                continue
            cameras[idx] = cap
            failures[idx] = 0
            settings = []
            if args.format != 'auto':
                settings.append(('format', cv2.CAP_PROP_FOURCC,
                                 cv2.VideoWriter_fourcc(*args.format)))
            settings.extend([
                ('width', cv2.CAP_PROP_FRAME_WIDTH, args.width),
                ('height', cv2.CAP_PROP_FRAME_HEIGHT, args.height),
                ('FPS', cv2.CAP_PROP_FPS, args.fps),
                ('color conversion', cv2.CAP_PROP_CONVERT_RGB, 1),
            ])
            for label, prop, value in settings:
                if not cap.set(prop, value):
                    print(f'[WARNING] Camera {idx}: driver did not accept {label}.')
            print(f'[OPEN] Camera {idx}: backend={cap.getBackendName()}, '
                  f'format={fourcc_name(cap.get(cv2.CAP_PROP_FOURCC))!r}, '
                  f'reported size={cap.get(cv2.CAP_PROP_FRAME_WIDTH):g}x'
                  f'{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):g}, '
                  f'reported FPS={cap.get(cv2.CAP_PROP_FPS):g}')

        if not cameras:
            print('No cameras opened. On Ubuntu, inspect: v4l2-ctl --list-devices')
            return 1
        print("Preview requires a desktop session. Press q or Esc in a preview window to quit.")
        while cameras:
            for idx, cap in list(cameras.items()):
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    failures[idx] += 1
                    if failures[idx] in (1, 30):
                        print(f'[WARNING] Camera {idx}: {failures[idx]} consecutive failed frame reads.')
                    if failures[idx] >= 30:
                        cap.release()
                        del cameras[idx]
                        if idx in reported:
                            cv2.destroyWindow(f'Camera {idx}')
                    continue
                failures[idx] = 0
                if idx not in reported:
                    print(f'[FRAME] Camera {idx}: shape={frame.shape}, dtype={frame.dtype}; '
                          f'actual width={frame.shape[1]}, height={frame.shape[0]}')
                    reported.add(idx)
                cv2.imshow(f'Camera {idx}', frame)
            if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
                return 0
        print('All cameras stopped producing frames. Try another supported format/mode.')
        return 1
    except KeyboardInterrupt:
        return 0
    except cv2.error as error:
        print(f'[ERROR] OpenCV: {error}')
        return 1
    finally:
        for cap in cameras.values():
            cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == '__main__':
    raise SystemExit(main())
