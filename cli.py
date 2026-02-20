import argparse
import sys


def main():
    p = argparse.ArgumentParser(prog="scanner")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="Start web UI server")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)

    st = sub.add_parser("step", help="Step turntable by raw steps (quick test)")
    st.add_argument("steps", type=int)
    st.add_argument("--speed", type=float, default=800.0)
    st.add_argument("--hold", type=int, default=1)

    args = p.parse_args()

    if args.cmd == "serve":
        from .webapp import run_server
        run_server(host=args.host, port=args.port)

    elif args.cmd == "step":
        from .hardware_io import gpio_open, stepper_init, stepper_enable, stepper_step
        h = gpio_open()
        stp = stepper_init(h)  # uses defaults
        stepper_enable(h, stp, True)
        stepper_step(h, stp, args.steps, speed_sps=args.speed, hold=bool(args.hold))

    else:
        p.print_help()
        sys.exit(1)
