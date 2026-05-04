import sys

from controller import TerminalFrontend, run_pipeline, setup_run


def main() -> int:
    frontend = TerminalFrontend()
    ctx = None
    try:
        ctx = setup_run(frontend)
        run_pipeline(ctx, frontend)
        return 0
    except KeyboardInterrupt:
        if ctx is not None:
            print(
                f"\n\nInterrupted. To resume, run `python main.py` again "
                f"and choose to resume run {ctx.response_dir.name}."
            )
        else:
            print("\n\nInterrupted before run setup completed.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
