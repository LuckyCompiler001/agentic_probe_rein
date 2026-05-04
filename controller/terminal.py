class TerminalFrontend:
    """Frontend backed by stdin/stdout. Mirrors the original CLI behaviour."""

    def _prompt(self, text: str) -> str:
        print(f"\n{text}")
        return input(">> ")

    def ask_yn(self, question: str, default: bool | None = None) -> bool:
        while True:
            raw = self._prompt(question).strip().upper()
            if raw == "" and default is not None:
                return default
            if raw in ("Y", "N"):
                return raw == "Y"
            print("  Invalid input — please enter Y or N.")

    def ask_int_range(self, question: str, lo: int, hi: int) -> int:
        while True:
            raw = self._prompt(question).strip()
            if raw.isdigit():
                value = int(raw)
                if lo <= value <= hi:
                    return value
            print(f"  Invalid input — please enter a number between {lo} and {hi}.")

    def ask_pos_int(self, question: str, default: int = 3) -> int:
        while True:
            raw = self._prompt(question).strip()
            if raw == "":
                return default
            if raw.isdigit() and int(raw) > 0:
                return int(raw)
            print(
                f"  Invalid input — please enter a positive integer, "
                f"or press Enter for default ({default})."
            )

    def ask_text(self, question: str) -> str:
        return self._prompt(question).strip()

    def show(self, message: str) -> None:
        print(message)

    def show_artifact(self, name: str, content: str) -> None:
        print("\n" + content)

    def select_run(self, existing: list[str]) -> str | None:
        if not existing:
            return None
        print(f"\nExisting runs: {', '.join(existing)}")
        if not self.ask_yn("Resume a previous run? (Y/N)"):
            return None
        while True:
            run_id = self._prompt(
                "Enter the run ID (the number shown above) to resume:"
            ).strip()
            if run_id in existing:
                return run_id
            print(f"  Run '{run_id}' not found. Please enter one of: {', '.join(existing)}")

    def select_project(self, available: list[str], default: str | None = None) -> str:
        if not available:
            raise RuntimeError(
                "No project folders found under project/. "
                "Add at least one workspace folder before starting a run."
            )
        print("\nAvailable projects:")
        for i, name in enumerate(available, 1):
            marker = "  ← default" if name == default else ""
            print(f"  [{i}] {name}{marker}")
        prompt_text = "Pick a project by number"
        if default and default in available:
            prompt_text += f" (or press Enter for '{default}')"
        prompt_text += ":"
        while True:
            raw = self._prompt(prompt_text).strip()
            if raw == "" and default and default in available:
                return default
            if raw.isdigit():
                idx = int(raw)
                if 1 <= idx <= len(available):
                    return available[idx - 1]
            if raw in available:
                return raw
            print(f"  Invalid choice. Enter a number 1..{len(available)} or one of: {', '.join(available)}")
