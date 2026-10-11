"""Master Runner for Energy-Aware Multi-Agent DRL in 5G O-RAN.

Supports:
1. Streamlit Web Dashboard: `streamlit run main.py` or `streamlit run streamlit_app.py`
2. Automated CLI Modes: `python main.py --mode [train|evaluate|compare|plot|test|dashboard]`
3. Interactive CLI Menu: `python main.py` in an interactive terminal.
"""

import argparse
import os
import subprocess
import sys


def is_running_in_streamlit() -> bool:
    """Detect whether this process is running inside a Streamlit web server."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_context
        if get_script_run_context() is not None:
            return True
    except Exception:
        pass

    # Detect if invoked via `streamlit run main.py`
    if any("streamlit" in str(arg).lower() for arg in sys.argv):
        return True

    # Detect Streamlit server environment variables
    if os.environ.get("STREAMLIT_SERVER_PORT") or os.environ.get("STREAMLIT_SERVER_HEADLESS"):
        return True

    return False


def print_banner() -> None:
    print("=" * 80)
    print(" 5G O-RAN Energy-Aware Dynamic Slicing & xApp Sleeping (MARL)")
    print(" Multi-Agent Deep Reinforcement Learning (MAPPO with CTDE)")
    print("=" * 80)


def run_tests() -> int:
    print("\n[Running Test Suite via PyTest]...")
    return subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"]).returncode


def run_training(config_path: str, episodes: int = 0) -> None:
    from train import train_mappo
    from utils.config import load_config
    cfg = load_config(config_path)
    train_mappo(cfg, episodes_override=episodes)


def run_evaluation(config_path: str, checkpoint: str, episodes: int = 10) -> None:
    from evaluate import evaluate_checkpoint
    from utils.config import load_config
    cfg = load_config(config_path)
    evaluate_checkpoint(cfg, checkpoint_path=checkpoint, episodes=episodes)


def run_comparison(config_path: str, checkpoint: str, episodes: int = 10) -> None:
    from compare_baselines import run_baseline_comparison
    from utils.config import load_config
    cfg = load_config(config_path)
    run_baseline_comparison(cfg, checkpoint_path=checkpoint, episodes=episodes)


def run_plotting(config_path: str) -> None:
    from plot_results import plot_all_results
    from utils.config import load_config
    cfg = load_config(config_path)
    plot_all_results(cfg)


def run_dashboard() -> None:
    """Launch the Streamlit web dashboard."""
    print("\n[Starting Streamlit Web Dashboard]...")
    cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8501"]
    subprocess.run(cmd)


def main():
    # If run through Streamlit (`streamlit run main.py`), immediately render web UI!
    if is_running_in_streamlit():
        try:
            from streamlit_app import render_dashboard
            render_dashboard()
            return
        except Exception as e:
            import streamlit as st
            st.error(f"Failed to render Streamlit dashboard: {e}")
            return

    parser = argparse.ArgumentParser(
        description="Master runner for Energy-Aware Multi-Agent DRL in 5G O-RAN"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dashboard", "train", "evaluate", "compare", "plot", "test", "menu"],
        default="dashboard",
        help="Execution mode (default: dashboard for web execution, or CLI commands)",
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mappo_final.pt", help="Path to checkpoint")
    parser.add_argument("--episodes", type=int, default=0, help="Number of episodes (0 to use config default)")

    args = parser.parse_args()
    print_banner()

    if args.mode == "dashboard":
        run_dashboard()
    elif args.mode == "test":
        sys.exit(run_tests())
    elif args.mode == "train":
        run_training(args.config, episodes=args.episodes)
    elif args.mode == "evaluate":
        ep_count = args.episodes if args.episodes > 0 else 10
        run_evaluation(args.config, args.checkpoint, episodes=ep_count)
    elif args.mode == "compare":
        ep_count = args.episodes if args.episodes > 0 else 10
        run_comparison(args.config, args.checkpoint, episodes=ep_count)
    elif args.mode == "plot":
        run_plotting(args.config)
    elif args.mode == "menu":
        # Check if running in an interactive terminal to prevent hangs in automated environments
        if not sys.stdin.isatty():
            print("\nNon-interactive environment detected. Run with '--mode train', '--mode compare', or '--mode dashboard'.")
            return

        while True:
            print("\nPlease select an action:")
            print("1. Launch Interactive Streamlit Web Dashboard")
            print("2. Run Automated Unit & Integration Tests")
            print("3. Train Multi-Agent DRL (MAPPO) Agents")
            print("4. Evaluate Trained Checkpoint")
            print("5. Compare all 5 Baselines (Table & CSV Export)")
            print("6. Generate Publication Plots (PNG)")
            print("7. Exit")

            try:
                choice = input("\nEnter choice [1-7]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if choice == "1":
                run_dashboard()
                break
            elif choice == "2":
                run_tests()
            elif choice == "3":
                ep_in = input("Enter number of episodes (press Enter for default in config.yaml): ").strip()
                eps = int(ep_in) if ep_in.isdigit() else 0
                run_training(args.config, episodes=eps)
            elif choice == "4":
                run_evaluation(args.config, args.checkpoint, episodes=10)
            elif choice == "5":
                run_comparison(args.config, args.checkpoint, episodes=10)
            elif choice == "6":
                run_plotting(args.config)
            elif choice == "7":
                print("Exiting. Thank you!")
                break
            else:
                print("Invalid selection. Please choose 1 to 7.")


if __name__ == "__main__":
    main()
