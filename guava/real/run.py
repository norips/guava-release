"""Run Guava episodes on the real Franka.

Reproduces the evaluation protocol of the released Guava-4B checkpoint
(``guava.inference.benchmark_compatible_tasks`` with ``configs/prompts/sft_v13b_short.txt``):
same system prompt, first-turn tool listing, table-aligned coordinates, one observation
(image + tool feedback + gripper state) after every action, first-tool-call parsing,
temperature 0. Only the simulator is replaced by the robot bridge.

Example (servers: robot bridge, SAM3 on :8114, vLLM on :8000):

    .venv/bin/python -m guava.real.run --config configs/real/franka.yaml \\
        --task "Pick up the carrot from the table." --output runs/real/carrot --episodes 3

Safety: keep a hand on the Franka user stop. With --confirm (default) every physical
action is shown before it runs. Ctrl-C stops the arm and ends the episode.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = ROOT / "configs/prompts/sft_v13b_short.txt"
_MUTATING = {"move", "grasp", "release", "rotate", "align", "close_gripper", "home_pose"}


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def build_context(cfg, *, need_sam3: bool = True):
    """ToolContext wired to the real robot. Shared by run / check / calibrate."""
    from guava.config import RobotConfig
    from guava.real.bridge_client import BridgeClient
    from guava.real.env import RealEnv, RealRobotArm
    from guava.tools.tools import ToolContext

    bridge = BridgeClient(cfg.bridge_url)
    if not bridge.is_available():
        raise SystemExit(f"Robot bridge not reachable at {cfg.bridge_url} (pixi run -e jazzy-crisp guava-bridge)")
    env = RealEnv(bridge, cfg)
    robot_cfg = RobotConfig(home_joints=list(cfg.home_joints), graspgen_url=cfg.graspgen_url, sam3_url=cfg.sam3_url)
    arm = RealRobotArm(env, robot_cfg, cfg)

    perception = None
    if need_sam3:
        from guava.perception.sam3 import SAM3Solver
        perception = SAM3Solver(cfg.sam3_url)
        if not perception.is_available():
            raise SystemExit(f"SAM3 server not reachable at {cfg.sam3_url} "
                             "(.venv-sam3/bin/python -m guava.scripts.sam3_server --port 8114)")
    planner = None
    if cfg.grasp == "graspgen":
        from guava.perception import get_graspgen_client
        planner = get_graspgen_client(robot_cfg, "graspgen")
    ctx = ToolContext(env=env, arm=arm, perception=perception, grasp_planner=planner,
                      model_z_offset=cfg.model_z_offset)
    return ctx


def _write(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _json_value(raw):
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, (dict, list)):
                return decoded
        except ValueError:
            pass
    return raw


def _decide(text: str):
    """First tool call wins (benchmark_compatible_tasks.compatible); else a terminal declaration."""
    from guava.inference.benchmark_engine import decision
    from guava.llm.client import _parse_tool_from_text
    name, args = _parse_tool_from_text(text)
    if name:
        return "tool", {"name": name, "arguments": args}
    return decision(text, "full")  # 'terminal' or ValueError


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return "q"


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

def run_episode(ctx, client, args, cfg, system_prompt: str, folder: Path) -> dict:
    from guava.collection.trajectory import _fmt_tool_result
    from guava.inference import sft_frame
    from guava.inference.rollout_env import _save_png, _to_data_url
    from guava.tools.coordinate_frame import model_registry
    from guava.tools.schema import tools_to_prompt
    from guava.tools.tools import _gripper_state_text, _refresh, make_tool_registry

    folder.mkdir(parents=True, exist_ok=False)
    offset = cfg.model_z_offset
    registry = make_tool_registry(ctx)                       # physical (robot-base) tools
    tool_text = tools_to_prompt(model_registry(ctx, registry))  # model-facing schema (table frame)

    if not args.no_home:
        print("[real] homing and opening the gripper ...")
        ctx.arm.home_pose()
        ctx.env.bridge.open_gripper()
    ctx.arm.init_orientation()
    ctx.was_holding = False

    messages = [{"role": "system", "content": system_prompt}]
    audit_messages, responses, actions = [], [], []
    terminal, end_reason = None, "action_limit"
    started = time.time()

    def observe(result: str = "", initial: bool = False) -> None:
        _refresh(ctx)
        image_path = folder / f"observation_{len(responses):03d}.png"
        _save_png(ctx.last_rgb, image_path)
        gripper = sft_frame.gripper_text(ctx, _gripper_state_text(ctx), offset)
        text = (f"Task: {args.task}\n\nAvailable tools:\n{tool_text}\n\n{gripper}" if initial
                else f"{result}\n{gripper}")
        parts = [{"type": "image_url", "image_url": {"url": _to_data_url(ctx.last_rgb)}},
                 {"type": "text", "text": text}]
        messages.append({"role": "user" if initial else "tool", "content": parts})
        audit_messages.append({"role": messages[-1]["role"], "content": [
            {"type": "image_url", "image_url": {"url": str(image_path.resolve())}}, parts[1]]})
        print(f"\n[observation]\n{text if not initial else gripper}")

    def save_messages():
        _write(folder / "messages.json", [{"role": "system", "content": system_prompt}] + audit_messages)

    try:
        observe(initial=True)
        while len(actions) < args.max_actions and len(responses) < args.max_decisions:
            t0 = time.time()
            response = client.chat.completions.create(model=cfg.model, messages=messages,
                                                      temperature=0, max_tokens=8192)
            text = response.choices[0].message.content or ""
            responses.append({"content": text, "finish_reason": response.choices[0].finish_reason,
                              "latency_s": round(time.time() - t0, 3),
                              "usage": response.usage.model_dump() if response.usage else None})
            _write(folder / "responses.json", responses)
            print(f"\n[model {responses[-1]['latency_s']:.2f}s]\n{text.strip()}")
            if response.choices[0].finish_reason == "length":
                end_reason = "incomplete_model_output"
                break
            try:
                kind, value = _decide(text)
            except ValueError:
                end_reason = "invalid_model_output"
                break
            messages.append({"role": "assistant", "content": text})
            audit_messages.append(messages[-1])
            if kind == "terminal":
                terminal, end_reason = value, "model_terminal"
                break

            name, kwargs = value["name"], value["arguments"]
            entry = {"index": len(actions), "name": name, "kwargs": kwargs, "started": time.time()}
            actions.append(entry)
            if args.confirm and name in _MUTATING:
                ans = _ask(f"\n>>> execute {name}({json.dumps(kwargs)})? [Enter]=run  v=veto  q=stop episode: ")
                if ans == "q":
                    entry.update(ok=False, error="stopped by the operator")
                    end_reason = "operator_stop"
                    break
                if ans == "v":
                    entry.update(ok=False, error="vetoed by the operator")
                    observe(f"Error calling {name}: the operator vetoed this action as unsafe.")
                    save_messages()
                    continue
            try:
                if name not in registry:
                    raise ValueError(f"Unknown tool: {name}")
                physical = sft_frame.physical_arguments(name, kwargs, offset)
                entry["physical_kwargs"] = physical
                raw = registry[name](**physical)
                entry["physical_result"] = _json_value(raw)
                raw = _json_value(sft_frame.model_result(name, raw, offset))
                entry.update(ok=True, result=raw)
                feedback_args = dict(kwargs)
                if name in ("get_position", "get_position_and_size"):
                    feedback_args["object_name"] = feedback_args.get("target_name", feedback_args.get("object_name", "object"))
                feedback = _fmt_tool_result(name, feedback_args,
                                            json.dumps(raw) if isinstance(raw, (dict, list)) else str(raw))
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - errors are model-visible feedback, as in simulation
                entry.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                feedback = f"Error calling {name}: {exc}"
            entry["seconds"] = round(time.time() - entry.pop("started"), 3)
            _write(folder / "actions.json", actions)
            observe(feedback)
            save_messages()
    except KeyboardInterrupt:
        end_reason = "operator_abort"
        print("\n[real] Ctrl-C: stopping the arm")
        try:
            ctx.env.bridge.stop()
        except Exception as exc:  # noqa: BLE001
            print(f"[real] stop failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        end_reason = "infrastructure_error"
        _write(folder / "error.json", {"type": type(exc).__name__, "error": str(exc)})
        print(f"[real] infrastructure error: {type(exc).__name__}: {exc}")
        try:
            ctx.env.bridge.stop()
        except Exception:  # noqa: BLE001
            pass

    save_messages()
    _write(folder / "actions.json", actions)
    try:
        _refresh(ctx)
        _save_png(ctx.last_rgb, folder / "final.png")
    except Exception:  # noqa: BLE001
        pass

    success = None
    if not args.no_label:
        ans = _ask(f"\n[{end_reason}{', model: ' + terminal if terminal else ''}] Task successful? [y/n/Enter=skip]: ")
        success = True if ans.startswith("y") else False if ans.startswith("n") else None
    result = {"task": args.task, "end_reason": end_reason, "terminal": terminal, "success": success,
              "actions": len(actions), "decisions": len(responses), "seconds": round(time.time() - started, 1),
              "mean_model_latency_s": round(sum(r["latency_s"] for r in responses) / max(1, len(responses)), 3)}
    _write(folder / "result.json", result)
    print(json.dumps(result))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "configs/real/franka.yaml")
    p.add_argument("--task", required=True, help='Instruction, e.g. "Put the lemon in the bin."')
    p.add_argument("--output", type=Path, required=True, help="New directory for the episode logs")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-actions", type=int, default=25)
    p.add_argument("--max-decisions", type=int, default=25)
    p.add_argument("--system-prompt", type=Path, default=DEFAULT_PROMPT)
    p.add_argument("--model", default=None, help="Overrides `model` of the config (served name in vLLM)")
    p.add_argument("--no-confirm", dest="confirm", action="store_false",
                   help="Run actions without asking (only once the setup is validated)")
    p.add_argument("--no-home", action="store_true", help="Start from the current pose")
    p.add_argument("--no-label", action="store_true", help="Do not ask for a success label after each episode")
    args = p.parse_args(argv)
    if not 0 < args.max_actions <= 30 or args.max_decisions <= 0:
        p.error("--max-actions must be in 1..30 (vLLM is served with at most 31 images per prompt)")

    from openai import OpenAI
    from guava.real.env import RealConfig

    cfg = RealConfig.load(args.config)
    if args.model:
        cfg.model = args.model
    client = OpenAI(base_url=cfg.llm_url, api_key="EMPTY", timeout=180, max_retries=2)
    served = [m.id for m in client.models.list().data]
    if cfg.model not in served:
        raise SystemExit(f"Model {cfg.model!r} is not served at {cfg.llm_url} (served: {served})")
    system_prompt = args.system_prompt.read_text()  # verbatim, as benchmark_engine reads it

    ctx = build_context(cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    _write(args.output / "protocol.json", {
        "task": args.task, "model": cfg.model, "config": vars(cfg) | {"model_z_offset": cfg.model_z_offset},
        "system_prompt": str(args.system_prompt), "max_actions": args.max_actions,
        "max_decisions": args.max_decisions, "temperature": 0, "max_tokens": 8192,
        "confirm": args.confirm, "argv": sys.argv})
    results = []
    existing = len(list(args.output.glob("episode_*")))
    for i in range(args.episodes):
        if i > 0 and _ask("\nReset the scene, then Enter (q to quit): ") == "q":
            break
        results.append(run_episode(ctx, client, args, cfg, system_prompt,
                                   args.output / f"episode_{existing + i:03d}"))
        _write(args.output / f"summary_{existing:03d}.json", results)
    labelled = [r for r in results if r["success"] is not None]
    if labelled:
        print(f"\nSuccess: {sum(r['success'] for r in labelled)}/{len(labelled)}")


if __name__ == "__main__":
    main()
