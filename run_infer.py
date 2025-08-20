import asyncio
import os
import json
import shutil
import hashlib
import time
import pandas as pd
import traceback
from tqdm import tqdm
from typing import Callable
import inspect
from urllib.parse import urlparse
from pathlib import Path
import hashlib
import subprocess

from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    compatibility_for_eval_history_pairs,
    get_default_sandbox_config_for_eval,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
)
# from openhands.runtime.impl.docker.containers import stop_all_containers, remove_all_containers

from openhands.controller.state.state import State
from openhands.core.config import OpenHandsConfig, get_llm_config_arg, parse_arguments
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import MessageAction
from openhands.utils.async_utils import call_async_from_sync

import re, hashlib

def win_safe_name(s: str, maxlen: int = 120) -> str:
    # 替换 Windows 非法字符  <>:"/\|?* 以及控制字符
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', s)
    # 去掉结尾的点或空格（Windows 不允许）
    safe = safe.rstrip(' .')
    # 太长就截断并加短哈希，避免路径过长问题
    if len(safe) > maxlen:
        digest = hashlib.md5(s.encode('utf-8')).hexdigest()[:8]
        safe = f"{safe[:maxlen-9]}_{digest}"
    return safe


# --- AGENT 相关配置 ---
AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': lambda *args, **kwargs: None
}

# --- 路径安全性: instance_id hash ---
def safe_instance_workspace(instance_id: str) -> str:
    workspace_hash = hashlib.md5(instance_id.encode()).hexdigest()
    return f"/tmp/workspace/{workspace_hash}"

# --- 专业 get_config：每个 instance 独立 hash workspace ---
def get_config(metadata: EvalMetadata, instance_id: str) -> OpenHandsConfig:
    unique_workspace = safe_instance_workspace(instance_id)
    sandbox_config = get_default_sandbox_config_for_eval()
    # if image_tag:
    #     sandbox_config.base_container_image = image_tag  # 直接替换默认镜像

    # 新添加的 >>> PATCH: force use pre-pulled image & disable auto-build
    if hasattr(sandbox_config, "runtime_startup_env_vars"):
        # 直接清空；或至少删除这一个键
        try:
            # 如果只是这个键导致问题
            sandbox_config.runtime_startup_env_vars.pop("NO_CHANGE_TIMEOUT_SECONDS", None)
        except Exception:
            pass

    custom_img = os.environ.get("OPENHANDS_RUNTIME_IMAGE")
    if custom_img:
        # 这些字段名在不同版本里可能叫法略不同；尽量都设一下
        if hasattr(sandbox_config, "base_container_image"):
            sandbox_config.base_container_image = custom_img
        if hasattr(sandbox_config, "container_image"):
            sandbox_config.container_image = custom_img
        # 有些版本用这个开关控制是否构建
        if hasattr(sandbox_config, "auto_build"):
            sandbox_config.auto_build = False
    # 新添加的  <<< PATCH

    config = OpenHandsConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        runtime=os.environ.get("RUNTIME", "docker"),
        max_iterations=metadata.max_iterations,
        sandbox=sandbox_config,
        workspace_base=unique_workspace,
        workspace_mount_path=unique_workspace,
    )
    config.set_llm_config(metadata.llm_config)
    agent_config = config.get_agent_config(metadata.agent_class)
    agent_config.enable_prompt_extensions = False
    return config


# --- 构建指令（可后续扩展 debug/auto-fix 模式）---
# def build_instruction(repo_url, commit):
#     return (
#         "You are an expert Python developer working in a real shell and code environment. "
#         "Your job is to make the provided repo at the specified commit fully pass its evaluation script (main.py or scripts/eval.py). "
#         "If you encounter any error during installation or running, analyze the error logs, reason about the possible cause, "
#         "and then attempt to fix the code or environment. After making changes, re-run the previous failed step. "
#         "Repeat this process (edit → run → debug) until the repo runs successfully or you hit your max iteration budget. "
#         "Do NOT ever ask for human help or explanation. All changes and fixes should be attempted autonomously.\n\n"
#         f"1. git clone {repo_url} repo && cd repo\n"
#         f"2. git checkout {commit}\n"
#         f"3. pip install -r requirements.txt OR pip install .\n"
#         f"4. python scripts/eval.py OR python main.py\n"
#         f"5. On success, output: <solution>{{\"success\": true}}</solution>\n"
#         f"6. On failure (after all reasonable attempts), output error in <solution> and stop.\n"
#     )

import docker

def remove_all_containers(name_filter: str = None):
    """
    Remove all Docker containers with optional name filter.
    If `name_filter` is provided, only containers whose names include the string will be removed.
    """
    client = docker.from_env()
    containers = client.containers.list(all=True)
    for container in containers:
        try:
            if (name_filter is None) or (name_filter in container.name):
                container.remove(force=True)
        except Exception as e:
            print(f"⚠️ Failed to remove container {container.name}: {e}")


# def build_instruction(repo_url, commit):
#     return (
#         f"Make main.py or scripts/eval.py in repo {repo_url} (commit {commit}) run successfully.\n"
#         "You may automatically install dependencies (e.g., pip install -r requirements.txt, pip install .), "
#         "run builds (e.g., make, cmake), and perform any environment setup required for the script to work.\n"
#         "If errors occur, analyze logs and attempt to auto-fix, retrying as needed. Do NOT request human help.\n"
#         "On full success, output <solution>{\"success\":true}</solution>. If still failed, output <solution>{\"success\":false, ...}</solution>.\n"
#     )

def build_instruction(repo_url, commit):
    return (
        f"Repo: {repo_url}\n"
        f"Commit: {commit}\n"
        "Goal: run the repository's evaluation (prefer `python scripts/eval.py`, else `python main.py`).\n"
        "On success, output <solution>{\"success\": true}</solution>; on failure, output <solution>{\"success\": false, \"error\": \"...\"}</solution>."
    )


from openhands.runtime.builder.docker import DockerRuntimeBuilder
from openhands.runtime.utils.runtime_build import build_runtime_image
import docker

# # 构建
# image_name = build_runtime_image(
#     base_image="nikolaik/python-nodejs:python3.12-nodejs22",
#     runtime_builder=DockerRuntimeBuilder(docker.from_env()),
#     platform="linux/amd64",  # optional
#     force_rebuild=False,
# )
# print("Final built image:", image_name)

# def build_and_set_runtime_image(instance_id: str, repo_url: str, repo_commit: str, runtime) -> str:
#     parsed = urlparse(instance_id.split("@")[0])
#     repo_name = Path(parsed.path).name
#     image_tag = f"nenus-ai/{repo_name}_image:{repo_name}_v1"

#     builder = DockerRuntimeBuilder(docker.from_env())

#     if not builder.image_exists(image_tag):
#         builder.build(
#             path=f"/workspace/repos/{repo_name}",  # 你 repo clone 的目录路径（确保和 clone 逻辑一致）
#             tags=[image_tag],
#             platform="linux/amd64"
#         )

    # # 设置 runtime image
    # runtime.container_image = image_tag
    # return image_tag

# --- 提取 metrics ---
def parse_metrics(output: str) -> dict:
    import re
    if not isinstance(output, str):
        return {"success": False, "error": "Output is not a string or is None"}
    match = re.search(r"<solution>(.*?)</solution>", output, re.DOTALL)
    if not match:
        return {"success": False, "error": "No <solution> tag found"}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {"success": False, "error": "Failed to parse JSON in <solution>"}

# --- 单实例评测流程 ---
import asyncio

def safe_instance_id(instance_id):
    return instance_id.replace('/', '_').replace(':', '_').replace('@', '_')

# def process_instance(instance: pd.Series, metadata: EvalMetadata, reset_logger: bool = True, auto_debug=False) -> EvalOutput:
#     import time
#     instance_id = instance.get("instance_id", instance.get("id", ""))
#     start_time = time.time()

#     # 日志归档目录
#     detail_dir = os.path.join(metadata.eval_output_dir, "detail_logs", instance_id.replace("/", "__"))
#     os.makedirs(detail_dir, exist_ok=True)

#     if reset_logger:
#         log_dir = os.path.join(metadata.eval_output_dir, "infer_logs")
#         reset_logger_for_multiprocessing(logger, instance_id, log_dir)

#     instruction = build_instruction(instance["repo_url"], instance["commit"])
#     # 不构造 instruction，直接让 agent 用系统自带的

#     config = get_config(metadata, instance_id)
#     runtime = create_runtime(config)
#     call_async_from_sync(runtime.connect)  # 别忘了连接 runtime

#     user_action = None  # 用系统自带 prompt

#     async def run_with_timeout():
#         return await asyncio.wait_for(
#             run_controller(
#                 config=config,
#                 initial_user_action=user_action,  # 这里传 None
#                 runtime=runtime,
#                 fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN.get(metadata.agent_class)
#             ),
#             timeout=INSTANCE_TIMEOUT
#         )

#     state: State | None = asyncio.run(run_with_timeout())

#     # === 统一强制设置 Docker 镜像 tag ===
#     # try:
#     #     # 从 instance_id 中解析 repo 名
#     #     # 格式类似：https://github.com/xxx/yyy@commit
#     #     parsed = urlparse(instance_id.split("@")[0])
#     #     repo_name = Path(parsed.path).name
#     #     image_tag = f"nenus-ai/{repo_name}_image:{repo_name}_v1"
#     #     build_and_set_runtime_image(instance_id, instance["repo_url"], instance["commit"], runtime)

#     #     config = get_config(metadata, instance_id, image_tag)

#     #     runtime.runtime_container_image = image_tag
#     #     logger.info(f"[PATCH] Runtime container image set to: {image_tag}")
#     # except Exception as e:
#     #     logger.warning(f"[PATCH] Failed to set image tag from instance_id: {e}")

#     call_async_from_sync(runtime.connect)

#     # 关键 patch：确保 container_name 被绑定（适用于 DockerRuntime）
#     if hasattr(runtime, "container") and runtime.container:
#         runtime._container_name = runtime.container.name
#         logger.debug(f"[PATCH] Set runtime._container_name = {runtime._container_name}")
#     else:
#         logger.warning("[PATCH] runtime.container is not available, _container_name not set")


#     logger.info(
#         f"[ACTION_CREATE] user_action in process_instance: content={repr(instruction)} | instance_id={instance_id} | at={inspect.stack()[0].function}"
#     )
#     user_action = MessageAction(content=instruction)

#     MAX_RETRY = 2
#     max_same_error_rounds = 2  # 连续N次同错提前break
#     last_errors = []
#     INSTANCE_TIMEOUT = 1200  # 每个repo最大秒数

#     for attempt in range(1, MAX_RETRY + 2):
#         try:
#             async def run_with_timeout():
#                 return await asyncio.wait_for(
#                     run_controller(
#                         config=config,
#                         initial_user_action=user_action,
#                         runtime=runtime,
#                         fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN.get(metadata.agent_class)
#                     ),
#                     timeout=INSTANCE_TIMEOUT
#                 )

#             state: State | None = asyncio.run(run_with_timeout())

#             if state is None:
#                 raise RuntimeError("No state returned")

#             histories = compatibility_for_eval_history_pairs(state.history)
#             pred = next(
#                 (entry.content for entry in reversed(state.history)
#                  if isinstance(entry, MessageAction) and getattr(entry, "role", "assistant") == "assistant"),
#                 None
#             )
#             if pred is None or not isinstance(pred, str):
#                 pred = "<solution>{\"success\": false, \"error\": \"No valid output\"}</solution>"

#             # ============= 【插入 DEADLOCK 检查】 =============
#             # 检查 agent 是否卡在需要用户输入，但输入是 None 的死锁
#             agent_state = getattr(state, "agent_state", None)
#             awaiting_input = False
#             if agent_state and hasattr(agent_state, "name"):
#                 awaiting_input = agent_state.name == "AWAITING_USER_INPUT"
#             # 保险起见再兼容一个旧写法
#             awaiting_input = awaiting_input or getattr(state, "awaiting_user_input", False)
#             if awaiting_input:
#                 # 检查最后一个 user action 有没有内容（skip recall时通常为空）
#                 last_action = getattr(state, "last_user_action", None)
#                 last_user_content = getattr(last_action, "content", None) if last_action else None
#                 if not last_user_content or (isinstance(last_user_content, str) and last_user_content.strip() == ""):
#                     logger.warning(f"[DEADLOCK] {instance_id}: agent is awaiting user input but content is empty, auto-fail.")
#                     elapsed = time.time() - start_time
#                     return EvalOutput(
#                         instance_id=instance_id,
#                         instance=instance.to_dict(),
#                         instruction=instruction,
#                         metadata=metadata,
#                         history=histories,
#                         metrics={"success": False, "error": "Agent deadlock: AWAITING_USER_INPUT with empty content."},
#                         error="Agent deadlock",
#                         test_result={"generated": None, "elapsed": elapsed, "attempt": attempt}
#                     )
#             # ============= DEADLOCK 检查结束 =============

#             from subprocess import run, PIPE
#             metrics = parse_metrics(pred)
#             print("[DEBUG] Evaluation metrics:", metrics)

#             import traceback

#             if metrics.get("success", False):
#                 try:
#                     # Step 0. Debug 当前 runtime
#                     logger.debug(f"[DEBUG] runtime={runtime}")

#                     # Step 1. 拿 container_name
#                     container_name = None
#                     if hasattr(runtime, "container") and hasattr(runtime.container, "name"):
#                         container_name = runtime.container.name
#                     elif hasattr(runtime, "_container_name"):
#                         container_name = runtime._container_name
#                     logger.debug(f"[DEBUG] Detected container_name={container_name}")

#                     if not container_name:
#                         logger.warning(f"[DOCKER COMMIT ERROR] container_name is None, cannot commit image!")
#                     else:
#                         repo_name = Path(urlparse(instance["repo_url"]).path).name
#                         image_tag = f"nenus-ai/{repo_name}_image:{repo_name}_v1"
#                         logger.info(f"[DEBUG] Ready to commit: container_name={container_name}, image_tag={image_tag}")

#                         # Step 2. commit 镜像（兜底保存整个容器快照）
#                         commit_result = run(["docker", "commit", container_name, image_tag], stdout=PIPE, stderr=PIPE, text=True)
#                         logger.info(f"[DOCKER COMMIT] commit_result={commit_result.stdout}, {commit_result.stderr}")
#                         if commit_result.returncode != 0:
#                             logger.warning(f"[DOCKER COMMIT ERROR] commit failed: {commit_result.stderr}")

#                         # Step 3. 试图多路径导出 repo（自动 find）
#                         possible_paths = [
#                             f"/workspace/repos/{repo_name}",
#                             f"/workspace/{repo_name}",
#                             f"/root/{repo_name}",
#                             f"/tmp/{repo_name}",
#                         ]
#                         # 动态 find，尽量找全路径
#                         try:
#                             find_result = run(
#                                 ["docker", "exec", container_name, "find", "/", "-type", "d", "-name", repo_name],
#                                 stdout=PIPE, stderr=PIPE, text=True
#                             )
#                             find_paths = [p for p in find_result.stdout.strip().split("\n") if repo_name in p]
#                             for p in find_paths:
#                                 if p not in possible_paths:
#                                     possible_paths.append(p)
#                             logger.info(f"[DEBUG] find all possible repo paths: {possible_paths}")
#                         except Exception as find_e:
#                             logger.warning(f"[DOCKER FIND ERROR] {find_e}")

#                         # Step 4. 自动 cp 导出 repo
#                         host_repo_path = os.path.join("/tmp/archives/repos", f"{repo_name}_{safe_instance_id(instance_id)}")
#                         os.makedirs(host_repo_path, exist_ok=True)
#                         found = False
#                         for path in possible_paths:
#                             cp_result = run(["docker", "cp", f"{container_name}:{path}", host_repo_path],
#                                             stdout=PIPE, stderr=PIPE, text=True)
#                             logger.info(f"[DOCKER CP] {container_name}:{path} => {host_repo_path}, code={cp_result.returncode}")
#                             if cp_result.returncode == 0 and os.path.exists(host_repo_path):
#                                 found = True
#                                 break

#                         if not found:
#                             logger.warning(f"[DOCKER CP] No repo copied out. All tried: {possible_paths}")

#                         # Step 5. pip freeze 导出所有包
#                         try:
#                             freeze_result = run(
#                                 ["docker", "exec", container_name, "pip", "freeze"],
#                                 stdout=PIPE, stderr=PIPE, text=True
#                             )
#                             pip_freeze_txt = os.path.join(host_repo_path, "pip_freeze.txt")
#                             with open(pip_freeze_txt, "w") as f:
#                                 f.write(freeze_result.stdout)
#                             logger.info(f"[PIP FREEZE] pip_freeze.txt saved at {pip_freeze_txt}")
#                         except Exception as e:
#                             logger.warning(f"[PIP FREEZE ERROR] {e}")

#                         # # Step 6. 生成 meta.json
#                         # if found:
#                         #     from evaluation.utils.generate_meta import auto_generate_meta_json
#                         #     try:
#                         #         meta_path = auto_generate_meta_json(host_repo_path)
#                         #         logger.info(f"[META] meta.json generated at {meta_path}")
#                         #     except Exception as meta_e:
#                         #         logger.warning(f"[META] meta.json generation failed: {meta_e}")

#                 except Exception as e:
#                     logger.warning(f"[POST-EVAL SAVE ERROR] {e}\n{traceback.format_exc()}")






#             # # --------- 连续同错检测 ---------
#             # error_msg = str(metrics.get("error", ""))[:200]
#             # last_errors.append(error_msg)
#             # if len(last_errors) > max_same_error_rounds:
#             #     last_errors = last_errors[-max_same_error_rounds:]
#             #     if all(e == last_errors[0] for e in last_errors) and error_msg:
#             #         logger.warning(f"[EARLY STOP] {instance_id}: 连续{max_same_error_rounds}次同样错误，提前终止。最后错误: {error_msg}")
#             #         break

#             # 日志归档
#             with open(os.path.join(detail_dir, f"history_attempt{attempt}.json"), "w") as f:
#                 json.dump(histories, f, indent=2)
#             with open(os.path.join(detail_dir, f"final_pred_attempt{attempt}.txt"), "w") as f:
#                 f.write(pred)
#             with open(os.path.join(detail_dir, f"metrics_attempt{attempt}.json"), "w") as f:
#                 json.dump(metrics, f, indent=2)
#             if state and hasattr(state, "last_error") and state.last_error:
#                 with open(os.path.join(detail_dir, f"last_error_attempt{attempt}.txt"), "w") as f:
#                     f.write(str(state.last_error))

#             return EvalOutput(
#                 instance_id=instance_id,
#                 instance=instance.to_dict(),
#                 instruction=instruction,
#                 metadata=metadata,
#                 history=histories,
#                 metrics=metrics,
#                 error=state.last_error if getattr(state, "last_error", None) else None,
#                 test_result={"generated": pred, "elapsed": elapsed, "attempt": attempt}
#             )
#         except asyncio.TimeoutError:
#             elapsed = time.time() - start_time
#             logger.warning(f"[TIMEOUT] {instance_id}: 超时终止（超过 {INSTANCE_TIMEOUT} 秒），判为失败。")

#             try:
#                 partial_histories = []
#                 if hasattr(runtime, "state") and hasattr(runtime.state, "history"):
#                     partial_histories = compatibility_for_eval_history_pairs(runtime.state.history)
#                 elif hasattr(runtime, "history"):
#                     partial_histories = compatibility_for_eval_history_pairs(runtime.history)

#                 if partial_histories:
#                     with open(os.path.join(detail_dir, f"history_attempt{attempt}_partial.json"), "w") as f:
#                         json.dump(partial_histories, f, indent=2)
#             except Exception as e:
#                 logger.warning(f"[TIMEOUT] Failed to save partial history: {e}")

#             # 保存 Docker logs 供 debug 使用
#             try:
#                 container_name = None
#                 if hasattr(runtime, "_container_name"):
#                     container_name = runtime._container_name
#                 elif hasattr(runtime, "container") and hasattr(runtime.container, "name"):
#                     container_name = runtime.container.name
#                 if container_name:
#                     log_path = os.path.join(detail_dir, f"docker_logs_attempt{attempt}.txt")
#                     logs = os.popen(f"docker logs {container_name}").read()
#                     with open(log_path, "w") as f:
#                         f.write(logs)
#             except Exception as log_e:
#                 logger.warning(f"[TIMEOUT] 无法抓取 docker logs: {log_e}")

#             # 写入异常信息
#             with open(os.path.join(detail_dir, f"exception_attempt{attempt}_timeout.txt"), "w") as f:
#                 f.write("TimeoutError: Instance evaluation exceeded time limit.\n")

#             return EvalOutput(
#                 instance_id=instance_id,
#                 instance=instance.to_dict(),
#                 instruction=instruction,
#                 metadata=metadata,
#                 history=[],
#                 metrics={"success": False, "error": f"Timeout: Exceeded {INSTANCE_TIMEOUT}s limit."},
#                 error=f"TimeoutError: Instance evaluation exceeded {INSTANCE_TIMEOUT} seconds.",
#                 test_result={"generated": None, "elapsed": elapsed, "attempt": attempt}
#             )

#         except Exception as e:
#             with open(os.path.join(detail_dir, f"exception_attempt{attempt}.txt"), "w") as f:
#                 f.write(traceback.format_exc())
#             if attempt == MAX_RETRY + 1:
#                 elapsed = time.time() - start_time
#                 return EvalOutput(
#                     instance_id=instance_id,
#                     instance=instance.to_dict(),
#                     instruction=instruction,
#                     metadata=metadata,
#                     history=[],
#                     metrics={"success": False, "error": str(e)},
#                     error=traceback.format_exc(),
#                     test_result={"generated": None, "elapsed": elapsed, "attempt": attempt}
#                 )

def process_instance(instance: pd.Series, metadata: EvalMetadata, reset_logger: bool = True, auto_debug=False) -> EvalOutput:
    import time
    instance_id = instance.get("instance_id", instance.get("id", ""))
    start_time = time.time()

    # # 日志目录
    # detail_dir = os.path.join(metadata.eval_output_dir, "detail_logs", instance_id.replace("/", "__"))
    # os.makedirs(detail_dir, exist_ok=True)
    # 新修改的
    raw_id = instance.get("instance_id", instance.get("id", ""))
    safe_id = win_safe_name(raw_id)
    detail_dir = os.path.join(metadata.eval_output_dir, "detail_logs", safe_id)
    os.makedirs(detail_dir, exist_ok=True)

    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, "infer_logs")
        reset_logger_for_multiprocessing(logger, instance_id, log_dir)

    # 1) 系统自带 system prompt + 极简用户指令（只传任务参数）
    instruction = build_instruction(instance["repo_url"], instance["commit"])
    user_action = MessageAction(content=instruction)   # 必须是 Action

    # 2) 准备 runtime
    config = get_config(metadata, instance_id)
    runtime = create_runtime(config)

    # 新添加的 >>> PATCH: runtime 层再强制一次
    custom_img = os.environ.get("OPENHANDS_RUNTIME_IMAGE")
    if custom_img:
        for attr in ("runtime_container_image", "container_image"):
            if hasattr(runtime, attr):
                setattr(runtime, attr, custom_img)
    # 新添加的  <<< PATCH

    call_async_from_sync(runtime.connect)

    # 3) 绑定 container_name（DockerRuntime）
    if hasattr(runtime, "container") and runtime.container:
        runtime._container_name = runtime.container.name
        logger.debug(f"[PATCH] Set runtime._container_name = {runtime._container_name}")
    else:
        logger.warning("[PATCH] runtime.container is not available, _container_name not set")

    MAX_RETRY = 2
    INSTANCE_TIMEOUT = 1200
    max_same_error_rounds = 2
    last_errors = []

    for attempt in range(1, MAX_RETRY + 2):
        try:
            async def run_with_timeout(timeout=INSTANCE_TIMEOUT):
                return await asyncio.wait_for(
                    run_controller(
                        config=config,
                        initial_user_action=user_action,   # 传入用户 Action
                        runtime=runtime,
                        fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN.get(metadata.agent_class)
                    ),
                    timeout=timeout
                )

            state: State | None = asyncio.run(run_with_timeout())
            if state is None:
                raise RuntimeError("No state returned")

            histories = compatibility_for_eval_history_pairs(state.history)
            pred = next(
                (entry.content for entry in reversed(state.history)
                 if isinstance(entry, MessageAction) and getattr(entry, "role", "assistant") == "assistant"),
                None
            )
            if pred is None or not isinstance(pred, str):
                pred = "<solution>{\"success\": false, \"error\": \"No valid output\"}</solution>"

            # DEADLOCK 检查（可留）
            agent_state = getattr(state, "agent_state", None)
            awaiting_input = False
            if agent_state and hasattr(agent_state, "name"):
                awaiting_input = (agent_state.name == "AWAITING_USER_INPUT")
            awaiting_input = awaiting_input or getattr(state, "awaiting_user_input", False)
            if awaiting_input:
                last_action = getattr(state, "last_user_action", None)
                last_user_content = getattr(last_action, "content", None) if last_action else None
                if not last_user_content or (isinstance(last_user_content, str) and last_user_content.strip() == ""):
                    logger.warning(f"[DEADLOCK] {instance_id}: agent is awaiting user input but content is empty, auto-fail.")
                    elapsed = time.time() - start_time
                    return EvalOutput(
                        instance_id=instance_id,
                        instance=instance.to_dict(),
                        instruction=instruction,
                        metadata=metadata,
                        history=histories,
                        metrics={"success": False, "error": "Agent deadlock: AWAITING_USER_INPUT with empty content."},
                        error="Agent deadlock",
                        test_result={"generated": None, "elapsed": elapsed, "attempt": attempt}
                    )

            metrics = parse_metrics(pred)
            elapsed = time.time() - start_time

            # 落盘
            with open(os.path.join(detail_dir, f"history_attempt{attempt}.json"), "w") as f:
                json.dump(histories, f, indent=2)
            with open(os.path.join(detail_dir, f"final_pred_attempt{attempt}.txt"), "w") as f:
                f.write(pred)
            with open(os.path.join(detail_dir, f"metrics_attempt{attempt}.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            if state and getattr(state, "last_error", None):
                with open(os.path.join(detail_dir, f"last_error_attempt{attempt}.txt"), "w") as f:
                    f.write(str(state.last_error))

            return EvalOutput(
                instance_id=instance_id,
                instance=instance.to_dict(),
                instruction=instruction,
                metadata=metadata,
                history=histories,
                metrics=metrics,
                error=getattr(state, "last_error", None),
                test_result={"generated": pred, "elapsed": elapsed, "attempt": attempt}
            )

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.warning(f"[TIMEOUT] {instance_id}: 超时终止（超过 {INSTANCE_TIMEOUT} 秒），判为失败。")

            # 保存 partial history（尽力而为）
            try:
                partial_histories = []
                if hasattr(runtime, "state") and hasattr(runtime.state, "history"):
                    partial_histories = compatibility_for_eval_history_pairs(runtime.state.history)
                elif hasattr(runtime, "history"):
                    partial_histories = compatibility_for_eval_history_pairs(runtime.history)
                if partial_histories:
                    with open(os.path.join(detail_dir, f"history_attempt{attempt}_partial.json"), "w") as f:
                        json.dump(partial_histories, f, indent=2)
            except Exception as e:
                logger.warning(f"[TIMEOUT] Failed to save partial history: {e}")

            # docker 日志
            try:
                container_name = getattr(runtime, "_container_name", None) or (getattr(runtime, "container", None).name if getattr(runtime, "container", None) else None)
                if container_name:
                    log_path = os.path.join(detail_dir, f"docker_logs_attempt{attempt}.txt")
                    logs = os.popen(f"docker logs {container_name}").read()
                    with open(log_path, "w") as f:
                        f.write(logs)
            except Exception as log_e:
                logger.warning(f"[TIMEOUT] 无法抓取 docker logs: {log_e}")

            return EvalOutput(
                instance_id=instance_id,
                instance=instance.to_dict(),
                instruction=instruction,
                metadata=metadata,
                history=[],
                metrics={"success": False, "error": f"Timeout: Exceeded {INSTANCE_TIMEOUT}s limit."},
                error=f"TimeoutError: Instance evaluation exceeded {INSTANCE_TIMEOUT} seconds.",
                test_result={"generated": None, "elapsed": elapsed, "attempt": attempt}
            )

        except Exception as e:
            with open(os.path.join(detail_dir, f"exception_attempt{attempt}.txt"), "w") as f:
                f.write(traceback.format_exc())
            if attempt == MAX_RETRY + 1:
                elapsed = time.time() - start_time
                return EvalOutput(
                    instance_id=instance_id,
                    instance=instance.to_dict(),
                    instruction=instruction,
                    metadata=metadata,
                    history=[],
                    metrics={"success": False, "error": str(e)},
                    error=traceback.format_exc(),
                    test_result={"generated": None, "elapsed": elapsed, "attempt": attempt}
                )


def get_clone_path_from_container(container_name: str) -> str | None:
    try:
        result = subprocess.run(
            ["docker", "exec", container_name, "find", "/workspace", "-name", ".git"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            # 输出路径到.git，例如 /workspace/repos/xgboost/.git
            git_path = result.stdout.strip().split("\n")[0]
            if git_path:
                # 去掉 /.git，得到 repo 路径
                return str(Path(git_path).parent)
    except Exception as e:
        logger.warning(f"[META] Failed to find clone path inside container: {e}")
    return None

# --- Patch run_evaluation：每个实例后强清理 docker + workspace ---
def patch_run_evaluation():
    import evaluation.utils.shared as shared
    from openhands.runtime.impl.docker.containers import stop_all_containers

    def run_evaluation(instances: list, metadata, output_file: str, num_workers: int,
                       fn: Callable[[pd.Series, any], EvalOutput]):
        results, failed = [], []
        failure_cnt = 0
        MAX_FAIL_BEFORE_BREAK = 10  # 连续失败则 break，防止 docker 死锁雪崩
        logger.info(f"🧪 Starting evaluation with {len(instances)} instances")

        for idx, inst in enumerate(tqdm(instances, desc="Running repos")):
            instance_id = None
            try:
                if isinstance(inst, str):
                    inst = json.loads(inst)
                instance_id = inst.get("instance_id") or inst.get("id") or str(inst)[:100]
                logger.info(f"[{idx+1}/{len(instances)}] Running: {instance_id}")
                try:
                    result = fn(inst, metadata, reset_logger=True)
                    results.append(result)
                    failure_cnt = 0

                    # === 【新增】自动生成 meta.json ===
                    # try:
                    #     from evaluation.utils.generate_meta import auto_generate_meta_json
                    #     from urllib.parse import urlparse
                    #     # 在保存镜像时，顺便尝试 dump container 内的路径
                    #     config = get_config(metadata, instance_id)
                    #     runtime = create_runtime(config)
                    #     container_name = getattr(runtime, "_container_name", None)
                    #     repo_path = get_clone_path_from_container(container_name)

                    #     if os.path.exists(repo_path):
                    #         meta_path = auto_generate_meta_json(repo_path)
                    #         logger.info(f"[META] meta.json generated at {meta_path}")
                    #     else:
                    #         logger.warning(f"[META] Repo path not found, skipping meta.json generation: {repo_path}")

                    # except Exception as meta_e:
                    #     logger.warning(f"[META] Failed to generate meta.json: {meta_e}")


                except Exception as ee:
                    failure_cnt += 1
                    logger.error(f"[FUNC ERROR] {instance_id}\n{traceback.format_exc()}")
                    failed.append(instance_id)
                    # 保证失败也有占位output，避免后续全丢
                    results.append({
                        "instance_id": instance_id,
                        "metrics": {"success": False, "error": str(ee)},
                        "error": traceback.format_exc(),
                        "test_result": {"generated": None, "elapsed": None, "attempt": None}
                    })
                    if failure_cnt >= MAX_FAIL_BEFORE_BREAK:
                        logger.error(f"Too many consecutive failures ({failure_cnt}), aborting batch run.")
                        break
            except Exception as e:
                logger.error(f"[FATAL ERROR] [{idx+1}] Exception before/after running instance: {e}\n{traceback.format_exc()}")
                failed.append(instance_id or f"[idx={idx}]")
                results.append({
                    "instance_id": instance_id or f"[idx={idx}]",
                    "metrics": {"success": False, "error": str(e)},
                    "error": traceback.format_exc(),
                    "test_result": {"generated": None, "elapsed": None, "attempt": None}
                })
            # finally:
            #     try:
            #         stop_all_containers("openhands-runtime")
            #         remove_all_containers("openhands-runtime")
            #     except Exception as e:
            #         logger.warning(f"Failed to stop/remove containers for {instance_id}: {e}")

            #     try:
            #         if instance_id:
            #             unique_workspace = safe_instance_workspace(instance_id)
            #             shutil.rmtree(unique_workspace, ignore_errors=True)
            #     except Exception as e:
            #         logger.warning(f"Failed to remove workspace {unique_workspace}: {e}")

        # 写入所有结果（即使部分是dict而不是EvalOutput）
        with open(output_file, "a") as f:
            for r in results:
                # 兼容EvalOutput对象与dict
                if hasattr(r, "to_dict"):
                    f.write(json.dumps(r.to_dict()) + "\n")
                else:
                    f.write(json.dumps(r) + "\n")

        logger.info(f"✅ Done. Success: {len([r for r in results if (isinstance(r, dict) and r.get('metrics', {}).get('success')) or (hasattr(r,'metrics') and getattr(r,'metrics',{}).get('success'))])}, Failed: {len(failed)}")
        if failed:
            fail_file = output_file.replace(".jsonl", ".failed.json")
            with open(fail_file, "w") as f:
                json.dump(failed, f, indent=2)
            logger.warning(f"❌ Some instances failed. List written to {fail_file}")

    shared.run_evaluation = run_evaluation


# --- Main ---
if __name__ == "__main__":
    try:
        args = parse_arguments()

        # 新添加的
        # ==== PATCH: defaults for older parse_arguments ====
        eval_note = getattr(args, "eval_note", "")
        eval_output_dir = getattr(args, "eval_output_dir", "outputs/csrbench")
        eval_n_limit = getattr(args, "eval_n_limit", None)  # None = 全量
        eval_num_workers = getattr(args, "eval_num_workers", 1)  # 单进程
        # ====================================================
        # 新添加的

        patch_run_evaluation()

        with open("evaluation/benchmarks/CSR-Bench/data/meta/CSRBench100_commit_ids.json") as f:
            meta_data = json.load(f)
        rows = [
            {"repo_url": url, "commit": cid[0], "branch": cid[1], "instance_id": f"{url}@{cid[0]}"}
            for url, cid in meta_data.items()
        ]
        df = pd.DataFrame(rows)

        # print("DEBUG available llm_config for:", args.llm_config)
        # print("DEBUG get_llm_config_arg result:", get_llm_config_arg(args.llm_config))

        llm_config = get_llm_config_arg(args.llm_config)
        llm_config.modify_params = False

        # metadata = make_metadata(
        #     llm_config, "csrbench", args.agent_cls,
        #     500, args.eval_note, args.eval_output_dir
        # )
        # output_file = os.path.join(metadata.eval_output_dir, "output.jsonl")
        # instances = prepare_dataset(df, output_file, args.eval_n_limit)
        # run_evaluation(instances, metadata, output_file, args.eval_num_workers, process_instance)

        # 新添加的
        metadata = make_metadata(
            llm_config, "csrbench", args.agent_cls,
            500, eval_note, eval_output_dir
        )
        output_file = os.path.join(metadata.eval_output_dir, "output.jsonl")
        instances = prepare_dataset(df, output_file, eval_n_limit)
        run_evaluation(instances, metadata, output_file, eval_num_workers, process_instance)
        # 新添加的


    except Exception as ee:
        logger.error(f"Main process crashed: {ee}\n{traceback.format_exc()}")

