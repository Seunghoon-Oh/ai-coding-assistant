"""Allow running the CLI as: `python -m deepagents_code`."""
# [해설][모듈 헤더] `python -m deepagents_code` 실행 경로. 패키지 `__init__`의 지연
# [해설]   `cli_main`을 받아 호출한다(`main.py`의 `cli_main`과 같은 함수).
# [해설] 자동 업데이트 후 재시작(`main.py`의 `_restart_current_process`)도 `python -m
# [해설]   deepagents_code`를 쓰므로 이 파일이 재시작 경로의 일부다.
# [해설][주의] 이 방식으로 직접 실행하면 `sys.argv[0]`이 `__main__.py`라서
# [해설]   `_invocation.invoked_name()`이 이를 거부하고 기본 이름 `dcode`로 돌아간다.
# [해설]   재시작은 `INVOKED_AS` 환경 변수로 원래 이름을 넘긴다.

from deepagents_code import cli_main

# [해설] `import`될 때는 실행되지 않고 `-m`으로 실행될 때만 `cli_main()`을 호출한다.
if __name__ == "__main__":
    cli_main()
