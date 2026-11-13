# AI Coding Assistant (AX Advanced 최종 프로젝트)

LangChain의 코딩 에이전트 CLI `dcode`(패키지 `deepagents_code` 0.2.5)에 네 가지를 더한 프로젝트입니다.

이 문서가 프로젝트 안내서입니다. 같은 폴더의 `README.md`는 업스트림 원본(영어) 그대로입니다.

| 추가한 것 | 한 줄 설명 |
|---|---|
| 계획 관문 | 에이전트가 계획서를 쓰고 AI 리뷰와 사람 승인을 받기 전에는 코드를 바꾸지 못합니다. |
| 작업 후 리뷰 | 턴이 끝나면 계획과 실제 변경을 비교한 보고서를 남깁니다. |
| 경험 기억과 자기 개선 | 끝난 작업을 기억으로 저장해 다음 요청에 주입하고, 실패한 작업에서 개선안을 만들어 검증을 통과한 것만 반영합니다. |
| 모니터링 | 전체 흐름과 지표를 Langfuse(session = thread)와 로컬 파일에 기록합니다. |

새 코드는 모두 `deepagents_code/harness/`에 있습니다. 기존 파일은 연결 지점만 고쳤습니다.

## 1. 설치

Python 3.14와 [uv](https://docs.astral.sh/uv/)가 필요합니다. 이 폴더에서 실행합니다.
Python 3.14가 설치되어 있지 않아도 됩니다. `uv sync`가 `pyproject.toml`의 `requires-python = ">=3.14,<3.15"`를 보고 3.14를 자동으로 내려받습니다. 미리 받아 두려면 `uv python install 3.14`를 실행합니다.

```bash
uv sync --locked              # 실행에 필요한 패키지
uv run dcode --version        # deepagents-code 0.2.5
```

테스트와 검사 도구까지 설치하려면 `uv sync --locked --all-groups`를 씁니다. 5절의 검사까지 할 생각이면 처음부터 이것으로 설치하세요. `--all-groups`로 설치한 뒤 `uv sync --locked`를 다시 실행하면 개발 그룹 패키지(`pytest`, `ruff`, `ty` 등 42개)가 지워지고, 그 뒤 5절의 `uv run --no-sync` 명령이 실패합니다.

## 2. 모델과 키 설정

dcode는 프로필 폴더(`DEEPAGENTS_HOME`, 기본 `~/.deepagents`)의 `config.toml`과 `.env`를 읽습니다. 아래는 개발과 확인에 쓴 설정입니다(OpenGateway의 Anthropic 호환 엔드포인트).

`~/.deepagents/config.toml`

```toml
[models]
default = "opengateway:deepseek/deepseek-v4.1-flash-ultrafast"

[models.providers.opengateway]
display_name = "OpenGateway"
class_path = "langchain_anthropic:ChatAnthropic"
api_key_env = "OPENGATEWAY_API_KEY"
base_url = "https://apis.opengateway.ai"
models = ["deepseek/deepseek-v4.1-flash-ultrafast"]

[models.providers.opengateway.params]
max_tokens = 32000
```

`max_tokens`는 꼭 넣으세요. dcode의 기본값은 4,096인데, 이 모델은 답하기 전에 생각(reasoning)에 출력 토큰을 씁니다. 4,096에서는 계획 리뷰어가 생각만 하다가 판정을 내기 전에 응답이 잘리는 일이 잦았습니다(실측: 리뷰어 실행 10번 중 6번). 그러면 리뷰가 "판정을 내지 못했습니다"로 끝나 회차를 소모합니다.

`~/.deepagents/.env` (값은 직접 채웁니다. 이 파일은 저장소에 넣지 않습니다.)

```
OPENGATEWAY_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=
```

- `dcode auth list`에 나오는 다른 공급자(Anthropic, OpenAI 등)도 쓸 수 있습니다.
- Langfuse 키 두 개가 모두 있을 때만 Langfuse로 보냅니다. 없으면 로컬 기록만 남습니다.
- `dcode config path`로 두 파일이 인식됐는지 볼 수 있습니다(모델 호출 없음).
- 프로필을 기본 위치가 아닌 곳에 두었다면(예: 수업용 `Day09/scripts/dcode`가 쓰는 `~/.deepagents-day09`) 이 문서의 모든 `dcode`·`trace_log` 명령 앞에 `DEEPAGENTS_HOME=<그 경로>`를 붙입니다. 붙이지 않으면 기본 프로필 `~/.deepagents`를 읽습니다.

## 3. 실행

에이전트에게 맡길 프로젝트 폴더(git 저장소)에서 실행합니다. 현재 폴더가 작업 폴더가 됩니다.

```bash
cd ~/my-project
uv run --project /이/폴더의/경로 dcode                       # 대화형
uv run --project /이/폴더의/경로 dcode -n "할 일" < /dev/null   # 비대화 한 번 실행
```

코드 변경을 요청하면 다음 순서로 진행됩니다.

1. 에이전트가 `.deepagents/plans/<thread ID>.md`에 계획서를 씁니다.
2. `request_plan_review`가 형식을 검증하고 AI 리뷰어가 검토합니다. 의견이 나오면 계획서를 고쳐 다시 요청합니다(최대 3회).
3. `request_plan_approval`이 사용자에게 승인을 묻습니다. 승인 전에는 작업 공간을 바꿀 수 있는 도구 `write_file`, `edit_file`, `delete`, `execute`, `js_eval`, `task`, `start_async_task`, `update_async_task`가 실행되지 않습니다(`plan_gate.py`의 `GATED_TOOLS`). `js_eval`은 도구 목록을 살펴보는 것처럼 읽기만 하는 용도라도 막히고, 이 거부는 단계 기록에 `gate.refused`로 남습니다.
4. 승인 뒤에는 계획서의 "수정 대상"에 적은 경로만 고칠 수 있습니다. 계획서를 바꾸면 승인이 무효가 됩니다.
   승인은 그 요청 하나에만 유효합니다. 같은 대화에서 다음 요청을 보내면 다시 계획 단계가 되고, 승인됐던 계획서는 보관본으로 남습니다.
5. 턴이 끝나면 작업 후 보고서, 경험 기록, (실패 신호가 있으면) 개선 절차가 이어집니다.

질문이나 읽기만 하는 요청에는 계획서가 필요 없습니다.

비대화 모드에는 승인할 사람이 없어서 기본으로는 계획과 리뷰까지만 합니다. 끝까지 실행하려면 AI 리뷰 통과를 승인으로 인정하는 옵션을 켭니다. 이 옵션은 리뷰를 **통과한** 계획만 승인합니다. 리뷰가 회차 상한(3회)까지 통과하지 못하면 승인하지 않고, 에이전트는 코드를 바꾸지 않은 채 계획서와 남은 의견을 보고하고 끝냅니다(대화형에서는 남은 의견을 사람에게 보여 주고 사람이 정합니다).

```bash
DEEPAGENTS_CODE_PLAN_AUTO_APPROVE=1 uv run --project /이/폴더의/경로 dcode -n "할 일" < /dev/null
```

비대화 모드는 기본으로 셸 도구(`execute`)가 없어 에이전트가 테스트를 돌리지 못합니다. 셸을 쓰게 하려면 dcode의 `-S`(`--shell-allow-list`)로 허용할 명령을 쉼표로 적습니다(`deepagents_code/main.py`의 인자 정의). 목록에 없는 명령은 거부되고, 셸 실행도 계획 관문의 대상이라 승인 전에는 실행되지 않습니다.

```bash
DEEPAGENTS_CODE_PLAN_AUTO_APPROVE=1 uv run --project /이/폴더의/경로 dcode -n "할 일" -S "python3,python,pytest,git,ls,cat" < /dev/null
```

`-S`에는 `recommended`(안전한 기본 목록)나 `all`(모든 명령)도 쓸 수 있습니다. `all`은 저장소 밖 실습 폴더에서만 쓰세요.

에이전트의 셸에 대해 알아 둘 점:

- `uv run --project …`은 이 폴더의 가상환경(`.venv/bin`)을 `PATH` 맨 앞에 둡니다. 그래서 에이전트가 부르는 `python`, `pytest`는 dcode 자신의 환경을 가리키고, 셸에서 `PATH` 앞에 둔 다른 가상환경은 쓰이지 않습니다. `pytest`는 `uv sync --locked --all-groups`로 설치하면 이 환경에 들어 있습니다.
- `-S`에 `python`이나 `python3`를 허용하면 `python -m pip install …`도 허용됩니다. 작업에 필요한 패키지(예: FastAPI)가 없으면 에이전트가 dcode 자신의 환경에 설치할 수 있습니다(6절 실측 기록). 그랬다면 이 폴더에서 `uv sync --locked --all-groups`로 잠금 파일 상태로 되돌립니다. 대상 프로젝트에 필요한 패키지는 미리 dcode 환경에 넣어 두기보다, 대상 프로젝트의 의존성 파일로 관리하는 편이 맞습니다.
- `uv run` 대신 이 폴더의 `.venv/bin/dcode`를 직접 실행하면 셸의 `PATH`가 그대로 쓰여 다른 가상환경의 `python`을 쓰게 할 수 있습니다. 이 방법은 실제 모델로 실행해 확인하지 않았습니다.

### 환경 변수

| 변수 | 기본 | 뜻 |
|---|---|---|
| `DEEPAGENTS_CODE_PLAN_GATE` | 켜짐 | 계획 관문. 끄면 아래 기능도 모두 꺼집니다. |
| `DEEPAGENTS_CODE_PLAN_AUTO_APPROVE` | 꺼짐 | 비대화 실행에서만 AI 리뷰 통과를 승인으로 인정합니다. |
| `DEEPAGENTS_CODE_EXPERIENCE_MEMORY` | 켜짐 | 경험 기록의 저장과 주입. |
| `DEEPAGENTS_CODE_PLAN_SELF_IMPROVE` | 켜짐 | 실패 신호가 있는 턴 뒤의 개선 절차(모델 호출이 늘어납니다). |
| `DEEPAGENTS_CODE_PLAN_STEP_LOG` | 켜짐 | 로컬 단계 기록 파일. |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | 없음 | Langfuse 전송. |

### 프로젝트 폴더에 생기는 파일

모두 프로젝트의 `.deepagents/` 아래에 있습니다. 에이전트는 계획서 파일 말고는 이 폴더에 쓸 수 없고, 나머지는 하네스 코드가 씁니다.

```
.deepagents/
  plans/<thread ID>.md              계획서 (에이전트가 작성)
  plans/<thread ID>.review.md       리뷰 회차별 판정과 승인 결정
  plans/<thread ID>.report.md       작업 후 보고서 (계획 대 실제)
  plans/<thread ID>.baseline.json   승인 시점의 git 상태
  plans/<thread ID>.<시각>.approved.md  이전 요청에서 승인됐던 계획서의 보관본
  memory/<시각>-<thread ID>.md       작업 경험 기록
  improvements/history.md           개선안의 반영·보류 이력과 전후 점수
  improvements/applied|pending/     반영된 개선안, 보류된 개선안
  improvements/cases/               평가 사례
  traces/<thread ID>.jsonl          단계 기록과 턴 지표
  AGENTS.md                         반영된 프로젝트 규칙
  prompt_addendum.md                반영된 프롬프트 덧붙임
  skills/<이름>/SKILL.md             반영된 Skill
```

### 기록 조회

프로젝트 폴더에서 실행합니다. thread ID를 빼면 기록된 thread 목록이 나옵니다.

```bash
uv run --project /이/폴더의/경로 python -m deepagents_code.harness.trace_log            # thread 목록
uv run --project /이/폴더의/경로 python -m deepagents_code.harness.trace_log <thread ID>  # 단계 목록과 첫 실패 지점
```

실패로 기록된 단계는 두 등급으로 나눠 보여 줍니다(`trace_log.py`의 `failure_level`).

| 표시 | 등급 | 해당 단계 |
|---|---|---|
| `!` | 절차상 거부 (`warning`) | 관문 거부(`gate.refused`), 계획서 형식 오류(`plan.invalid`), 리뷰의 "수정 필요"(`plan.review`), 순서가 틀린 승인 요청과 사람의 "수정 요청"(`plan.approval`), 검증에서 보류된 개선(`improvement.decision`) |
| `✗` | 실패 (`error`) | 작업 후 리뷰가 문제를 찾음(`work.review`), 리뷰어가 판정을 내지 못함, 리뷰가 회차 상한까지 수렴하지 않음, 비대화 실행에 승인 옵션이 없거나 리뷰를 통과하지 못해 승인할 수 없음(`plan.approval`), 그 밖의 알 수 없는 실패 |

"첫 실패 지점"은 첫 `✗` 단계입니다. 절차상 거부는 그 위에 건수만 따로 적습니다. 새 기록에는 등급이 `level` 필드로 저장되고 Langfuse의 `harness.*` 실행 입력에도 들어갑니다. `level`이 없는 예전 기록은 단계 종류와 내용으로 같은 판정을 내리므로, 이전에 남은 기록도 같은 방식으로 읽힙니다.

Langfuse에서는 thread ID가 session ID입니다. session 하나에 사용자 턴마다 trace가 하나씩 생기고, 하네스 단계는 `harness.<종류>` 이름의 실행으로, 턴 지표는 `harness.<이름>` 점수로 보입니다. Langfuse에는 리뷰어·작업 후 리뷰어·개선 에이전트·서브에이전트의 호출도 모두 남습니다. 턴 지표도 이 호출을 모두 셉니다(아래 표). 모델 호출은 그래프에 묶인 계수 콜백(`trace_log.RunCounterCallback`)이 보조 에이전트와 서브에이전트의 실행까지 따라가 셉니다. 도구 호출은 각 에이전트의 미들웨어 맨 앞(가장 바깥)에 둔 `trace_log.ToolResultCounterMiddleware`가 에이전트에게 돌아가는 결과로 셉니다. 그래서 도구를 실행하지 않고 미들웨어가 돌려준 오류(셸 허용 목록에 없는 명령의 거부 등)도 잡힙니다. 관문 거부는 호출로는 세고 오류로는 세지 않습니다(따로 `gate_refusals`). 역할은 보조 에이전트에 붙은 `harness:<역할>` 태그로 나눕니다.

| 기록 | 지표 | 뜻 |
|---|---|---|
| `turn.metrics` | `all_model_calls`, `all_model_errors`, `all_tool_calls`, `all_tool_errors` | 에이전트가 일하는 동안 모든 역할의 모델 호출(시도), 실패한 시도, 끝난 도구 호출, 도구 오류 |
| `turn.metrics` | `model_calls.main`, `model_calls.plan_review`, `model_calls.subagent` | 역할별 모델 호출 |
| `turn.post_work` | `post_work_model_calls`, `post_work_model_errors`, `post_work_tool_calls`, `post_work_tool_errors` | 작업 후 리뷰와 자기 개선의 같은 값 |
| `turn.post_work` | `model_calls.work_review`, `model_calls.improvement` | 역할별 모델 호출 |

`all_model_calls`와 `post_work_model_calls`의 합이 턴 전체 모델 요청 수이고, `Usage Stats`의 요청 수와 Langfuse GENERATION 수와 같아야 합니다. 예전부터 있던 `model_calls`, `tool_calls`, `tool_errors`, `model_retries`는 그대로 메인 에이전트 대화만 센 값입니다. 바뀌기 전 기록과 비교할 수 있게 남겨 두었습니다. 작업 후 단계가 돌지 않은 턴(바뀐 파일이 없는 턴)에는 `turn.post_work`가 없습니다.

### 반영된 개선 되돌리기

자기 개선은 사람 승인 없이 반영되므로, 반영된 것을 보고 되돌리는 명령을 둡니다. 프로젝트 폴더에서 실행합니다.

```bash
uv run --project /이/폴더의/경로 python -m deepagents_code.harness.improvement list         # 반영된 개선 목록
uv run --project /이/폴더의/경로 python -m deepagents_code.harness.improvement revert <id>  # 되돌리기
```

대상 파일에 반영된 형태 그대로 남아 있을 때만 되돌립니다. 사람이 그 뒤에 고친 내용은 건드리지 않습니다. 되돌린 사실도 `improvements/history.md`에 남습니다.

## 4. 추가한 코드

| 파일 (`deepagents_code/harness/`) | 역할 |
|---|---|
| `plan.py` | 계획서 위치, 필수 절 검증, "수정 대상" 범위 판정 |
| `plan_review.py` | 읽기 전용 AI 리뷰어, 보조 에이전트 조립 |
| `plan_gate.py` | 관문 미들웨어: 도구 차단, 리뷰·승인 도구, 범위 대조, 턴 종료 훅 |
| `plan_report.py` | 턴의 변경 수집, git 기준선, 계획 대 실제 비교, 작업 후 리뷰어, 보고서 |
| `experience.py` | 경험 기록 파일, 글자쌍 유사도 검색, 관련 기록 주입 미들웨어 |
| `improvement.py` | 실패 신호, 개선안, 정적 검사, 전후 비교 평가, 판정, 반영, 이력 |
| `trace_log.py` | 단계 기록(로컬 파일과 trace), 턴 지표, 모델 재시도 계수, 조회 명령 |
| `observability.py` | Langfuse 콜백 연결, thread ID를 session ID로 |

기존 파일의 수정: `agent.py`(미들웨어 조립), `server_graph.py`(Langfuse 연결), `_env_vars.py`(환경 변수 등록), `auto_mode.py`(관문이 거부할 호출은 승인 질문에서 제외, `review_exempt`), `pyproject.toml`(이 폴더만으로 `uv sync`, `langfuse` 의존성). 원본에 있던 `# [해설]` 주석은 줄 길이 규칙에 맞게 줄바꿈만 했습니다.

4일차에 실습한 Skill은 `deepagents_code/built_in_skills/fastapi-endpoint-with-tests/SKILL.md`에 있습니다. 내장 Skill로 패키지에 포함되고 실행 중에 읽힙니다.

## 5. 검사와 테스트

```bash
uv sync --locked --all-groups
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync ty check
uv run --no-sync pytest -n 8 --disable-socket --allow-unix-socket tests/unit_tests/harness   # 추가한 기능
uv run --no-sync pytest -n 8 --disable-socket --allow-unix-socket tests/unit_tests           # 전체
```

- 테스트는 대본을 재생하는 가짜 모델로 돌아서 API 비용이 들지 않고 네트워크를 쓰지 않습니다.
- 작업자 수를 `-n 8`로 고정한 이유: 전체 테스트에는 실행 시간에 민감한 원본 테스트가 섞여 있습니다. 코어 수만큼 작업자를 띄우는 `-n auto`에서는 기계가 바쁠 때 그런 테스트가 한두 개씩 무작위로 실패했습니다(22코어 기계. 자주 보인 것: `tests/unit_tests/tui/widgets/test_chat_input.py`의 `test_vscode_space_workaround_keeps_run_in_sync`). `-n 8`에서는 대부분 전부 통과했고, 위 테스트 하나만 드물게 실패했습니다. 실패한 테스트는 단독으로 돌리면 통과합니다.

## 6. 평가 항목별 확인 방법

테스트 파일은 `tests/unit_tests/harness/` 아래에 있습니다.

### 1. 개발 기준

| 항목 | 확인 |
|---|---|
| 1-1 코드 스타일 | `ruff format --check .`, `ruff check .` (설정은 `pyproject.toml`의 `[tool.ruff]`, 규칙 전체 선택) |
| 1-2 명명 규칙, import 정리 | 같은 `ruff check .` (N, I 규칙 포함) |
| 1-3 docstring, 주석 | 같은 `ruff check .` (D 규칙, google 방식). 새 모듈의 함수마다 역할·인자·반환값을 적었습니다. |
| `uv` | `uv sync --locked --all-groups`, 잠금 파일 `uv.lock` (`uv lock --check`로 `pyproject.toml`과 맞는지 확인) |

### 2. 작업 계획 구체화와 리뷰

| 항목 | 구현 | 테스트 |
|---|---|---|
| 2-1 요구사항, 작업 범위, 완료 조건 | 계획서 필수 절 (`plan.py`의 `REQUIRED_SECTIONS`, `validate_plan`) | `test_plan.py` |
| 2-2 수정 대상, 작업 순서, 테스트 방법 | 같은 필수 절. 수정 대상은 경로 목록, 작업 순서는 번호 목록이어야 합니다. | `test_plan.py` |
| 2-3 AI 리뷰와 의견 반영 | `request_plan_review` (`plan_gate.py`의 `_review`, `plan_review.py`). 리뷰 기록은 `.review.md` | `test_plan_gate.py`의 리뷰 테스트 |
| 2-4 승인 뒤에만 변경, 범위 변경 시 재검토 | 승인 전 도구 차단, 승인 뒤 "수정 대상" 대조, 계획서가 바뀌면 승인 무효 (`plan_gate.py`의 `_decide`). 새 요청이 오면 계획 단계로 되돌림 (`_start_turn`) | `test_plan_gate.py`, 실제 조립 에이전트는 `test_plan_wiring.py` |
| 계획서 파일 필수 (에이전트 작성) | `.deepagents/plans/<thread ID>.md` | `test_plan_gate.py` |
| 계획 대 실제 차이 분석 | 작업 후 보고서 (`plan_report.py`) | `test_plan_report.py` |

### 3. Memory와 Self-Improving

| 항목 | 구현 | 테스트 |
|---|---|---|
| 3-1 저장, 세션 뒤에도 유지 | 경험 기록 파일 (`experience.py`의 `save`), 프로젝트 규칙은 `.deepagents/AGENTS.md` | `test_experience.py` |
| 3-2 검색해서 계획·코드 작성에 활용 | 턴 시작 때 검색해 상위 3건 주입. 주입된 턴의 계획서는 "참고한 기억" 절에 주입된 기록의 id를 적거나 해당 없음을 밝혀야 리뷰를 통과합니다. 주입은 턴의 모든 모델 호출에 붙어서, 승인 뒤 코드를 쓰는 호출도 같은 기록을 봅니다 (`experience.py`의 `wrap_model_call`). | `test_experience.py`. 코드 작성 중에도 기억이 보이는지는 `test_shown_memory_stays_in_view_while_the_approved_work_is_done` |
| 3-3 실패에서 개선안 생성 | 실패 신호가 있으면 개선 에이전트가 규칙·프롬프트 덧붙임·Skill 중 하나를 제안 (`improvement.py`). 작업 후 리뷰가 성공으로 본 턴에서 "계획에 없던 추가 작업"(`unplanned`)만 있으면 개선하지 않습니다. 그런 개선은 에이전트가 쓸모 있는 일을 덜 하게 만들기 때문입니다(`Signals.failed`) | `test_improvement.py`의 `test_extras_alone_do_not_fail_a_successful_turn` |
| 3-4 전후 검증 뒤 반영 | 정적 검사 → 사례별 전후 계획 비교와 심사 → 근거 사례가 오르고 떨어진 사례가 없을 때만 반영. 평가 사례는 개선안을 보지 못하는 별도 역할이 실패 신호만 보고 씁니다. 이력은 `improvements/history.md`, 되돌리기는 `improvement revert`. 반영된 개선은 관문을 열지 못하고, 반영 뒤에도 기억 주입·리뷰·승인·보고서가 그대로 동작합니다. | `test_improvement.py`. 반영 뒤 기존 기능의 회귀는 `test_an_applied_prompt_note_cannot_open_the_gate`, `test_memory_review_and_report_still_work_after_improvements` |

### 4. 모니터링

| 항목 | 구현 | 테스트 |
|---|---|---|
| 4-1 전체 실행 흐름 | 단계 기록 12종 (`trace_log.py`의 `record`, 호출은 `plan_gate.py`와 `experience.py`) | `test_trace_log.py` |
| 4-2 LLM 호출, 도구, Memory, 자기 개선 | LLM 호출과 도구 실행은 콜백이 자동 기록. 모든 역할의 실패한 모델 시도(`RunCounterCallback`)와 도구 오류(`ToolResultCounterMiddleware`, 미들웨어가 도구 없이 돌려준 오류 포함)를 역할별로 셈. `memory.search`, `memory.saved`, `improvement.decision`은 직접 기록 | `test_trace_log.py`, `test_observability.py`, `test_plan_wiring.py`의 `test_reviewer_calls_are_counted_under_the_plan_review_role` |
| 4-3 지표 | 턴마다 실행 시간, 모델 호출, 도구 호출, 모델 재시도, 도구 오류, 관문 거부, 리뷰 회차, 토큰 (`turn_metrics`, `ModelRetryCounterMiddleware`). 리뷰어·서브에이전트·작업 후 리뷰·개선까지 모든 역할의 호출 수와 실패 수를 역할별로 (`RunCounter`, `all_*`, `model_calls.<역할>`). 작업 후 리뷰와 개선에 든 시간과 호출은 따로 (`turn.post_work`). Langfuse에는 점수로 | `test_trace_log.py`, `test_plan_wiring.py`의 `test_assembled_agent_records_the_calls_it_made` |
| 4-4 조회, 실패 지점 | `python -m deepagents_code.harness.trace_log <thread ID>`가 단계 목록과 첫 실패 지점을 출력. 절차상 거부(`!`)와 실패(`✗`)를 나누고, 첫 실패 지점은 실패만 가리킵니다 (`failure_level`, `first_failure`) | `test_trace_log.py`의 `test_failure_level_follows_the_kind_and_its_details`, `test_first_failure_is_the_earliest_error_not_a_refusal` |
| Langfuse, session별 저장 | thread ID를 session ID로 (`observability.py`) | `test_observability.py` |

### 직접 돌려 보는 순서

실제 모델을 부르므로 키가 필요하고 과금됩니다. 요청 하나에 모델 요청 약 30~60회, 아래 두 요청을 합쳐 80~100회 정도이고 2~4분 걸립니다(2026-11-13에 같은 순서로 세 번 실측: 합계 97·82·85회, 실행 끝 `Usage Stats` 기준). 같은 요청이라도 계획 리뷰를 몇 회차 받는지에 따라 29회에서 61회까지 흔들렸습니다.

```bash
mkdir -p ~/dcode-practice && cd ~/dcode-practice && git init -q
printf 'def add(a, b):\n    return a + b\n' > calc.py
git add . && git -c user.name=practice -c user.email=practice@example.com commit -qm init

export DCODE="uv run --project /이/폴더의/경로"
DEEPAGENTS_CODE_PLAN_AUTO_APPROVE=1 $DCODE dcode -n "calc.py에 subtract 함수를 추가하고 tests/test_calc.py에 테스트를 추가해 줘" < /dev/null
DEEPAGENTS_CODE_PLAN_AUTO_APPROVE=1 $DCODE dcode -n "calc.py에 multiply 함수를 추가하고 tests/test_calc.py에 테스트를 추가해 줘" < /dev/null

ls .deepagents/plans .deepagents/memory .deepagents/improvements
$DCODE python -m deepagents_code.harness.trace_log
```

- 첫 실행 뒤: 계획서, 리뷰 기록, 보고서, 경험 기록 1건이 생깁니다.
- 둘째 실행의 계획서에는 "참고한 기억" 절이 있고 첫 실행의 기록 id가 적힙니다.
- 위 순서는 `-S`가 없어 셸 도구가 없으므로 테스트 실행 단계가 빠집니다. 보고서가 이를 "빠진 단계"로 적고, 개선 절차가 돌아 `improvements/history.md`에 결정(반영 또는 보류)이 남습니다. 그래서 `trace_log`에서 `work.review`가 `✗`로 보이는 것은 이 순서에서 정상입니다.
- Langfuse 키가 있으면 두 요청이 session 두 개로 따로 보입니다(session ID는 `trace_log` 목록의 thread ID).
- 사람 승인 화면은 대화형(`$DCODE dcode`)에서 볼 수 있습니다.

**테스트 실행까지 보려면 (`-S`)**

새 실습 폴더에서 같은 두 요청에 `-S`를 붙입니다. 에이전트가 쓰는 `pytest`는 dcode 환경에 개발 그룹으로 들어 있습니다(1절 `uv sync --locked --all-groups`, 3절 "에이전트의 셸에 대해 알아 둘 점").

```bash
ALLOW="python3,python,pytest,git,ls,cat"
DEEPAGENTS_CODE_PLAN_AUTO_APPROVE=1 $DCODE dcode -n "calc.py에 subtract 함수를 추가하고 tests/test_calc.py에 테스트를 추가해 줘" -S "$ALLOW" < /dev/null
DEEPAGENTS_CODE_PLAN_AUTO_APPROVE=1 $DCODE dcode -n "calc.py에 multiply 함수를 추가하고 tests/test_calc.py에 테스트를 추가해 줘" -S "$ALLOW" < /dev/null
```

- 보고서의 "실행한 명령"에 에이전트가 돌린 pytest 명령이, "테스트 결과"에 통과 수가 적힙니다. 단계 기록의 `work.review`에는 `commands=<실행한 명령 수>`가, Langfuse에는 `execute` 도구 실행이 남습니다.
- 2026-11-13 실측(`git` 없이 `-S "python3,python,pytest,ls,cat"`): 모델 요청 31회와 26회, 첫 요청은 `python -m pytest -q …/tests/test_calc.py`로 4 passed, 둘째는 `python -m pytest -q`로 7 passed. `work.review`에는 `commands=1`, `commands=2`가 남았습니다.
- 허용 목록에 없는 명령은 실행되지 않고 보고서의 "빠진 단계"로 남습니다. 위 실측에서는 계획에 넣은 `git diff`가 목록에 없어 `work.review`가 `✗`였습니다. 그래서 위 `ALLOW`에는 `git`을 넣었습니다.

### 실측 기록 (2026-11-13)

위의 `-S` 순서(`ALLOW="python3,python,pytest,git,ls,cat"`, `DEEPAGENTS_CODE_PLAN_AUTO_APPROVE=1`)를 새 실습 폴더에서 두 번 돌린 결과입니다. 모델은 2절의 OpenGateway `deepseek/deepseek-v4.1-flash-ultrafast`이고, Langfuse 값은 Langfuse 공개 API(`GET /api/public/v2/observations`)로 session별로 모아 셌습니다. thread ID는 각 실행의 값이라 다시 돌리면 달라집니다.

**1회차** (커밋 `81c0bb06`, 실패 등급과 역할별 계수를 넣기 전)

| 요청 | thread = Langfuse session | trace | `Usage Stats` 요청 | Langfuse GENERATION | 흐름 |
|---|---|---|---|---|---|
| subtract | `01a1d32b-4705-7720-8ee5-fe367ace3ba2` | 1 | 41 | 41 | 승인 전 `execute` 3회 거부(`gate.refused`) → 리뷰 1회차 `revise`, 고친 뒤 2회차 `approve` → pytest 5 passed → 작업 후 리뷰가 완료 조건의 `python3 -c …` 확인 명령을 실행하지 않은 것을 찾음(`work.review`) → 개선안 생성, 전후 비교 뒤 **보류** |
| multiply | `01a1d32d-21a5-72d3-a32e-d47a47dd5f5f` | 1 | 22 | 22 | `memory.search`가 subtract의 기록을 찾음 → 계획서 "참고한 기억" 절에 그 기록 id와 교훈(테스트 실행 위치, 코드 스타일)을 적음 → 리뷰 1회에 `approve` → pytest 9 passed → 첫 실패 지점 없음 |

- 두 요청이 Langfuse에서 session 두 개로 나뉘고, 각 session에 사용자 턴의 trace가 하나씩 있었습니다. 하네스 단계는 `harness.*` 노드로 보였습니다. subtract에서는 10종(`memory.search`, `gate.refused`, `plan.written`, `plan.review`, `plan.approval`, `turn.metrics`, `work.review`, `memory.saved`, `improvement.decision`, `turn.post_work`)이 보였습니다.
- 메인 에이전트 밖에서 난 도구 오류(`ls` 2건)는 Langfuse에 ERROR 수준으로 남았습니다.
- 이 기록을 지금 코드의 `trace_log`로 다시 읽으면 subtract의 첫 실패 지점은 11번 `work.review`이고, 절차상 거부 5건(`gate.refused` 3, `plan.review` 1, `improvement.decision` 1)이 따로 표시됩니다. `level` 필드가 없는 기록이라 단계 종류와 내용으로 판정한 결과입니다.

**2회차** (커밋 `dc95cc5d`, 역할별 계수를 넣은 뒤)

| 요청 | thread = Langfuse session | `turn.metrics` | `turn.post_work` | 합계 | `Usage Stats` 요청 | Langfuse GENERATION |
|---|---|---|---|---|---|---|
| subtract | `01a1d34a-24cb-7bd2-8d9b-5baa1fd1dddf` | main 22 + plan_review 14 = 36 | work_review 2 | **38** | 38 | 38 |
| multiply | `01a1d34b-1409-7d82-bdbc-73989b9d2a12` | main 11 + plan_review 7 = 18 | work_review 3 | **21** | 21 | 21 |

- 턴 지표의 모델 호출 합계(`all_model_calls + post_work_model_calls`)가 `Usage Stats`, Langfuse GENERATION 수와 같았습니다. 예전 지표 `model_calls`(메인 에이전트만)는 22와 11이었습니다.
- 도구 호출도 Langfuse의 TOOL 수와 같았습니다(subtract: 42 + 2 = 44).
- subtract에서는 리뷰가 세 번 모두 `revise`라 회차 상한에 닿았고, 그 단계가 `✗`(`plan.review`, `verdict=limit`)로 기록됐습니다. 당시 코드는 그래도 비대화 AI 승인을 내주어 작업이 진행됐습니다. 지금 코드에서는 이 경우 승인하지 않고 끝납니다(아래 "4일차 Skill 확인").
- 이 실행에서 `all_tool_errors`가 예전 지표 `tool_errors`보다 작게 기록됐습니다(subtract 0 대 1, multiply 0 대 1). 원인은 두 가지였습니다. 하나는 관문 도구가 `Command`에 담아 돌려준 오류 응답이고(커밋 `cb6754ec`에서 고침), 다른 하나는 셸 허용 목록이 도구를 실행하지 않고 돌려준 거부입니다. 당시 도구는 콜백으로 셌기 때문에, 실행되지 않은 도구의 결과는 보이지 않았습니다. 지금은 도구 호출을 가장 바깥 미들웨어에서 결과로 셉니다. 같은 경우를 조립된 에이전트 테스트(`test_a_command_the_allow_list_rejects_is_counted_as_a_tool_error`)와 단위 테스트(`test_a_gate_tool_error_reply_is_counted_like_the_message_count`)로 확인했습니다.

**4일차 Skill 확인** (커밋 `96acb506`)

FastAPI 실습 프로젝트(`app/main.py`에 `/health`와 그 테스트만 있는 git 저장소)에서 Skill을 언급하지 않고 요청했습니다: "할 일 API를 만들어 줘. POST /todos 로 할 일을 추가하고, GET /todos/{todo_id} 로 하나를 조회해. 없는 id는 404를 돌려줘." (`ALLOW="python3,python,pytest,git,ls,cat"`, `DEEPAGENTS_CODE_PLAN_AUTO_APPROVE=1`)

| 확인 | 결과 |
|---|---|
| Skill 선택 | 시작 2초 뒤에 `built_in_skills/fastapi-endpoint-with-tests/SKILL.md`를 읽음 (Langfuse 도구 기록) |
| Skill의 순서 | 계획과 승인 → `app/schemas/todo.py` → `app/deps.py` → `app/routers/todos.py` → `app/main.py`에 등록 → 테스트 → 실행 결과 보고. Skill 예시와 같은 구조 |
| 성공·실패 테스트 | 성공 2개(생성 201, 조회 200), 실패 3개(없는 id 404, 빈 제목 422, 잘못된 id 422). 기존 테스트 포함 6 passed |
| 지표 | 모델 호출 83(메인 63 + 계획 리뷰 20) + 작업 후 9 = 92 = `Usage Stats` 92. `tool_errors` 15 = `all_tool_errors` 15 |

이 실행에서 하네스 문제 두 가지를 찾았습니다.

- 리뷰가 세 번 모두 `revise`로 회차 상한에 닿았는데 비대화 AI 승인이 났습니다. 승인 옵션이 남은 리뷰 의견을 확인하지 않았기 때문입니다. 지금은 남은 의견이 있으면 승인하지 않습니다(`plan.approval`, 이유 "리뷰 상한에 닿아 남은 의견이 있음", `✗`). 테스트는 `test_plan_gate.py`의 `test_ai_approval_refuses_a_plan_left_with_open_findings`입니다.
- 에이전트의 `python`이 dcode 자신의 환경을 가리켜, FastAPI가 없다고 판단한 에이전트가 `python -m pip install fastapi`로 그 환경에 설치했습니다. 제출 파일은 바뀌지 않았고 `uv sync --locked --all-groups`로 되돌렸습니다. 3절 "에이전트의 셸에 대해 알아 둘 점"에 적었습니다.

## 7. 알려진 한계

- 관문은 dcode의 내장 도구만 다룹니다. MCP 도구는 막지 않습니다. 파일을 바꾸는 MCP 서버가 붙어 있으면 승인 전에도 그 도구로 바꿀 수 있습니다.
- 승인 뒤의 `execute`와 `task`는 무엇을 바꾸는지 알 수 없어 "수정 대상"과 대조하지 않고, 작업 후 보고서에서 잡습니다.
- 서브에이전트 안에서는 관문이 개입하지 않습니다. 서브에이전트는 계획이 승인된 뒤에만 시작되지만(`task`가 관문 대상), 그 안의 쓰기는 "수정 대상"과 대조하지 않습니다.
- 승인이 요청 하나에만 유효해서, "변수 이름만 바꿔 줘" 같은 작은 후속 요청에도 계획과 승인이 다시 필요합니다. 에이전트가 작업 도중에 묻는 질문(`ask_user`)에 답하는 것은 새 요청이 아닙니다.
- 샌드박스 실행에는 관문이 없습니다. 계획서 파일을 호스트 파일 시스템에서 읽기 때문입니다.
- 기억 검색은 글자 겹침을 봅니다. 같은 내용을 다른 낱말로 쓰면 찾지 못합니다.
- 개선안의 평가는 계획 수준입니다. 계획이 나아지는지를 보고, 실제 작업 결과가 나아지는지는 보지 않습니다. 평가 사례는 개선안을 보지 못하는 역할이 쓰지만, 같은 실패 신호에서 나오므로 개선안과 완전히 독립은 아닙니다. 사례마다 계획을 한 번씩만 받아 모델 출력의 흔들림에 영향을 받습니다.
- 고칠 수 없는 환경 제약도 실패 신호가 됩니다. `-S` 없는 비대화 모드처럼 셸이 없어서 테스트를 못 돌린 턴마다 개선안이 새로 만들어지고, 계획 수준 점수가 오르면 반영됩니다. 실측(6절의 순서를 세 번)에서 이런 개선이 반영된 뒤에도 다음 요청은 같은 이유로 테스트를 돌리지 못했습니다. 반영된 것이 쓸모없으면 `improvement revert`로 되돌립니다.
- 절차상 거부와 실패의 구분은 단계 종류와 기록 내용으로 정한 규칙입니다(3절 "기록 조회"). 관문 거부가 같은 턴에 여러 번 반복되는 것처럼, 절차상 거부라도 에이전트가 헤매고 있다는 신호일 수 있으니 `!`의 건수도 함께 봅니다.
- 예전부터 있던 지표 `model_calls`, `tool_errors`, `model_retries`는 메인 에이전트의 대화만 셉니다. 모든 역할의 값은 `all_*`, `post_work_*`, `model_calls.<역할>`에서 봅니다(3절 "기록 조회"). 계수 콜백은 계획 관문이 켜져 있을 때만 붙습니다. 입력 토큰은 게이트웨이가 보내지 않아 여전히 비어 있습니다(아래 항목). `duration_seconds`는 에이전트의 작업이 끝날 때까지의 시간이고(승인 질문에서 사람이 기다린 시간 포함), 그 뒤의 시간은 `turn.post_work`에 따로 남습니다. 재시도 횟수와 역할별 호출 수는 서버 프로세스의 메모리에 두므로 턴 도중에 프로세스가 다시 시작되면 그 턴의 값이 0부터 다시 셉니다.
- 도구 호출을 세는 미들웨어는 도구 호출 경로를 지나는 결과만 봅니다. 보조 에이전트가 도구 호출 상한을 넘긴 호출은 `ToolCallLimitMiddleware`가 모델 응답 뒤의 훅(`after_model`)에서 오류 메시지를 끼워 넣어 처리하므로 `all_tool_calls`와 `all_tool_errors`에 들어가지 않습니다(`test_reviewer_reads_are_counted_up_to_its_tool_call_limit`). 메인 에이전트는 이 미들웨어를 쓰지 않습니다.
- 추론 모델을 작은 출력 상한으로 쓰면 리뷰어와 개선 에이전트가 판정 없이 끝날 수 있습니다. 리뷰 기록에 "리뷰어가 판정을 내기 전에 응답이 끝났습니다"가 보이면 모델 설정의 `max_tokens`를 올려야 합니다(2절).
- OpenGateway에서는 입력 토큰 수를 수집하지 못합니다. 출력 토큰은 있는데 입력 토큰이 0으로 오면 0으로 적지 않고 단계 기록에 `input_tokens=null`과 `unreported`로 남기며, Langfuse 점수에서도 뺍니다 (`trace_log.turn_metrics`, `metrics_step`). 원인: 게이트웨이는 스트리밍의 시작 이벤트(`message_start`)에 입력 토큰 수를 싣는데, 설치된 `langchain-anthropic`은 끝 이벤트(`message_delta`)에서 읽고 게이트웨이는 거기에 넣지 않습니다. 스트리밍 없이 부르면 값이 옵니다. 모델 클래스 내부를 덮어써야 고칠 수 있어 그대로 두었습니다.

## 8. 향후 개선

하네스에는 글을 쓰지 않고 판정만 하는 곳이 네 군데 있습니다. 이런 곳은 수업에서 다룬 Decision 모델(예: Jev의 Noul·Choice·Score 질문)로 바꿀 수 있습니다. 이번 제출에는 넣지 않았습니다. 판정용 키가 없는 환경에서도 같은 동작을 보장하기 위해서이고, 검증을 마친 판정 경로를 기한 직전에 바꾸지 않기 위해서입니다.

| 판정 | 지금 방식 | 바꾸면 보완되는 한계 | 틀렸을 때의 피해 |
|---|---|---|---|
| 기억이 이번 요청과 관련 있는가 | 글자쌍 겹침 점수 (`experience.py`) | 다른 낱말로 쓴 같은 내용을 찾지 못함 | 작음. 기억 하나가 더 들어가거나 빠짐 |
| 개선안이 사례의 계획을 낫게 했는가 | 언어 모델 심사 (`improvement.py`) | 심사에 드는 시간과 호출 수 | 중간. 개선이 잘못 반영되거나 보류됨 (되돌리기 가능) |
| 실패 신호가 개선할 만한 실패인가 | 규칙 (`improvement.py`의 `Signals`) | 일부러 유도한 거부도 실패로 셈 | 작음. 개선 절차가 한 번 더 돌거나 덜 돎 |
| 새 메시지가 새 요청인가 | 항상 새 요청으로 봄 (`plan_gate.py`의 `_start_turn`) | 작은 후속 요청에도 계획과 승인이 다시 필요 | 큼. 계획 없이 코드가 바뀔 수 있음 |

도입한다면 피해가 작은 것부터가 맞습니다. 계획 리뷰와 작업 후 리뷰는 판정과 함께 지적 내용(글)이 필요해서 대상이 아닙니다.

**개선안을 실제 실행으로 검증하기.** 지금 개선안의 검증은 계획 수준입니다(7절). 반영하기 전에 평가 사례와 이전에 성공한 요청을 개선안을 넣은 에이전트로 실제로 실행해 보면, 작업 결과가 나아지는지와 기존 기능에 회귀가 없는지를 직접 확인할 수 있습니다. 개선안 하나마다 실제 에이전트 실행이 사례 수만큼 늘어 턴당 모델 호출이 크게 늘어나므로, 이번 제출에는 넣지 않았습니다. 넣는다면 git worktree 같은 격리된 작업 공간에서 돌리고, 실행 비용 상한을 두는 것이 전제입니다.

## 원본과 라이선스

원본은 [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)의 `libs/code`(deepagents-code 0.2.5)이고 MIT 라이선스입니다(`pyproject.toml`). 원본의 README는 `README.md`에, 개발 문서는 이 폴더의 `ARCHITECTURE.md`, `DEVELOPMENT.md`, `COMMANDS.md` 등에 그대로 있습니다.

이 폴더의 코드는 LangChain, Inc.의 MIT License를 따릅니다. 원본에 있던 `LICENSE` 파일은 수업 자료로 옮기면서 따로 두지 않았으므로, 그 전문을 아래에 그대로 싣습니다. 이 사본을 배포할 때 함께 유지해야 합니다. 원본 위에 더한 것(`deepagents_code/harness/`, 4일차 Skill, 한국어 `# [해설]` 주석, 이 문서)도 같은 조건으로 둡니다.

```text
MIT License

Copyright (c) LangChain, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
