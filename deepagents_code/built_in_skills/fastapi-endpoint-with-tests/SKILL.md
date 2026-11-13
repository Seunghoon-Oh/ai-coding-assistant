---
name: fastapi-endpoint-with-tests
description: >
  FastAPI 프로젝트에 엔드포인트(GET/POST 등 API, 라우터, CRUD)를 추가하거나 수정하라는 요청에 사용한다.
  "엔드포인트"라는 말이 없어도 FastAPI에서 API를 만드는 요청이면 쓴다.
  계획을 승인받은 뒤 성공·실패 테스트와 함께 만든다.
  일반 코딩 질문에는 사용하지 않는다.
---

<overview>
엔드포인트만 만들고 테스트를 빼면 그 변경이 맞는지 아무도 확인할 수 없다.
이 스킬은 엔드포인트를 추가할 때 "요구 확인 → 기존 구조 읽기 → 계획 제시와 승인 → 스키마·데이터 접근·라우터·등록 → 테스트 → 실행 결과 보고" 순서를 지키게 한다.
</overview>

<python-rules>
- PEP8을 따른다: 들여쓰기 4칸, 적절한 공백과 줄바꿈.
- 이름: 함수·변수는 snake_case, 클래스(Pydantic 모델 포함)는 PascalCase.
- import는 표준 라이브러리 → 서드파티 → 프로젝트 순으로 묶고, 쓰지 않는 import는 지운다.
- 주요 함수에는 역할과 입출력을 설명하는 docstring을 쓴다. 필요한 곳에만 주석을 단다.
</python-rules>

<workflow>
기존 엔드포인트를 수정하는 요청이면 1~3단계와 8~9단계는 그대로 하고, 4~7단계는 바뀌는 부분에만 적용한다.
수정에는 새 스키마나 라우터 등록이 필요 없는 경우가 많기 때문이다. 바뀐 동작의 성공·실패 테스트는 추가하거나 고친다.

1. 요구를 정리한다: HTTP 메서드, 경로, 입력(경로·쿼리·본문), 출력, 거절해야 하는 경우.
   요청에 빠진 것이 있으면 임의로 정하지 말고 묻거나, 가정이라고 적는다.
2. 기존 구조를 읽는다: 라우터 위치, 스키마 위치, DB 접근 방식, 의존성 함수, 이름 규칙, 기존 테스트의 위치와 스타일.
3. 계획을 쓰고 사용자에게 보여 준다: 수정할 파일, 작업 순서, 테스트 방법.
   사용자가 승인하기 전에는 파일을 만들거나 고치지 않고 읽기만 한다. 방향이 틀렸을 때 계획 단계에서 바로잡는 것이 코드를 고친 뒤 되돌리는 것보다 싸기 때문이다.
   사용자가 수정 의견을 주면 계획에 반영한 뒤 다시 보여 주고 승인을 기다린다.
   작업 범위가 계획과 달라지면 계획을 고쳐 다시 보여 주고 승인을 받는다.
4. 입력·출력 스키마(Pydantic 모델)를 정의한다. 입력 검증은 스키마에 맡기고 직접 만들지 않는다.
   스키마의 필드에는 `description`과 예시를 달아 문서에 나오게 한다.
5. 데이터 접근과 의존성을 정한다.
   - 저장·조회가 필요하면 기존 프로젝트의 DB 접근 방식(세션, ORM, 저장소 계층)을 따른다.
   - 라우터 함수 안에서 연결을 직접 만들지 않고 `Depends`로 받는다. 현재 사용자 같은 공유 자원도 같다. 의존성으로 받으면 테스트에서 바꿔 끼울 수 있기 때문이다.
   - DB 모델과 API 응답 스키마는 분리한다. DB 모델을 그대로 응답에 쓰면 내부 필드가 드러난다.
6. 라우터 함수를 구현한다: `response_model`과 상태 코드를 지정하고, 실패는 `HTTPException`으로 알맞은 상태 코드(404, 409 등)를 돌려준다.
   문서는 FastAPI가 OpenAPI로 자동 생성하므로 `summary`를 달고, 코드가 돌려주는 오류 응답은 `responses`로 선언한다.
7. 새 라우터를 앱에 등록한다. 등록을 빼면 엔드포인트가 만들어져도 호출되지 않는다.
8. 테스트를 쓴다.
   - 성공 케이스 1개 이상, 실패 케이스 1개 이상이 모두 있어야 한다. 성공 테스트만 있으면 엔드포인트가 거절해야 할 때 거절하지 않아도 알 수 없기 때문이다. 파일만 있고 한쪽 케이스가 없으면 테스트를 만든 것으로 치지 않는다.
   - 실패 케이스는 엔드포인트가 거절해야 하는 경우(404, 409, 422 등)다. 목록 조회처럼 거절할 경우가 자연스럽게 없어 보여도 건너뛰지 않고, 쿼리·경로 파라미터 검증에 걸리는 잘못된 입력(422)을 쓴다. 검증할 입력이 정말 없으면 그 사실과 이유를 보고에 적는다.
   - 로그인이나 인증이 걸린 엔드포인트는 아래 security 구간의 규칙을 따르고, 인증 실패(401·403)도 실패 케이스에 포함한다.
   - DB 같은 의존성은 `app.dependency_overrides`로 테스트용 값으로 바꾸고, 테스트가 끝나면 되돌린다. 테스트가 실제 DB나 운영 값에 닿지 않게 하고, 테스트끼리 데이터를 공유하지 않게 한다.
9. 테스트를 실행한다.
   - 프로젝트가 쓰는 테스트 명령으로 실행한다. 명령은 기존 설정, README, CI 설정에서 찾고, 보통은 `pytest`다.
   - 테스트에 필요한 패키지(예: `TestClient`가 쓰는 `httpx`)가 없으면 임의로 설치하지 말고 사용자에게 묻는다.
   - 실패하면 원인을 읽고 고친 뒤 다시 실행한다. 실행하지 못했으면 못 했다고 적는다.
</workflow>

<security>
로그인, 사용자 정보, 비밀번호·토큰처럼 민감한 값을 다루는 엔드포인트에 적용한다.
- 기존 프로젝트에 인증 방식(의존성, 미들웨어)이 있으면 새로 만들지 않고 재사용한다.
- 비밀번호는 평문으로 저장하거나 비교하지 않는다. 프로젝트가 쓰는 해시 방식을 따르고, 없으면 검증된 비밀번호 해시 라이브러리(bcrypt, argon2 등)를 쓴다. 해시를 직접 만들지 않는다.
- 응답 모델에는 비밀번호·해시·토큰 원문 같은 민감 필드를 넣지 않는다. 응답 스키마를 요청 스키마와 따로 둔다.
- 로그와 오류 메시지에 비밀번호나 토큰을 남기지 않는다.
- 인증 실패 메시지는 아이디가 없는지 비밀번호가 틀린지 구분하지 않는다. 계정 존재 여부가 드러나기 때문이다.
- 비밀 값(키, 비밀번호 원문)을 코드에 직접 쓰지 않고 설정이나 환경 변수에서 읽는다.
- 테스트에 쓰는 계정과 비밀번호는 테스트 전용 값으로 만든다.
</security>

<best-practice>
아래는 모양을 보여 주는 예시다. 경로와 이름은 기존 프로젝트의 규칙이 우선한다.
저장소는 설명을 짧게 하려고 딕셔너리로 만들었다. 실제 프로젝트에서는 `get_items_store` 자리에서 DB 세션을 열고 닫는다.

```python
# app/schemas/item.py
from pydantic import BaseModel, Field


class ItemCreate(BaseModel):
    """항목 생성 요청 본문."""

    name: str = Field(min_length=1, description="항목 이름", examples=["pen"])


class ItemOut(BaseModel):
    """항목 응답."""

    id: int = Field(description="항목 번호")
    name: str = Field(description="항목 이름")
```

```python
# app/deps.py
from app.schemas.item import ItemOut

_ITEMS: dict[int, ItemOut] = {}


def get_items_store() -> dict[int, ItemOut]:
    """항목 저장소를 돌려준다. 라우터는 이 함수를 Depends로 받는다."""
    return _ITEMS
```

```python
# app/routers/items.py
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.deps import get_items_store
from app.schemas.item import ItemCreate, ItemOut

router = APIRouter(prefix="/items", tags=["items"])


@router.post(
    "",
    response_model=ItemOut,
    status_code=status.HTTP_201_CREATED,
    summary="항목 만들기",
    responses={409: {"description": "같은 이름의 항목이 이미 있음"}},
)
def create_item(
    body: ItemCreate,
    store: dict[int, ItemOut] = Depends(get_items_store),
) -> ItemOut:
    """항목을 만든다. 같은 이름이 이미 있으면 409."""
    if any(item.name == body.name for item in store.values()):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="item exists",
        )
    item = ItemOut(id=len(store) + 1, name=body.name)
    store[item.id] = item
    return item


@router.get(
    "/{item_id}",
    response_model=ItemOut,
    summary="항목 하나 조회",
    responses={404: {"description": "항목이 없음"}},
)
def get_item(
    item_id: int,
    store: dict[int, ItemOut] = Depends(get_items_store),
) -> ItemOut:
    """item_id에 해당하는 항목을 돌려준다. 없으면 404."""
    item = store.get(item_id)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="item not found",
        )
    return item


@router.get("", response_model=list[ItemOut], summary="항목 목록 조회")
def list_items(
    limit: int = Query(10, ge=1, le=100, description="가져올 최대 개수"),
    store: dict[int, ItemOut] = Depends(get_items_store),
) -> list[ItemOut]:
    """항목 목록을 최대 limit개 돌려준다. limit이 1~100 밖이면 422."""
    return list(store.values())[:limit]
```

```python
# app/main.py (등록)
app.include_router(items.router)
```

```python
# tests/test_items.py
import pytest
from fastapi.testclient import TestClient

from app.deps import get_items_store
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_store():
    """테스트마다 빈 저장소로 바꿔 끼우고, 끝나면 되돌린다."""
    store = {}
    app.dependency_overrides[get_items_store] = lambda: store
    yield
    app.dependency_overrides.clear()


def test_create_item_ok():
    res = client.post("/items", json={"name": "pen"})
    assert res.status_code == 201
    assert res.json()["name"] == "pen"


def test_create_item_duplicate():
    client.post("/items", json={"name": "ruler"})
    res = client.post("/items", json={"name": "ruler"})
    assert res.status_code == 409


def test_create_item_invalid():
    res = client.post("/items", json={"name": ""})
    assert res.status_code == 422


def test_get_item_ok():
    created = client.post("/items", json={"name": "eraser"}).json()
    res = client.get(f"/items/{created['id']}")
    assert res.status_code == 200
    assert res.json()["name"] == "eraser"


def test_get_item_not_found():
    res = client.get("/items/999")
    assert res.status_code == 404


def test_list_items_ok():
    client.post("/items", json={"name": "pen"})
    res = client.get("/items", params={"limit": 5})
    assert res.status_code == 200
    assert len(res.json()) == 1


def test_list_items_invalid_limit():
    res = client.get("/items", params={"limit": 0})
    assert res.status_code == 422
```
</best-practice>

<existing-code>
기존 코드가 있을 때:
- 폴더 구조, 이름 규칙, 테스트 스타일을 새로 만들지 않고 기존 것을 따른다.
- 기존에 있는 DB 접근 방식과 의존성 함수는 새로 만들지 않고 재사용한다.
- 기존 테스트가 깨지면 고치지 말고 어떤 테스트가 왜 깨졌는지 보고한다.
- 기존 코드가 없는 빈 프로젝트면 위 예시의 구조를 출발점으로 쓰고, 그렇게 정했다고 보고한다.
</existing-code>

<report>
작업이 끝나면 다음을 적는다:
- 승인받은 계획: 사용자가 승인한 계획과, 작업 중 범위가 바뀌어 다시 승인받은 내용
- 변경한 파일
- 추가한 테스트: 성공 케이스와 실패 케이스를 구분해서
- 테스트 실행 결과
- 확인하지 못한 것
</report>
