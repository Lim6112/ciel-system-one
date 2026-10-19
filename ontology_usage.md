# 창고 데드락을 온톨로지로 해결하기

`static/warehouse.html`의 배송 로봇 데모에서 **영구 정지(deadlock)** 를 어떻게 진단했고,
왜 해법이 "규칙을 온톨로지에 선언하는 것"이었는지, 그리고 그 선언이 실제로 어떻게
동작에 연결되는지를 기록한다.

이 문서는 *설계 의도*가 아니라 *측정된 사실*을 기준으로 쓴다. 모든 수치는 커밋
메시지와 `eval_warehouse.mjs` / 테스트 스위트에서 나온 것이며, 근거가 된 커밋 해시를
옆에 적었다.

관련 코드

| 파일 | 역할 |
| --- | --- |
| `static/warehouse_ontology.js` | TBox(선언) + ABox(사실) + `reason()` + `divergence()` |
| `static/warehouse_sim.js` | 시뮬레이터. 두 개의 시계, 질문, 상태 JSON, 후진 기동 |
| `static/warehouse_panel.js` | 온톨로지 라이브 에디터 DOM 어댑터 |
| `static/ontology_core.js` | 도메인 비의존 에디터 기계장치(세 도메인 공용) |
| `test_warehouse_ontology.mjs` / `test_warehouse_sim.mjs` | 선언-코드 일치와 게이트 검증 |

---

## 1. 문제 정의 — "데드락"은 세 가지 서로 다른 상황이다

데모에서 로봇이 영구히 멈추는 경우는 하나가 아니고, **원인이 다르면 해법도 다르다**.
이걸 한 덩어리로 다룬 것이 초기 실패의 뿌리였다.

| 상황 | 리다가 보는 것 | 왜 못 움직이는가 | 필요한 행동 |
| --- | --- | --- | --- |
| 랙(정적 장애물) | cone 안에 랙 | 랙은 절대 비켜주지 않는다 | 옆으로 비키기 (기다림은 *증명 가능하게* 틀림) |
| 정면 조우(두 로봇) | 횡오프셋 0 | 읽을 "옆"이 없다. 기하가 판정 불가 | 한 대를 선출해 전진, 다른 한 대는 양보 |
| 끼임(웨지) | **cone 안에 아무것도 없음** | 충돌 판정이 `MIN_SEP`에서 세우는데, 막는 물체가 15–20° cone **밖**에 있다 | 후진만이 기하를 바꾼다 |

세 번째가 가장 늦게 발견됐고, 조용히 실패했다. 전체 **stuck-in-contact 틱의 53.6%** 가
"cone에 아무것도 없는 끼임"이다 — 예외가 아니라 다수 케이스였다. 이 상태에서는
`scan === null`이므로 페이지가 아무 질문도 하지 않았고, 선택될 수 있는 선택지 자체가
없었다.

**측정된 비용** (15 seed × 960 s, `53e9c2f` 기록): 후진 이전에는 grid 라우터 10건 /
chain 9건의 데드락 이벤트, 최악 정지 **887 s**(러닝 전체가 사실상 정지).

---

## 2. 왜 온톨로지인가 — 규칙이 두 곳에 살면 드리프트한다

처음에는 규칙이 손으로 쓴 삼항식이었다.

```js
// warehouse_sim.js — 예전 형태
feasibleOptions(kind, stalled)  // "랙은 기다려도 소용없고, 정지한 로봇도 못 기다린다"
goldDirection(scan, stalled)    // 정답 방향 (기하에서 계산)
```

`if` 자체는 틀리지 않았다. 문제는 **검사할 수 없다**는 것, 그리고 이 규칙의
쌍둥이(`goldDirection`)와 **반드시 일치해야 하는데도** 일치를 강제하는 장치가 없다는
것이었다.

두 함수가 어긋났을 때 실제로 일어난 일:

> 에스컬레이션이 방금 `stay`를 회수한 로봇을, 게이트가 오라클의 "stay" 답변으로
> 되돌려보냈다. 로봇들은 영구적인 정면 대치에 갇혔고, 기록된 집계는
> **보류 791 대 판정 40**이었다.

즉 온톨로지 도입의 동기가 "지식 표현"이 아니라 **회귀 방지**였다. 규칙을 선언하면
양쪽 절반이 하나의 문장에서 내려오고, `divergence()`가 선언과 손으로 쓴 술어가
일치하는지 검사할 수 있다.

---

## 3. 선언 구조 (TBox)

`static/warehouse_ontology.js`는 세 번째 TBox/ABox 도메인이다 (`ontology.js` = fleet,
`breakout_ontology.js` = breakout). 도메인 비의존 에디터는 `ontology_core.js`를 공유한다.

```
StaticBlocker  ⊥ Stay               랙은 양보하지 않으므로 기다림은 증명 가능하게 틀리다
StalledBlocker ⊥ Stay               기다림이 실패한 것이 증명된 뒤에는 답이 아니다
StaticBlocker  ⊥ MobileBlocker      블로커는 한 종류뿐이다
AdvancingRobot ⊥ YieldingRobot      정면 조우에서는 정확히 한 대만 전진한다
BlockedRobot ⊑ ≥1 goldAction        막힌 로봇에게는 뚫는 행동이 있다
BlockedRobot ⊑ ≥1 permittedAction   … 그리고 합법적인 수가 최소 하나는 있다
DeadlockedRobot ⊑ permittedAction ∈ {backward, left, right}   ← 유일하게 행동을 *부여*하는 공리
```

### 3.1 부분집합 제약이 선언의 형태를 결정했다

`ontology_core.js`가 구현하는 form은 `maxCardinality` / `minCardinality` / `range` /
`disjoint` 네 가지뿐이다. **부정(negation)도 함의(implication)도 없다.**

그래서 "정적 블로커라면 `stay`는 허용되지 않는다"를 직접 쓸 수 없다. 대신
**서로소(disjointness)** 로 쓴다. `Stay`가 행동 분류체계의 클래스인 순간
"StaticBlocker의 허용집합은 Stay를 포함하지 않는다" ≡ "StaticBlocker ⊥ Stay"이므로,
같은 주장을 표현 가능한 형태로 옮긴 것이다.

`checkTBox`는 구현되지 않은 form의 공리를 **거부**한다. 즉 아무것도 평가하지 않는
선언이 조용히 남을 수 없다.

### 3.2 `backward`는 왜 ACTION_SET에 없는가

```js
export const ACTION_SET  = ["left", "stay", "right"];   // 일상적 선택
export const ESCAPE_SET  = ["backward"];                // 데드락 전용
export const DEADLOCK_SET = ["backward", "left", "right"];
```

`backward`는 행동 분류체계에는 **존재하는 클래스**다(로봇이 할 수 있는 일이므로).
하지만 일상적 선택 집합에는 **일부러 없다**. 후진은 이미 지나온 경로를 포기하는
행동이므로, 다른 세 가지가 시도되고 실패한 뒤에만 값어치가 있다.

이 규칙은 `disjoint`가 아니라 `range`로 선언된다. 이 파일에서 **유일하게 행동을
회수하지 않고 부여하는** 공리이기 때문이다. 나머지 규칙은 "무엇을 하면 안 되는지"를
말하고, 이 규칙만 "이 상황에서만 무엇이 허용되는지"를 말한다.

### 3.3 관찰 문장도 공리 위에 산다

상태가 모델에게 전달하는 문장(`stateLeaf`)은 시뮬레이터의 리터럴이 아니라
**공리 위에** 있다.

```js
stateLeaf: { deadlock: "wedged, no progress for 8s, should move backward " },
```

`buildState`가 이 값을 인자로 받아 문서에 써 넣는다. 공리를 지우면 **허가와 문장이
함께 사라진다**. 하드코딩이었다면 라이브 에디터에서 공리를 지워도 모델은 계속
"너는 끼어 있다"는 말을 듣게 된다 — 이 프로젝트가 이미 두 번 물린
"하나의 규칙, 두 개의 집" 형태다.

### 3.4 문장의 문구는 제약하지 않는다 (철회된 원칙)

한때 "관찰문은 행동을 이름 붙이지 않는다"는 원칙이 있었고, 두 번 시도됐고,
**둘 다 제거됐다**.

| 시도 | 근거 | 결과 |
| --- | --- | --- |
| `sidestep`/`left`/`right` 단어 금지 | 그 단어가 head를 해당 criteria로 끈다는 이론 | **측정으로 반박.** 두 표현 모두 `backward`를 답한다(창고 head 0.78/0.57, 기본 0.57/0.72) |
| 정답(`backward`/`reverse`) 자체를 금지 | 관찰문은 장면을 서술하고 선택은 모델에 맡긴다는 일반 원칙 | **끝내 측정된 적 없음.** 문장이 `"should move backward"`를 얻자 그대로 실패했고, 제거됨 |

#### 원칙의 동기는 실재했다 — 방법론이 문제였다

동기는 페이지의 핵심 측정 장치다. `agrees with geometry N/M`
(`warehouse.html:414`)은 Model 드롭다운 비교의 눈금이고, **그 눈금의 정당성**이
걱정이었다:

> 답이 프롬프트에 적혀 있으면, 이 카운터는 헤드를 재는 게 아니라 프롬프트를 잰다.

이 우려 자체는 옳다. 그러나 문장 하나를 봉인하는 것은 그 우려를 막는 **수단이
되지 못했다**. 답을 유도하는 것은 이 코드에서 문장뿐이 아니며, 오히려 문장이 가장
약한 요소다.

| 같은 요청 안의 요소 | 실측 효과 |
| --- | --- |
| `obstacle_side` 필드 | 있을 때 `left` 1.00 / `backward` 0.00, 없을 때 `backward` 0.90 — **답 전체를 결정** |
| 정면 조우 side 재작성 | 25/25 상태가 `stay` → 옆걸음(93–99%). 재학습 없이 문구만으로 |
| deadlock `instructions` | "Sidestepping has not freed you. What should you do now?" — 전제를 못박고 행동을 요구 |
| `stay` criteria | "yield and wait" — 행동 이름은 물론 전제까지 부정 |
| ↑ 그리고 **문장 하나** | 선택을 **바꾸지 못함**(0.94 대 0.94) |

`obstacle_side` 재작성은 이 프로젝트가 **자랑하는** 수정이다(재학습 회피). 즉 "관찰문은
답을 말하지 않는다"는 시스템이 *적극적으로* 답을 유도하는 와중에 문자열 하나에만
걸린 규칙이었다 — 원칙이 아니라 그 문장에 대한 각서.

더 근본적으로, **보호한다고 주장한 측정을 애초에 보호하지 못했다.** `agrees`는
"criteria 라벨 위 argmax가 기하를 따라가는가"를 재는데, 그 criteria 텍스트와 상태
텍스트는 **기하를 따라가도록 손으로 쓴 것**이다. 답을 결정하는 것이
`obstacle_side`라는 게 측정된 이상, 이 카운터가 재는 것은 "헤드가 장면을
추론했는가"가 아니라 **"상태 구성이 기하와 일관되게 작성됐는가"** 다. 문장 하나를
봉인한다고 달라질 것이 없었다.

원칙은 두 번 모두 **측정 없이** 세워졌고, 그 제약은 "측정된 이득 없이 더 명확할 수
있는 문구를 금지"하는 형태였다. 이 문서가 8.2/8.3에서 세운 규율("측정 없이
주장하지 않기")을 이 원칙 자신이 계속 위반하고 있었던 셈이다.

#### 남는 진짜 질문

"행동을 이름 붙이지 말 것"이 아니라, **어느 측정이 헤드의 것이고 어느 것이 상태
구성의 것인지 구분하는 것**이다. `agrees with geometry`는 계속 유효한 눈금이지만
README가 말하는 것보다 좁다 — 최소한 `obstacle_side`가 결정하는 상태들에서는.

현재 문장은 행동을 이름 붙인다. 남는 비용은 실재하고, 반박된 것이 아니라
**감수된** 것이다. 문구가 아무리 바뀌어도 답을 움직이는 것은 **어느 sub-path로
도달하는가**와 `obstacle_side`의 유무이며, 이 문자열은 둘 다에 영향을 줄 수 없다.
그래서 테스트도 문구를 단언하지 않고 **문장과 허가가 함께 존속·소멸하는지**만
검증한다.

---

## 4. ABox — 사실은 시뮬레이터가, 기하는 확장이 공급한다

`buildABox(robot, scan, opts)`는 **살아 있는 시뮬레이터에서** 사실을 읽는다.
추론된 것은 없고, 온톨로지가 **주입받아야 하는 유일한 것**은 어느 쪽이 비었는지 —
즉 **기하**다.

```js
const options = opts.feasibleOptions ?? feasibleOptions;   // 기하는 확장으로
const gold    = opts.goldDirection   ?? goldDirection;     // 두 번째 구현은 두 번째 버그
```

fleet이 `detour`를, breakout이 `rollout`을 주입하는 것과 같은 구조다. 비용·기하는
논리가 아니므로, 기하를 다시 구현하면 틀릴 곳이 하나 더 생길 뿐이다.

### 4.1 데드락 게이트는 lidar가 아니라 두 개의 시계다

```js
export const DEADLOCK_MS = 8000;   // STALL_MS(4 s)보다 길다
export function isDeadlocked(stuckMs, blockedMs) {
  return stuckMs >= DEADLOCK_MS && blockedMs >= DEADLOCK_MS;
}
```

- `stuckMs`  = 진전 없이 지난 시간 (마지막 progress 이후)
- `blockedMs` = 연속으로 막혀 있던 시간

**둘 다 필요하고 어느 하나도 충분하지 않다.** 순서를 기다리는 로봇은 blocked이지만
stuck이 아니고, 긴 복도를 달리는 로봇은 progress 샘플 사이에 stuck처럼 보이지만
blocked가 아니다. **논리곱만이 데드락이다.**

`DEADLOCK_MS`가 `STALL_MS`(4 s)보다 긴 것도 의도다. 에스컬레이션은 먼저 경로를
지키는 수단을 시도해야 한다: 4초에 `stalled`가 `stay`를 회수하고, 그것마저 실패한
뒤에야 후진할 값어치가 생긴다. 8초는 이 프로젝트가 이미 "영구 정지"라고 부르던
수치다.

---

## 5. 실행 경로 — 선언이 행동이 되는 다섯 단계

```
step()                          두 시계를 갱신 → isDeadlocked()
   ↓
permittedFor(robot, scan, stalled, deadlocked)      [warehouse_panel.js]
   ↓
permittedActions()              [warehouse_ontology.js]
   ├─ buildABox()               사실 구성. escape 집합을 공리에서 읽는다
   └─ reason()                  공리 대조 → violations / unasserted / derived
   ↓
{ options, role, deadlocked, stateLeaf, violations }
   ↓
buildQuestion(kind, stalled, deadlocked, permitted)  criteria = options
buildState(..., role, deadlocked, stateLeaf)
   ↓
모델 호출 → applyDecision()
```

### 5.1 공리가 "검사"가 아니라 "출처"가 되는 유일한 지점

```js
const escapeRule = axiom("deadlocked-robot-may-reverse");
const escapeSet = (escapeRule?.value?.length) ? escapeRule.value : null;
const permitted = (deadlocked && escapeSet) ? escapeSet : ordinary;
```

파일의 나머지 규칙은 반대 방향으로 동작한다: `feasibleOptions`가 집합을 계산하고
공리가 그 계산을 **검증**한다. 그러므로 공리를 지우면 검사만 멈추고 선택지는
그대로다 — 구현이 삼항식인 규칙에는 그게 맞다.

하지만 이 규칙의 내용 전체가 **"온톨로지가 이 행동을 허가한다"** 이므로 다르다.
리터럴 `DEADLOCK_SET`을 두면 공리의 `value`는 이미 쓰고 있는 집합과 비교만 될
뿐이고, 에디터에서 선언을 지워도 탈출은 멀쩡히 동작한다 — **장식**.

라이브 에디터에서 공리를 지웠을 때 실측(`8ee2785`):

```
공리 있음       options [backward, left, right]   state "wedged, ..."
공리 삭제       options [left, stay, right]        state (없음)
복원            options [backward, left, right]   state "wedged, ..."
```

**공리가 없으면 탈출도 없다.** 로봇은 하드코딩된 탈출로가 아니라 평범한 집합으로
폴백하고, `reason()`은 그 공리를 `unasserted`로 보고한다. 이것이 선언을
load-bearing으로 만든다.

### 5.2 후진은 양보 홀드보다 우선순위가 높다

```js
if (now < robot.backingUpUntil) {
  const back = backwardVelocity(robot, speed);   // yield 홀드보다 먼저
  vx = back.vx; vy = back.vy;
} else if (held || yielding) { /* 정지 */ }
```

이 순서가 틀렸던 첫 버전은 **존재하는 유일한 상황에서 단 한 번도 실행되지 않았다**.
정면 조우는 항상 한 대를 양보자로 선출하고, 양보 홀드는 "전혀 움직이지 않음"이다.
따라서 선출된 양보자는 매 틱 `backward`를 받아도 움직일 수 없었다. 측정 결과:
15 seed에서 **후진 선택 38,121건, 데드락은 완전히 동일**, 최악 정지 887 s(전후 동일).

같은 이유로 dock zone 홀드도 무시한다. 두 홀드 모두 "어딘가로 **진입**하지 못하게"
막는 규칙인데, 후진은 경합 지대에서 **멀어지는** 운동이다. 홀드를 존중하면 규칙의
목적을 깨뜨려 규칙을 지키는 셈이 된다.

후진 속도는 현재 heading의 정확한 **역방향**이다. 랜덤워크가 아니라 "앞에 있는
무엇에게서 물러나기" — 공간을 여는 것이 보장된 유일한 방향이고, 새 곳으로 보내는
대신 이미 얻은 땅을 대가로 낸다. yaw는 고정되므로, 기동이 끝나면 heading은 여전히
장애물을 향하고 경로는 그대로 재개할 수 있다.

### 5.3 질문의 criteria가 곧 메뉴다

`ciel.py`는 모델의 선택지 목록을 **criteria의 KEY**로 만든다. 즉 criteria 객체가
문자 그대로 메뉴다 — 온톨로지의 허용집합은 거기까지 **도달하지 않는다**.

수정 전에는 데드락 질문이 `backward, left, stay, right`를 나열했고, 공리는 이미
`stay`를 회수한 상태였다. 모델은 온톨로지가 허가한 적 없는 집합에서 고르고 있었다.

```js
buildQuestion(kind, stalled, deadlocked, permitted)   // permitted로 criteria를 필터
```

실측: 수정 전 전송 `[backward, left, stay, right]` → 수정 후 `[backward, left, right]`.

### 5.4 데드락에서는 `obstacle_side`를 일부러 뺀다

`obstacle_side`는 로봇을 어떻게 돌아갈지 알려주는 필드이고, 창고 헤드는 그 매핑을
강하게 학습했다. **필드가 있으면 모든 데드락 상태에서 신뢰도 1.00으로 `left`를
답하고 `backward`를 완전히 억제한다**(`f98a602`).

실제 웨지 상태에서 필드를 하나씩 제거해 분리한 결과:

```
path=warehouse   →  left      backward=0.00  left=1.00  right=0.00
path=(default)   →  backward  backward=0.48  left=0.30  right=0.22
```

`obstacle_side` 하나가 **원인의 전부**다. 다른 어떤 필드를 빼도 답은 바뀌지 않고,
그것만 제거하면 같은 상태가 `backward`를 0.90으로 답한다.

빼는 것이 정보를 숨기는 것도 아니다. 데드락 로봇에게도 "옆"은 있지만,
`DEADLOCK_MS` 동안 옆으로 비켜보고도 못 빠져나온 상태에서 그 옆이란 **실패가 이미
무효화한 사실**이다. 실패를 초래한 지시를 다시 읽어주는 것은 데이터를 감추는 게
아니라 무효한 정보를 재진술하는 것이다. `obstacle_position`은 남으므로 장애물이
어디 있는지는 여전히 말해준다.

정면 조우용 side 재작성도 같은 조건에서 건너뛴다. 그 재작성은 "정지 → 옆걸음"을
만들기 위한 것인데, 데드락은 이미 그 옆걸음이 실패한 상태다.

---

## 6. 검증 — 선언이 worth having해지는 지점

### 6.1 `divergence()`: 선언과 손으로 쓴 술어의 대조

```js
const direct   = feasibleOptions(c.kind, c.stalled, role);
const declared = permittedActions(c.robot, c.scan, { stalled: c.stalled }).options;
if (join(direct) !== join(declared)) broken.push({ ...c, role, direct, declared });
```

시뮬레이터가 만들어낼 수 있는 **모든** 블로킹 상황에 대해 둘 다 실행한다 —
10 seed에 걸쳐 **20,000 케이스**, `test_warehouse_ontology.mjs`가 불일치 0을 단언한다.

> 이 검사가 없으면 온톨로지는 규칙의 *두 번째* 집이 되고, 그것은 하나보다
> 엄격히 나쁘다. 가상의 시나리오가 아니다: 이 규칙의 두 절반(어떤 선택지를
> 주는가 / 어떤 답이 정답인가)이 별개 함수에 살다가 드리프트해서 로봇들을 영구
> 정면 대치에 가둔 것이 **보류 791 대 판정 40**이었다.

주의: `divergence()`는 role을 **비교에 포함해야 한다**. 이를 빼면 실제 오탐이
생긴다 — 온톨로지는 role을 계산해 전진 로봇의 `stay`를 회수하는데 직접 호출은
하지 않아, 둘이 2,476회 불일치했다(검사는 맞았고 하네스가 옛 시그니처와 비교했다).

### 6.2 이 도메인이 주장할 수 있는 것 / 없는 것

| 도메인 | 주장 | 성립 여부 |
| --- | --- | --- |
| fleet | 온톨로지를 켜도 어떤 결정도 바뀌지 않는다 | 구성상 성립 |
| warehouse | 선언된 규칙 = 손으로 쓴 술어 (모든 블로킹 상황에서) | `divergence()`로 검사 |

창고에서는 "켜도 결정이 안 바뀐다"가 **성립할 수 없다**. 선택지를 제거하는 것이
바로 그 변경이고, 그게 설계 의도다 — 절대 움직이지 않는 장애물 앞에서 "기다림"은
답이 아니라는 것이 이 규칙의 전부이기 때문이다.

그래서 패널은 더 약하고 **검사 가능한** 주장을 대신 진술한다. 데모의 선택지 집합이
온톨로지를 통과하므로 패널은 실제로 작동 중인 규칙을 보여준다 — 랙 블록은
`permitted left/right`로 읽힌다.

### 6.3 판정 vs 보류의 분리

`reason()`은 두 가지를 구분해서 돌려준다.

- `violations` — 실제로 위반된 공리 (버그이거나, 잘못된 답)
- `unasserted` — 에디터에서 지워져 **검사가 꺼진** 공리

모든 공리는 **자기 존재 여부로 가드된다**. 지우면 그 검사가 비활성화되고,
크래시 대신 `unasserted`로 보고된다. 시뮬레이터는 여전히 답을 알고 있고, 온톨로지가
단지 그것을 더 이상 단언하지 않을 뿐이다 — 그 사실을 말하는 것이 에디터의 존재
이유다.

---

## 7. 측정 결과

### 7.1 후진 옵션의 효과 (15 seed × 960 s, `53e9c2f`)

| 라우터 | 설정 | 배송량 | 데드락 이벤트 | 최악 정지 |
| --- | --- | --- | --- | --- |
| grid | 탈출 off | 1,256 | 10 | 911 s |
| grid | **탈출 on** | **4,453** | 8 | **0 s** |
| chain | 탈출 off | 1,314 | 9 | 869 s |
| chain | **탈출 on** | **3,963** | 4 | **0 s** |

800 s 이상의 영구 정지가 두 라우터 모두에서 사라진다. 후진은 약 15 시뮬레이션 시간
동안 grid 41회 / chain 28회 발생 — **드물게 발동하고, 발동할 때는 그 비용 전체의
값어치를 한다.**

### 7.2 필드 제거 전후 (실제 웨지, 창고 sub-path, `f98a602`)

| | 결과 |
| --- | --- |
| `obstacle_side` 있을 때 | `left` 신뢰도 1.00, `backward` 0.00 |
| `obstacle_side` 없을 때 | `backward` 0.90 |
| 수정 후 (4 seed) | 0.00 → **0.93 / 0.95 / 0.97** |
| 5-seed 실주행 | 데드락 19건, **최대 지속 0.3 s** (작업 이전에는 887 s) |

### 7.3 어느 sub-path로 가는가가 실제 결정 요인 (`561442b`)

같은 웨지 상태, 같은 세 선택지로 측정:

| sub-path | `backward` 신뢰도 |
| --- | --- |
| `warehouse` | 0.81 / 0.92 / 0.99 / 0.77 (네 가지 상태 형태) |
| 서비스 기본(doc) | 0.58 / 0.69 / 0.59 / 0.49 |
| `typed` | `left` 또는 `right`. `backward`는 0.33을 넘지 못함 |

`typed` 헤드는 창고 학습이 없어 이 질문에 답할 수 없다. 그런데도 **전송받은
선택지 중 하나는 항상 반환**하므로, 그 실패는 out-of-domain 답변이 아니라
**고려된 `left`** 처럼 보인다. `ciel_typed_v1`이 Model 드롭다운에서 여기에 도달한다.

---

## 8. 실패와 교훈 — 이 섹션의 값어치

### 8.1 코드가 아니라 구조가 원인이었던 것

| 증상 | 실제 원인 |
| --- | --- |
| 후진을 골라도 전혀 안 움직임 | 역주행 분기가 `held \|\| yielding` **아래**에 있었다. 양보자가 매 틱 후진을 받아도 이동 불가 |
| 끼었는데 아무 선택지도 없음 | `scan === null`(cone 밖 블로커). 게이트가 lidar였다 — 다수 케이스(53.6%)가 여기 |
| 옵션은 오는데 로봇이 멈춤 | `reason()`의 조회가 `BlockedRobot \|\| Robot`만 필터. `DeadlockedRobot`을 못 찾아 `derived.permitted`가 **빈 배열** → 평범한 집합으로 폴백 |
| 모든 데드락 상태에서 공리 위반 | `DEADLOCK_SET`에 `stay`가 있었다. 데드락은 정의상 stalled보다 길어서 stalled-blocker 공리를 매번 위반. `stay`는 무해한 추가처럼 읽히지만 **에스컬레이션 전체가 제거하려는 그 옵션**이다 |
| 페이지에서 탈출이 완전히 죽음 | `permittedFor` 래퍼가 `stateLeaf`를 전달 목록에 안 넣음 + 페이지 `step()`이 lidar가 비면 `blockedSince`를 0으로 리셋(정확히 데드락 케이스에서 두 시계가 리셋됨) |
| 모델이 온톨로지가 허가 안 한 집합에서 선택 | criteria KEY가 곧 메뉴인데 `permitted`가 전달되지 않음 |

### 8.2 측정 오류 — "된다"고 믿었던 두 번

둘 다 코드 버그가 아니라 **측정 오류**였고, 그 형태를 기록해 둔다.

1. **첫 A/B가 모델을 아예 호출하지 않았다.** `goldDirection` + `applyDecision`을
   직접 부른 결과, 911 s → 0 s는 **오라클이** 데드락을 깬 것이지 데모가 아니었다.
2. **두 번째는 "헤드는 `backward`를 낼 수 없다(학습된 적 없으니)"고 가정하고 그
   가정을 시뮬레이션했다.** 거짓이다 — 서비스는 criteria 라벨을 채점하므로
   학습되지 않은 라벨도 답할 수 있다.

그리고 **405를 모델의 의견으로 읽은 사례**: `ciel_warehouse_v1`이 "무관심하다"는
세 번의 관측(4개 라벨에 걸친 균일 0.250)은 Decision Lab 프록시가 405를 반환한
서비스에 물어본 결과였다. **균일한 분포는 모델의 판단이 아니라 전송 실패였다.**

### 8.3 귀속(attribution)을 틀렸던 두 번

데드락이 `left`를 답했을 때 원인 후보로 두 가지가 지목됐고, **둘 다 측정으로
반박**됐다.

| 가설 | 측정 결과 |
| --- | --- |
| 관찰문이 `sidestep`을 언급해서 `left`/`right` criteria로 끌린다 | **거짓.** 짧은 문구와 긴 설명 문구 모두 `backward`를 같은 신뢰도(0.94)로 답한다. 문장은 선택을 바꾸지 못한다 |
| `typed` sub-path가 범인이다 | **원인은 아님.** 실제로 이 질문은 실패하지만, 기본으로 도달하는 헤드는 창고 헤드다 |
| `obstacle_side` | **진짜 원인.** 필드 하나가 답 전체를 결정한다 |

교훈: "X 때문에 이렇게 바꿨다"는 나중에 믿어지는 종류의 주장이다. 그리고 여기서
한 발 더 나아가, **측정되지 않은 원칙도 같은 방식으로 제거된다.** 문구를
제약하던 두 어서션 — `sidestep` 단어 금지, 그리고 정답 자체 금지 — 은 각각
반박된 이론과 끝내 측정되지 않은 일반 원칙에 근거했고, 둘 다 사라졌다(3.4).
관찰문 문구는 온톨로지 작성자의 선택이며, 테스트가 지키는 것은 문구가 아니라
**문장과 허가가 함께 존속·소멸한다**는 성질이다.

---

## 9. 라이브 에디터 — 선언을 지워보는 것이 데모다

`Ontology` 패널에서 공리를 지우면:

- 그 공리의 검사가 비활성화되고 `unasserted`로 보고된다 (크래시 없음)
- 탈출 공리를 지우면 **후진 허가와 관찰 문장이 함께 사라진다**
- `Reset`은 페이지 마운트 이전의 스냅샷(`PRISTINE`)으로 되돌린다 — 세션 중의
  실수 편집이 "원본"이 되지 않도록

**집행은 스위치가 아니다.** breakout의 on/off 버튼이나 fleet의 체크박스와 달리,
창고 규칙은 선택 사항이 아니다. 선택지 집합이 곧 이 샘플이 보여주는 것이고,
페이지에는 이미 geometry gate라는 스위치가 있다. 경쟁하는 두 번째 스위치는
주제 자체가 그 규칙인 페이지에서 규칙을 끌 수 있다는 인상을 줄 뿐이다.

dock-zone 공리는 **일부러 여기 없다.** 시뮬레이터에는 구현돼 있지만 순손실로
측정돼(처리량 약 40%) 기본 off로 출하되므로, 여기에 선언하면 데모가 집행하지 않는
규칙을 광고하게 된다. 그 부정적 결과는 `warehouse_sim.js`의 `DOCK_ZONE_RADIUS`
위에 기록돼 있다.

---

## 10. 재현 방법

```bash
# 선언과 술어의 일치 (시뮬레이터가 만드는 모든 블로킹 상황)
node --test test_warehouse_ontology.mjs

# 게이트 양방향, 속도, 상태 계약, 랙/정면 브랜치
node --test test_warehouse_sim.mjs

# 오라클 대비 정확도 (서비스 필요, warehouse sub-path)
node eval_warehouse.mjs --model ciel_warehouse_v1

# 브라우저
# http://localhost:8000/warehouse.html — Model 드롭다운에서 sub-path 비교
```

데드락을 눈으로 보려면 autoplay를 끈다: 막힌 로봇들이 영원히 멈춘다 — laya가
막는 바로 그 정지다.

---

## 11. 한계 (주장 범위를 넘지 않기 위해)

- **오라클 하네스는 피드백 루프가 아니다.** `eval_warehouse.mjs`는 시뮬레이터를
  구동하지만 답을 다시 시뮬레이터에 넣지 않는다. 따라서 샘플링하는 시나리오는
  시뮬레이터가 **저항 없이** 도달한 것들이고, 진짜 다중 로봇 데드락을
  과소표본한다(기다리라는 말을 들은 적 없는 로봇은 정지하지 않는다). 기록된 한
  회의 실행에서 40/40이 랙 블록이었다. 결정을 피드백하는 것이 명백한 다음 단계다.
- **이 도메인의 온톨로지 주장은 나머지 두 데모보다 약하다** (6.2).
- **`stateLeaf` 문구에 대한 강한 주장은 어느 방향으로도 지지되지 않는다.**
  측정된 것은 sub-path와 `obstacle_side`의 유무뿐이다 (8.3).
- **`agrees with geometry`는 README가 말하는 것보다 좁은 눈금이다.** 그 카운터는
  "헤드가 장면을 추론했는가"가 아니라 "상태 구성이 기하와 일관되게 작성됐는가"를
  재는 몫을 갖는다 — 최소한 `obstacle_side`가 답을 결정하는 상태들에서는 (3.4).
  두 측정을 분리하는 일은 아직 **하지 않았다**.
- **후진의 편향**: 로봇은 들어온 선을 따라 물러난다. 경로를 바꾸지 않으므로
  재개할 수 있지만, 좁은 공간에서 다른 로봇과의 상호작용은 별도로 측정되지 않았다.

---

## 부록: 관련 커밋

| 커밋 | 내용 |
| --- | --- |
| `53e9c2f` | 후진 옵션 도입 (데드락에서만), 6 파일 +481 |
| `21148db` | 데드락 질문을 온톨로지 판정에서 생성 (criteria = 메뉴) |
| `8ee2785` | 탈출 공리가 능력의 **출처**가 됨 (장식 → load-bearing) |
| `9c4a745` | 페이지에서 데드락 상태와 탈출이 도달 불가였던 두 가지 파손 |
| `561442b` | 문장 단축 + sub-path가 결정 요인임을 기록 (반증된 가설 정정) |
| `f98a602` | `obstacle_side` 보류 — `left` 답변의 진짜 원인 |
| `43843d5` | lidar 거리 기본값 70 → 60 px |
| `b265bda` | 문장에 `backward` 추가 — 문구를 제약하던 원칙과 어서션을 철회시킨 변경 (3.4) |
