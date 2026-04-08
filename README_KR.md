# AI Intro MultiLexNorm 2026 작업공간

English version: [README.md](/home/lgs/KiostKY/STUDY26/AI_intro/README.md)

이 저장소는 인공지능개론 수업의 MultiLexNorm 2026 과제를 위한 4인 팀
공용 작업공간입니다. 과제 주제는 multilingual lexical normalization으로,
소셜미디어나 커뮤니티의 noisy token을 표준 형태로 바꾸는 문제입니다.

이 저장소는 의도적으로 "Kaggle 스타일 팀 실험" 방식으로 구성했습니다.

- 공용 코드는 작고 안정적으로 유지
- 각 팀원은 루트 바로 아래 개인 폴더에서 독립적으로 가설 실험
- 매주 또는 격주로 validation 결과와 leaderboard 변화를 비교
- 유의미한 아이디어만 공용 경로로 반영

## 이 저장소에 무엇이 있나

- `baseline/`: 데이터 로드, baseline, 평가, submit 파일 생성용 공용 코드
- `ky/`, `member1/`, `member2/`, `member3/`: 팀원별 개인 실험 폴더
- `outputs/`: 생성된 제출 파일
- `Assignment Guideline.pdf`: 과제 요구사항
- `Assignment Paper Writing Guideline.pdf`: 보고서 작성 가이드

현재 개인 실험 폴더는 다음과 같습니다.

- `ky/`
- `member1/`
- `member2/`
- `member3/`

나중에 실제 이름으로 바꾸는 것은 괜찮습니다. 중요한 것은 각 팀원이
루트 아래 자기 폴더를 따로 쓰는 것입니다.

## 처음 세팅할 때

1. 루트 환경을 `uv`로 설치합니다.

```bash
uv sync --python 3.13
```

2. Hugging Face gated dataset 접근 권한을 받습니다.
   최소한 `weerayut/multilexnorm2026-dev-pub`는 승인되어야 합니다.

3. 루트 `.env` 파일에 Hugging Face 토큰을 넣습니다.

```bash
HF_TOKEN=your_token_here
```

4. 토큰만 넣는 것으로 끝이 아닙니다. 데이터셋 페이지에 직접 들어가서
   access/agree 버튼까지 눌러야 실제 다운로드가 됩니다.

## 가장 빠른 시작

한국어 baseline 점수를 확인하려면:

```bash
uv run python baseline/run_baseline.py evaluate --phase dev --lang ko
```

public dev 제출용 파일을 만들려면:

```bash
uv run python baseline/run_baseline.py submit --phase dev
```

중요: `submit`은 리더보드에 업로드하는 명령이 아니라, 업로드할 파일을
로컬에 생성하는 명령입니다.

## 저장소 구조

```text
AI_intro/
  baseline/
    run_baseline.py
    shared.py
    utils.py
    demo.ipynb
  ky/
  member1/
  member2/
  member3/
  outputs/
  .env
  pyproject.toml
  README.md
  README_KR.md
```

## 공용 코드

`baseline/`은 모든 팀원이 재사용하는 공용 코드입니다.

- [baseline/run_baseline.py](/home/lgs/KiostKY/STUDY26/AI_intro/baseline/run_baseline.py): 평가와 제출 파일 생성을 위한 CLI 진입점
- [baseline/shared.py](/home/lgs/KiostKY/STUDY26/AI_intro/baseline/shared.py): 데이터셋 로드, 예측, 제출 파일 저장 helper
- [baseline/utils.py](/home/lgs/KiostKY/STUDY26/AI_intro/baseline/utils.py): counting, MFR, ERR 평가 함수
- [baseline/README.md](/home/lgs/KiostKY/STUDY26/AI_intro/baseline/README.md): baseline 전용 설명

개인 실험 코드에서 자주 쓰는 import는 다음과 같습니다.

```python
from baseline.shared import load_local_env, load_public_dataset, predict_with_mfr
from baseline.shared import get_submission_predictions, write_submission
from baseline.utils import counting, evaluate, mfr
```

## 개인 실험 워크플로

각 팀원은 기본적으로 루트 바로 아래 자기 폴더에서 작업하면 됩니다.

추천 흐름:

1. shared baseline을 기준으로 시작합니다.
2. 변경은 우선 자기 개인 실험 코드에서만 합니다.
3. `notes.md`에 가설, 수정사항, 결과를 기록합니다.
4. validation 점수나 leaderboard 결과를 팀과 주기적으로 비교합니다.
5. 팀 논의 후 가치가 있는 로직만 공용 코드에 반영합니다.

루트에서 개인 실험을 실행하는 예시는 다음과 같습니다.

```bash
uv run python -m ky.run_experiment
uv run python -m member1.run_experiment
```

새 팀원이나 새 실험 폴더가 필요하면 기존 멤버 폴더 하나를 복사해서
시작하면 됩니다.

## 팀 운영 규칙

- 특별한 이유가 없으면 명령은 루트 디렉터리에서 실행합니다.
- 공용 로직은 `baseline/`에만 둡니다.
  데이터 접근, 평가, 공용 유틸, 제출 포맷 같은 것들입니다.
- 위험한 실험 코드는 먼저 개인 폴더에서 검증합니다.
- 공식 제출은 팀에서 합의한 하나의 리더보드 계정을 사용합니다.
- 나중에 개인 기여 보고서를 써야 하므로, 누가 무엇을 언제 바꿨는지
  기록을 남기는 편이 좋습니다.

## 자주 쓰는 명령

환경 동기화:

```bash
uv sync --python 3.13
```

전체 언어 baseline 평가:

```bash
uv run python baseline/run_baseline.py evaluate --phase dev
```

한국어 baseline 평가:

```bash
uv run python baseline/run_baseline.py evaluate --phase dev --lang ko
```

dev 제출용 zip 생성:

```bash
uv run python baseline/run_baseline.py submit --phase dev
```

full 제출용 zip 생성:

```bash
uv run python baseline/run_baseline.py submit --phase full
```

개인 실험 실행:

```bash
uv run python -m member2.run_experiment
```

## 출력 파일

공용 submit 명령은 루트 기준 `outputs/` 아래에 결과를 생성합니다.

- `outputs/submission_dev/predictions.json`
- `outputs/submission_dev.zip`
- `outputs/submission_full/predictions.json`
- `outputs/submission_full.zip`

이 파일들은 소스코드가 아니라 생성 산출물입니다.

## 자주 헷갈리는 점

- `submit`은 "제출 파일 생성"이지 "리더보드 업로드"가 아닙니다.
- Hugging Face 토큰만 있다고 끝이 아닙니다. 그 토큰의 계정이 gated
  dataset 승인 목록에 있어야 합니다.
- 공식 환경은 루트 `.venv`입니다. `baseline/.venv`는 예전 로컬 흔적이라
  공식 환경으로 보면 안 됩니다.
- validation 점수와 leaderboard 점수 둘 다 중요하지만, 과제는 보고서와
  재현성도 같이 중요합니다.

## 추천 미팅 운영 방식

1. 각 팀원이 자기 폴더에서 가설 실험을 수행합니다.
2. 각자 `notes.md`에 변경점과 결과를 기록합니다.
3. 매주 또는 격주로 팀 미팅을 합니다.
4. validation 점수, leaderboard 변화, 실패 사례, 구현 난이도를 비교합니다.
5. 최종적으로 shared path와 보고서에 넣을 아이디어를 결정합니다.

## 처음 들어온 팀원이 바로 해야 할 순서

1. 이 파일을 읽습니다.
2. [baseline/README.md](/home/lgs/KiostKY/STUDY26/AI_intro/baseline/README.md)를 읽습니다.
3. 한국어 baseline 평가를 한 번 실행합니다.
4. 루트에 있는 자기 개인 폴더를 엽니다.
5. `run_experiment.py`와 `notes.md`를 수정합니다.
6. 큰 변경 전에 작은 가설 하나부터 검증합니다.
