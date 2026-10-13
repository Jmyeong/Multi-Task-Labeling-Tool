# MT-Labeling 사용 안내

```bash
cd MT-Labeling
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run_gui.sh
```

Windows CMD에서는 가상환경을 만든 뒤 `.venv\Scripts\activate.bat`, `python -m pip install -r requirements.txt`, `run_gui.bat` 순서로 실행합니다. PowerShell에서는 활성화 없이 `.\.venv\Scripts\python.exe run.py gui`로 실행할 수 있습니다. 자세한 설정은 [Windows 안내](WINDOWS.md)를 참고하세요.

기본 주소는 `http://localhost:8766`입니다. 서버에서 실행하면 VS Code **포트** 탭에서 8766을 전달한 뒤 로컬 브라우저로 접속하세요.

## 입력과 저장 위치

- `dataset/`에 이미지를 넣거나, GUI 상단 **입력·출력 폴더**에서 서버 폴더를 선택합니다. 출력은 새 폴더 경로를 직접 입력해도 됩니다.
- 원본 이미지와 입력 manifest는 변경하지 않습니다. 라벨·검수 상태·별도 manifest·백업은 선택한 출력 폴더에 저장됩니다.
- 다른 데이터셋에는 다른 출력 폴더를 사용합니다. 같은 입력·출력으로 다시 열면 기존 수정 사항을 유지합니다.
- 일반 이미지 폴더는 자동 발견됩니다. manifest가 없으면 train으로 지정합니다. train/test가 필요하면 manifest로 명시하거나 해당 하위 폴더에 배치하세요.
- NIR은 선택 사항입니다. `sequence/images/a.png`와 `sequence/nir/a.png`처럼 같은 파일명으로 배치하면 GUI에서 볼 수 있습니다.

```bash
INPUT=/path/to/images OUTPUT_DIR=/path/to/labels ./run_gui.sh
# 기존 가상환경을 사용하려면:
PYTHON=/path/to/env/bin/python INPUT=/path/to/images OUTPUT_DIR=/path/to/labels ./run_gui.sh
```

## 편집과 검수

박스 추가·이동·크기 조절·클래스 변경·삭제, 도로 브러시·다각형, Undo/Redo, 메모, 프레임 검색·필터·다중 제외 기능을 유지했습니다.

| 단축키 | 동작 |
|---|---|
| B / V | 박스 추가 / 선택 |
| R / P / G | 브러시 / 다각형 / SAM |
| 1 / 2 / 3 | 도로 / 비도로 / ignore |
| Shift+1~4 | 선택한 박스 클래스 변경 |
| Ctrl+Z / Ctrl+Shift+Z | Undo / Redo |
| Ctrl+S / Ctrl+Enter | 저장·승인 / 승인 후 다음 |
| A / D | 이전 / 다음 프레임 |
| X 또는 Delete | 선택한 박스 삭제 |

새 수동 작업은 빈 박스와 배경=0 마스크, 검수 대기 상태로 시작합니다. 기존 출력 폴더의 마스크는 불확실(ignore=255) 픽셀을 포함해 그대로 유지합니다. **임시 저장**은 대기 상태로, **저장·승인**은 승인 상태로 저장합니다. **데이터 추출**은 승인된 프레임만 새 폴더에 내보냅니다. 저장 전 라벨 백업, 중단된 저장 복구, 다른 탭과의 충돌 검사는 유지됩니다.

## SAM과 자동 라벨

SAM 설치와 가중치는 선택 사항입니다. `sam/checkpoint/`에 가중치를 넣고 SAM 실행 Python을 지정하세요. 포함점·제외점 → 미리보기 → 적용 → 저장 순서입니다.

자동 라벨 모델 구조와 가중치는 `pseudo_labeling/model/<작업>/network.py` 및 `<작업>/checkpoint/`에 나눠 둡니다. 작업은 segmentation / object_detection / multi_task입니다. RGB 단일 작업 모델 또는 기존 4클래스 RGB/NIR 공동 모델을 GUI에서 선택할 수 있습니다. 실행 방법은 아래 GUI 절차와 루트 README를 참고하세요.

기존 EasyLabeling_Tool과 8765 서버는 이 복사본과 별개입니다. MT-Labeling의 기본 포트는 8766입니다.

## 클래스 추가·제거·색상 지정

상단 **클래스·색상 설정**을 누릅니다. Bounding box와 Segmentation 각각의 추가 버튼으로 대상을 추가하고 이름·색상을 지정한 뒤 **설정 저장·적용**을 누르세요. 박스 클래스 선택 메뉴와 세그멘테이션 팔레트에 즉시 반영됩니다.

- 클래스 제거는 새 레이블링 대상에서 제외하는 동작입니다. 기존 박스·마스크·ID는 유지하며 설정에서 **복원**할 수 있습니다. 제거된 ID는 새 클래스에 재사용하지 않습니다.
- 세그멘테이션 ID 0(배경/지우기)과 255(ignore)는 필수입니다. 나머지 클래스는 2~254에서 자동 할당합니다. 색상 변경은 저장된 마스크 ID를 바꾸지 않습니다. 배경 0은 RGB가 보이도록 투명하게 표시됩니다.
- 새 클래스도 브러시·다각형·SAM으로 지정할 수 있습니다. SAM의 추가/지우기/교체는 현재 선택한 클래스를 기준으로 처리합니다. 교체할 때 다른 클래스 영역은 유지합니다.
- Shift+1~9는 활성 박스 클래스 순서, 1~9는 화면의 세그멘테이션 팔레트 순서입니다. 더 많은 클래스는 메뉴·팔레트에서 직접 선택합니다.
- 현재 프레임의 미저장 편집을 먼저 저장하거나 취소하고 설정을 적용하세요. 다른 창에서 설정을 바꾸면 이전 상태의 저장은 충돌로 차단되므로 해당 창을 새로고침하세요.
- 설정은 출력 폴더의 `label_schema.json`에 저장되며 이전 설정은 `_schema_history/`에 백업됩니다. 데이터 추출 시 설정과 `classes.txt`를 함께 저장합니다.

자동 라벨 모델의 학습 클래스는 GUI 설정과 별개입니다. 추가한 클래스는 기존 모델이 자동 검출하지 않으며 수동/SAM으로 지정해야 합니다.

## GUI에서 Pseudo labeling 실행

1. **입력·출력 폴더**에서 대상 입력 데이터를 먼저 여세요.
2. 상단 **Pseudo labeling**을 누르고 segmentation / object_detection / multi_task 모델을 선택하세요.
3. 해당 `pseudo_labeling/model/<작업>/checkpoint/`에 넣은 .pt 파일을 선택하거나 외부 경로를 지정하세요. multi_task는 호환 joint 체크포인트와 YOLO26 검출기 템플릿이 필요합니다.
4. `requirements-pseudo.txt`가 설치된 Python 실행 파일과 장치(cpu 또는 GPU 번호)를 지정하세요.
5. 현재 작업 폴더 및 원본 데이터와 분리된 **새 빈 출력 폴더**를 지정하세요. 프레임 수 0은 전체, 1은 첫 프레임 확인용입니다.
6. 단일 작업 모델은 GUI 클래스 이름으로 자동 매핑합니다. 다른 이름은 JSON으로 지정하세요. 예: `{"person": "Person"}`. 세그멘테이션은 세그멘테이션 팔레트의 이름/ID를 사용합니다. 매핑 없는 클래스는 생성하지 않고 로그에 표시합니다.
7. **생성 실행**을 누르고 진행률·로그를 확인하세요. **중단**은 추론을 종료하며 이미 생성된 파일은 남깁니다. 작업 중 창을 닫아도 생성은 계속됩니다. 생성 중 작업 폴더 변경은 차단됩니다.
8. 완료 후 **완료 결과 열기**를 누르면 새 출력 폴더를 검수할 수 있습니다. 결과는 모두 검수 대기 상태입니다.

가중치는 GitHub에 포함되지 않습니다. 세그멘테이션 단독은 마스크만, 검출 단독은 박스만 생성하며 기존 라벨에 자동 병합하지 않습니다. 공동 모델은 기존 학습 클래스 구성을 사용합니다. 일반 COCO 모델이 도로/Golfcart를 자동으로 인식하는 모델로 바뀌는 것은 아닙니다.

## 공식 모델 예시

공식 YOLO26n(객체 검출)과 Grounded-SAM(세그멘테이션)을 사용할 수 있습니다. 가중치는 GitHub에 포함하지 않으며, 아래 명령으로 각 모델의 checkpoint 폴더에 설치합니다.

```bash
python -m pip install -r requirements-grounded-sam.txt
python -m scripts.download_example_models all
```

- 객체 검출: 공식 COCO 사전학습 `yolo26n.pt`. 기본 Person 클래스는 이름으로 연결됩니다. COCO에는 Golfcart/Tree가 없으므로 다른 클래스로 자동 치환하지 않습니다.
- 세그멘테이션: `grounded_sam.json` 선택. Grounding DINO Tiny가 영어 텍스트로 박스를 찾고, SAM ViT-B가 그 박스의 마스크를 생성합니다. `{"road": 1}`은 도로 프롬프트를 GUI ID 1에 연결합니다. 여러 대상은 `{"road": 1, "water": 2}`처럼 지정하되 ID 2를 GUI에 먼저 추가하세요.
- Grounded-SAM을 선택하면 Text threshold가 표시됩니다. 두 모델은 고유 리사이즈를 사용하므로 YOLO 입력 높이·너비는 숨겨집니다. 최종 마스크는 원본 RGB 픽셀 좌표로 저장합니다.
- `.json`은 가중치 파일이 아니라 로컬 Grounding DINO/SAM 폴더 경로와 다운로드 revision을 기록한 번들 설정입니다. 실제 가중치는 그 옆의 하위 폴더에 있습니다.

추론용 Python에 모델 패키지가 필요합니다. 서버에서는 별도 `.venv-pseudo` 환경을 기본으로 사용합니다. 원본 데이터는 수정하지 않으며, 생성한 pending 결과를 검수한 뒤 승인하세요.
