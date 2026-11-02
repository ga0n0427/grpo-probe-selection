# GRPO Probe Selection

Qwen3-VL-4B-Instruct에서 **Probe CE gradient와 후보의 유효 BF16 업데이트 내적**으로 QA를 선택하고, 선택한 QA로 실제 GRPO 학습을 이어가는 연구용 코드입니다.

이 저장소는 단일 GPU 실험용 harness입니다. EasyVideoR1의 GRPO advantage와 policy loss 함수 원문을 사용하지만, 전체 EasyVideoR1/Ray/FSDP trainer를 실행하는 코드는 아닙니다.

## 실험 구성

| 항목 | 설정 |
|---|---|
| 모델 | Qwen3-VL-4B-Instruct |
| 데이터 | 사용자 제공 GQA 이미지 + 짧은 QA |
| 학습 후보 | 80문항을 고정 seed로 섞어 4개씩 20묶음 |
| 선택 | 매 스텝 해당 묶음에서 점수가 가장 큰 유효 후보 1개 |
| 비교 조건 | Probe 32 / Probe 128, 각각 같은 초기 checkpoint에서 20스텝 |
| Probe | 고정된 128문항의 앞 32개가 작은 Probe; gradient는 매 스텝 재계산 |
| 후보 답변 | 현재 모델로 질문당 8개 생성, temperature=1, 최대 32토큰 |
| 정답 reward | 소문자·공백·주변 문장부호 정규화 후 exact match |
| 실제 학습 | 선택된 1문항의 8개 답변으로 on-policy GRPO 1회 |
| 평가 | 공통 Probe 128, 그 앞 32개, 별도 128문항 |
| 평가 시점 | 0·5·10·15·20스텝 CE; 0·20스텝 greedy 정확도 |
| 저장 | 각 조건의 10·20스텝 모델/optimizer를 별도 보존 |

80개 모두를 학습하는 방식은 아닙니다. 한 조건에서 80개를 검사하고 **20개를 선택해 순차적으로 업데이트**합니다. Random 비교군이나 Questioner 학습은 포함하지 않습니다.

원래 실험에서는 같은 초기 상태에서 reward 분산, nonzero GRPO gradient, nonzero BF16 업데이트를 확인한 77문항에 새 3문항을 추가했습니다. 이 저장소에는 그 이미지·질문·답변·checkpoint를 포함하지 않습니다. 사용자가 준비한 80문항을 입력받습니다.

## 계산 방식

각 스텝의 현재 모델에서 평균 Probe gradient `h`를 한 번 구합니다. 후보마다 새 rollout으로 GRPO gradient `g_i`를 계산하고, 같은 현재 master weight와 AdamW 상태에서 업데이트를 **CPU에서 미리 계산**합니다.

```text
g_i       = clip(GRPO_gradient(current_model, candidate_i), max_norm=1)
theta'_i  = AdamW_preview(theta_t, optimizer_state_t, g_i)
d_i       = BF16(theta'_i) - BF16(theta_t)
score_i   = - dot(h, d_i)
selected  = argmax(score_i)
```

선택 점수는 업데이트 후 Probe forward나 별도 평가 정답을 사용하지 않습니다. 그러나 후보별 GRPO backward와 AdamW preview 계산은 필요합니다. 비용이 없는 선택법은 아닙니다.

선택된 gradient로 실제 AdamW 상태를 갱신합니다. 이때 preview와 실제 commit의 내적 점수 및 변경 원소 수가 일치하는지 검사합니다. 다음 스텝은 갱신된 모델과 optimizer 상태에서 시작합니다. 각 조건의 모델이 달라진 뒤에는 답변과 gradient도 각자의 현재 모델에서 계산합니다.

Advantage는 **각 질문의 답변 8개 안에서** 계산합니다. 전부 정답 또는 전부 오답이면 해당 후보를 제외합니다. 한 묶음 전체가 무효이면 같은 4문항을 최대 5회 다른 seed로 다시 생성하고, 끝내 유효 업데이트가 없으면 실패 상태를 저장하고 중단합니다. 유효 후보의 점수가 모두 음수여도 가장 큰 것을 선택합니다.

## 정밀도와 optimizer

- GPU model 및 backward: BF16. 모든 parameter 학습.
- CE: logits를 FP32로 바꿔 계산. 정답 토큰 + EOS 평균 후 문항 평균.
- Probe/GRPO gradient 누적: CPU FP32. Activation checkpointing 사용.
- FP32 master weight와 AdamW momentum을 CPU에 유지하고 GPU에는 BF16으로 반영.
- AdamW: `lr=1e-6`, `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.01`, global gradient clip=1.
- EasyVideoR1 advantage: sample std (`correction=1`), `eps=1e-6`. 예전 pilot의 population std + `1e-4`와는 다릅니다.
- Policy loss: sequence mean, clipping 0.2, dual clip 3, KL=0, 새 rollout당 optimizer step 한 번.
- Math SDPA, TF32 비활성, 결정적 계산. 다른 hardware/library 조합에서 bitwise 재현은 보장하지 않습니다.

## 환경

Linux + CUDA 환경용입니다 (`fcntl`, symbolic link/hard link 사용). 측정 환경은 RTX 5090 한 장과 약 256 GiB CPU RAM입니다. **CPU offload를 사용하는 코드**이며 GPU 메모리만으로 전체 비용을 판단하면 안 됩니다. 체크포인트 하나는 원래 실험에서 약 53 GB였으므로, 초기 상태와 두 조건의 10·20스텝 상태를 보관할 충분한 저장 공간이 필요합니다.

원래 실행 환경에서 관측한 버전:

```text
torch          2.12.0 (CUDA build)
transformers   5.10.0.dev0
accelerate     1.13.0
numpy          2.4.6
Pillow         12.2.0
safetensors    0.7.0
```

개발 버전의 정확한 wheel/commit은 이 저장소에 포함하지 않습니다. 따라서 위 표만으로 환경이 완전히 고정되는 것은 아닙니다. Qwen3-VL을 지원하는 Transformers 및 GPU에 맞는 PyTorch를 먼저 설치하고 나머지를 설치합니다.

```bash
python -m pip install -r requirements.txt
```

## 입력과 실행

각 JSON은 아래 형식의 배열입니다. `image`의 상대 경로는 JSON 파일 위치를 기준으로 해석합니다.

```json
[
  {
    "id": "question-001",
    "image_id": "image-001",
    "image": "images/example.jpg",
    "prompt": "What is on the table? Answer with a short word or phrase only.",
    "target": "book"
  }
]
```

후보 80개, Probe 128개, validation 128개가 필요합니다. split 내 이미지와 질문 ID는 고유해야 하며 split 간에도 겹치면 안 됩니다. 평가용 데이터를 후보 선택이나 초기 유효 후보 선별에 사용하지 마세요.

초기 checkpoint는 다음 형태의 PyTorch 파일을 준비해야 합니다. 이름/shape는 모델의 `named_parameters()`와 일치해야 합니다.

```python
{
    "parameters": {name: cpu_fp32_master_tensor, ...},
    "moments": {
        name: {
            "step": scalar_tensor,
            "exp_avg": cpu_fp32_tensor,
            "exp_avg_sq": cpu_fp32_tensor,
        },
        ...
    },
}
```

`moments={}`로 새 optimizer에서 시작할 수도 있지만, 원래의 GRPO warmup 2스텝 이후 상태와는 다른 실험입니다. 해당 warmup checkpoint는 배포하지 않습니다. 신뢰할 수 있는 checkpoint만 사용하세요.

```bash
python prepare.py \
  --run-dir /workspace/runs/probe-selection \
  --model /models/Qwen3-VL-4B-Instruct \
  --checkpoint /checkpoints/initial.pt \
  --candidates /datasets/candidates80.json \
  --probe /datasets/probe128.json \
  --validation /datasets/validation128.json

export GRPO_RUN_DIR=/workspace/runs/probe-selection
python -u train.py
```

Probe 32 조건을 먼저 완료하고 Probe 128을 실행합니다. 실행 전에 GPU 사용 가능 여부는 직접 확인하세요. 다른 작업을 종료하는 코드는 포함하지 않습니다.

## 결과와 재개

```text
results/eval_00.json             공통 초기 평가
probe32/history.json             매 스텝 선택, 점수, 유효 후보 수, 시간, 메모리
probe32/candidates_stepNN.json    후보별 reward와 예측 점수
probe32/eval_NN.json              CE 및 시작점 대비 개선량
probe32/step_010.pt               10스텝 모델 + optimizer
probe32/step_020.pt               20스텝 모델 + optimizer
probe32/latest.pt                최신 checkpoint에 대한 hard link
probe128/...                     동일 구조
results/comparison.json          두 조건 종료 후 비교
```

동일한 `GRPO_RUN_DIR`에서 명령을 다시 실행하면 저장된 checkpoint에서 재개합니다. 첫 10스텝 이전에 중단하면 시작점부터 다시 계산하며, 10스텝 이후에는 저장되지 않은 스텝을 다시 계산합니다. JSON 로그만으로 모델 상태를 복원할 수는 없습니다. rollout 캐시는 같은 arm/step/retry에만 재사용하며 서로 다른 현재 모델 간 공유하지 않습니다.

CE 개선량은 `CE_before - CE_after`입니다. **CE가 감소했다고 답변 정확도까지 향상됐다고 해석하면 안 됩니다.** 같은 데이터로 Probe 크기나 후보 정책을 반복 선택했다면 별도 최종 평가가 필요합니다. 이 코드는 비교 실험을 실행하는 도구이며 어느 선택법의 우위를 보장하지 않습니다.

## 검증

```bash
python -m unittest discover -s tests -v
# GPU 학습 없이 loss/AdamW preview-commit 대조 검사
GRPO_RUN_DIR=/tmp/probe-selection-selftest python train.py --self-test
```

저장소 공개용 정리에서는 경로 설정과 데이터 준비를 분리하고 로컬 서버 전용 다운로드·선별 코드를 제외했습니다. 핵심 선택/업데이트 연산은 실험 코드에서 가져왔습니다. 공개용 코드에 대해 CPU self-test와 데이터 split 검사를 실행하며, 정리한 버전으로 전체 20스텝 GPU 학습을 새로 실행한 것은 아닙니다.

## 출처

`easy_loss.py`의 네 함수는 EasyVideoR1에서 함수 본문을 그대로 가져왔으며 advantage 등록 decorator만 제외했습니다. 원본 commit, 파일 hash와 라이선스는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 기록했습니다.
