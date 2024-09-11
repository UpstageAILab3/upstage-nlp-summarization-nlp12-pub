[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zHsKfIy0)
# Dialogue Summarization | 일상 대화 요약
## Team


| ![박범철](https://avatars.githubusercontent.com/u/117797850?v=4) |![김나리](https://avatars.githubusercontent.com/u/137861675?v=4) |   ![조용중](https://avatars.githubusercontent.com/u/5877567?v=4) | ![최윤설](https://avatars.githubusercontent.com/u/72685362?v=4) ||
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|                       [박범철](https://github.com/Bomtori)             | [김나리](https://github.com/narykkim)             |                      [조용중](https://github.com/paanmego)             |            [최윤설](https://github.com/developzest)             |
|                            팀장, 발표, EDA, Pre-processing, Modeling(T5-large model)                             |                            EDA, Pre-processing, Modeling(Kobart, T5-large model)                             |                 EDA, Pre-processing, Modeling(T5-large model)                   |                            EDA, Pre-processing, Modeling(Kobart, Llama3 model)                             | 
## 1. Competiton Info

### Overview

- Dialogue Summarization 경진대회는 일상 대화를 효과적으로 요약할 수 있는 모델을 구축하는 대회이다. 대화 중 요약의 필요성과 이를 통해 주관적 오류를 최소화하는 것이 목표이다. 우리는 이번 대회를 통해 대화 요약 모델 개발을 완성할 것이다.
  
### 평가 기준
- ROUGE-1-F1, ROUGE-2-F1, ROUGE-L-F1  세 가지 metric을 사용해 최종 점수 산출
- Multi-Reference Dataset의 특성에 맞춘 평가 방법: 여러 정답 요약 문장 중 3개를 비교하여 평균 점수를 계산함
- 랜덤하게 선택된 요약 문장의 평균 점수가 약 70점임


### Timeline

- 2024.08.29 ~ 2024.09.02 - 대회 시작, 데이터 EDA와 Baseline 분석
- 2024.09.03 ~ 2024.09.06 - 모델 설정, 학습 및 파인튜닝
- 2024.09.09 ~ 2024.09.11 - inference 튜닝

## 2. Data descrption

### Dataset overview
- 모든 데이터는 .csv 형식으로 제공되고 있으며, 각각의 데이터 건수는 다음과 같습니다.

- train : 12457

- dev : 499

- test : 250

- hidden-test : 249

### Data Processing

- 오탈자 수정 (철자 오류 등 수정)
- 마스킹 처리 (Special token 적용)
- 자/모음으로만 구성된 문자열 제거 (정규식 활용) 

## 3. Modeling

- Kobart ([digit82/kobart_summarization](https://huggingface.co/digit82/kobart-summarization))
- T5-Large ([lcw99/t5-large-korean-text-summary](https://huggingface.co/lcw99/t5-large-korean-text-summary))
- Llama3 ([beomi/Llama-3-Open-Ko-8B](https://huggingface.co/beomi/Llama-3-Open-Ko-8B)) 
  

## 4. Result

### Leader Board

- Public, Private 5위 

### Presentation

- [일상대화요약 발표자료 PDF 다운로드](https://github.com/UpstageAILab3/upstage-nlp-summarization-nlp12/blob/main/%EC%9D%BC%EC%83%81%EB%8C%80%ED%99%94%EC%9A%94%EC%95%BD%20%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C%20Team12.pdf)


