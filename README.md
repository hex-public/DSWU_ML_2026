# DSWU_ML_2026

## Project Structure

- `data/`: 원본 Excel 데이터
- `csv/`: 데이터 가공 및 모델 평가 결과 CSV
- `figure/`: 데이터 분포, descriptor, 모델 비교 시각화
- `experiment/`: 최종 모델 아티팩트와 확장 실험 결과
- 각 섹션의 `archive/`: 루트나 `figures/`에 있던 이전/중복 결과물 보존본
- `secrets/`: 로컬 실행용 토큰 등 Git에 올리지 않는 개인 파일

## Main Notebook

- `template1_data_pipeline.ipynb`: 데이터 선정, descriptor 계산, 전처리, feature selection, 모델 비교 실험, 최종 모델 저장까지 포함한 제출용 노트북
- `template1_data_pipeline-Copy1.ipynb`: 동일 내용의 백업 복사본
