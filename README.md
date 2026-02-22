# Conversational_Order_Understanding
문장을 분석하여 품목과 수량을 알려주는 시스템

# Modeling & Optimization
• Llama-3.2-1B-Instruct 모델에 LoRA 어댑터를 적용하여 경량 미세조정을 수행

• 약 3,000개 규모의 데이터셋을 기반으로 정보 추출 태스크를 학습

• 4-bit 양자화 환경을 적용하여 메모리 효율적인 학습 구조를 구현

• TRL SFTTrainer + PEFT(LoRA) 기반 생성 모델 학습 파이프라인을 설계

• sacreBLEU 기반 BLEU Score 평가를 통해 모델 응답 품질을 검증
