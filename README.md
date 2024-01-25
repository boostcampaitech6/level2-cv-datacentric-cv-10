
## 1. Project Overview
Project Notion : [Team Communication](https://www.notion.so/CV10-Data-Centric-0a7a65958da2496b8e1d5352e18b2817)


---



스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되는 경우가 있습니다. 또 주차장에 들어가면 차량 번호가 자동으로 인식되는 경우도 흔히 있습니다. 이처럼 OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.

OCR task는 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있습니다. 본 대회는 아래와 같은 특징과 제약 사항이 있습니다.

본 대회에서는 '글자 검출' task 만을 해결하게 됩니다.

본 대회는 예측 csv 파일 제출 (Evaluation) 방식을 사용합니다.

대회 기간과 task 난이도를 고려하여 코드 작성에 제약사항이 있습니다. 상세 내용은 베이스라인 코드 탭 하단의 설명을 참고해주세요.

Input : 글자가 포함된 전체 이미지

Output : bbox 좌표가 포함된 UFO Format (상세 제출 포맷은 평가 방법 탭 및 강의 5강 참조)
