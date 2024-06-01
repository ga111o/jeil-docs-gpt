### jeil-docs-gpt

[youtube](https://youtu.be/ONHblOZc1iY)

---

### 코드 다운로드

[코드 페이지](https://github.com/ga111o/jeil-docs-gpt/tree/main)

위 페이지 우상단에 `<> Code` - `Download ZIP` 클릭 후 원하는 폴더에서 압축 풀기

---

### 코드 수정

`/pages/main.py`에서 코드 수정 필요.

해당 파일에서 주석처리된 부분 3개 확인.

---

### 설치 및 구동

> 2번부터는 모두 해당 디렉토리에서 진행.<br>
> Windows 명령어의 경우 다를 수도 있음. 명령어가 안되거나 할 경우 인터넷 검색으로 해결.

1. 파이썬 3.11.x 버전 확인

   - Windows: `python --version`
   - mac/Linux: `python3 --version`

2. 가상환경 세팅

   - Windows: `python venv ./env`
   - mac/Linux: `python3 venv ./env`

3. 가상환경 진입

   - Windows: `env/Scripts/activate`
   - mac/Linux: `source env/bin/activatex`

4. 의존성 다운로드

   - Windows: `pip install -r requirements.txt`
   - mac/Linux: `pip3 install -r requirements.txt`

5. 서버 구동
   - `streamlit run main.py`

---

- tips:
  - 만약 한국어 논문 등을 돌리고 싶다면 [kollama](https://huggingface.co/fiveflow/KoLlama-3-8B-Instruct) 모델을 사용하는 것을 추천함.
  - 뜬금없는 답변이 나오는 등 이상한 결과가 나온다면 더 좋은 모델을 사용하거나, 프롬프트(main.py의 line 11x의 prompt)를 수정하는 것이 필요
